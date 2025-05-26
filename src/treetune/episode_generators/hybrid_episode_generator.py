import copy
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Tuple, Callable, List, Optional, Union
from collections import defaultdict
import random

import numpy as np
from accelerate.utils import release_memory
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

from treetune.common import Lazy
from treetune.common.vllm_server import VLLMServer
from treetune.episode_generators import EpisodeGenerator, MathEpisodeGenerator
from treetune.episode_generators.base_episode_generator import Episode
from treetune.inference_strategies import InferenceStrategy
from treetune.logging_utils import get_logger
from treetune.episode_generators.exception import NoTrainingDataException

logger = get_logger(__name__)

class OnPolicyReplayBuffer:
    def __init__(self):
        self.edges = []

    def add_edges(self, new_edges, iteration):
        # Add iteration to edges
        for edge in new_edges:
            edge["iteration"] = iteration
        self.edges.extend(new_edges)

    def get_edges(self, iteration):
        new_buffer = []
        removed_this_time = 0
        for edge in self.edges:
            if iteration - edge['iteration'] >= 8: # We ensure that the edges in the buffer is not so far, maximum 8 iteration edges in the buffer
                removed_this_time += 1
            else:
                new_buffer.append(edge)
        self.edges = new_buffer
        
        
        edges_by_question = defaultdict(list)
        for edge in self.edges:
            qid = edge['question_id']
            edges_by_question[qid].append(edge)

        sampled_edges = []
        for qid, edge_list in edges_by_question.items():
            if len(edge_list) > 32:
                sampled_edges.extend(random.sample(edge_list, 32)) # Each question we at most use 32 edges for this iteration
            else:
                sampled_edges.extend(edge_list)

        # Remove the edges we have used
        self.edges = [edge for edge in self.edges if edge not in sampled_edges]

        return sampled_edges, removed_this_time


class OffPolicyReplayBuffer:
    def __init__(self):
        self.edges = []

    def add_edges(self, new_edges, iteration):
        # Add iteration to edges
        for edge in new_edges:
            edge["iteration"] = iteration
        self.edges.extend(new_edges)

    def get_edges(self, iteration, cnt=512):
        if len(self.edges) <= cnt:
            sampled_edges = self.edges[:]
        else:
            # Randomly sample cnt edges
            sampled_edges = random.sample(self.edges, cnt)

        return sampled_edges, 0


@EpisodeGenerator.register("hybrid_episode_generator")
class HybridEpisodeGenerator(MathEpisodeGenerator):
    def __init__(
        self,
        value_estimation_inference_strategy: Lazy[InferenceStrategy],
        max_step_for_value_estimation: Optional[int] = None,
        replay_buffer_type = "on_policy",
        adv_method: Optional[str] = "rloo",
        only_adv_greater_than_zero: Optional[bool] = True,
        use_hard_estimation: Optional[bool] = False,
        use_pav: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.value_inference_strategy_lazy = value_estimation_inference_strategy
        self.max_step_for_value_estimation = max_step_for_value_estimation
        self._logger = logger
        self.replay_buffer_type = replay_buffer_type
        if self.replay_buffer_type == "on_policy":
            self.replay_buffer = OnPolicyReplayBuffer()
        elif self.replay_buffer_type == "off_policy":
            self.replay_buffer = OffPolicyReplayBuffer()
        else:
            raise ValueError(f"Unknown replay buffer type: {self.replay_buffer_type}")
        self._vllm_server_ptr = []
        self._guidance_llm_kwargs_ptr = []
        self.adv_method = adv_method
        self.only_adv_greater_than_zero = only_adv_greater_than_zero
        self.use_hard_estimation = use_hard_estimation
        self.use_pav = use_pav

    def _run_inference(
        self,
        dataset_shard: Dataset,
        vllm_init_fn: Callable[[], Tuple[VLLMServer, Dict[str, Any]]],
        vllm_cleanup_fn: Callable[[], None],
        results_root_dir: Path,
        seed: int,
        iteration: int,
    ):
        # vllm_server_ptr, guidance_llm_kwargs_ptr = [], []
        vllm_server_ptr, guidance_llm_kwargs_ptr = self._vllm_server_ptr, self._guidance_llm_kwargs_ptr
        assert len(vllm_server_ptr) == 0

        def get_vllm_server():
            if len(vllm_server_ptr) == 0:
                out = vllm_init_fn()
                vllm_server_ptr.append(out[0])
                guidance_llm_kwargs_ptr.append(out[1])
                # cache vllm server
                self._vllm_server_ptr = vllm_server_ptr
                self._guidance_llm_kwargs_ptr = guidance_llm_kwargs_ptr

            return vllm_server_ptr[0], guidance_llm_kwargs_ptr[0]

        def kill_vllm_server():
            if len(vllm_server_ptr) > 0:
                vllm_server_ptr[0].stop_server()
                vllm_server_ptr.pop()
                guidance_llm_kwargs_ptr.pop()

        def try_loading_inference_results(results_path: Path) -> Optional[Dataset]:
            logger.info(f"Always generating from scratch")
            return None

        metrics = {}

        #####################################################################################
        # Sample Trajectories from the current policy
        #####################################################################################
        traj_result_path = results_root_dir / "traj_results_ds"
        traj_infer_results = try_loading_inference_results(traj_result_path)
        if traj_infer_results is None:
            _, guidance_llm_kwargs = get_vllm_server()

            t0 = time.time()
            traj_infer_results = self._obtain_inference_results(
                inference_strategy_lazy=self.inference_strategy_lazy,
                requests_ds=dataset_shard,
                guidance_llm_kwargs=guidance_llm_kwargs,
                results_path=traj_result_path,
                seed=seed,
            )
            metrics["timing/episode_generation/traj_inference"] = time.time() - t0
            release_memory()
        try:
            metrics = {
                "parse_failed": [],
                "once_hit": [],
                "is_unfinished_response": [],
                "is_truncated_response": [],
                "finish_reason_is_length": [],
                "trajectory_bleu": [],
                "num_unique_responses": [],
                "logprobs_mismatch": [],
                "num_cutpoints": []
            }
            trajectories = []

            all_edges = []

            # Add all the edges
            for idx, item in enumerate(traj_infer_results):
                # noinspection PyTypeChecker
                tree = json.loads(item["_treetune__reasoning_tree"])
                edges = self.extract_edges_from_tree(tree, 
                                                     adv_method=self.adv_method, 
                                                     only_adv_greater_than_zero=self.only_adv_greater_than_zero,
                                                     use_hard_estimation=self.use_hard_estimation)

                all_edges.extend(edges)

            new_all_edges = []

            for edge in all_edges:
                query_text = edge["query_text"]
                response_text = edge["response_text"]

                if len(edge["response_text"]) == 0:
                    continue # In some rare cases, the response text may be empty, we simply discard these edges
                # metrics["finish_reason_is_length"].append(finish_reason == "length")


                query_token_ids, response_token_ids, offsets = (
                    self._tokenize_trajectory(
                        {"query_text": query_text, "response_text": response_text},
                        return_offsets=True,
                    )
                )

                edge["query_token_ids"] = query_token_ids
                edge["response_token_ids"] = response_token_ids
                edge["offsets"] = offsets
                edge["process_idx"] = self.distributed_state.process_index

                new_all_edges.append(edge)

            # if len(all_edges) == 0:
            #     raise NoTrainingDataException()

            # Now we got all the edges from the data_shard in this process, we add logprob to these edges

            
            kill_vllm_server()
            release_memory()
            vllm_cleanup_fn()
            release_memory()

            


            new_all_edges = self._add_logprobs_to_edges(new_all_edges, results_root_dir)

            self.replay_buffer.add_edges(new_all_edges, iteration)

            # Get edges for this iteration
            edges_this_iteration, discard_cnt = self.replay_buffer.get_edges(iteration)
            # assert discard_cnt == 0
            self._cloud_log({"replay_buffer/discard_cnt": discard_cnt, 
                                    "replay_buffer/samples": len(edges_this_iteration),  
                                    "iteration": iteration})

            # Create trajectory
            for edge in edges_this_iteration:
                query_text = edge["query_text"]
                response_text = edge["response_text"]
                query_token_ids = edge["query_token_ids"]
                response_token_ids = edge["response_token_ids"]
                offsets = edge["offsets"]
                advantage = edge["advantage"]
                prover_advantage = edge["prover_advantage"]
                reward = edge["reward"]
                leaf = edge["leaf"]
                instance = edge["instance"]
                logps = edge["actor_shifted_log_probs"]
                value = edge["value"]

                if len(edge["response_text"]) == 0:
                    continue # In some rare cases, the response text may be empty, we simply discard these edges
                # metrics["finish_reason_is_length"].append(finish_reason == "length")


                # query_token_ids, response_token_ids, offsets = (
                #     self._tokenize_trajectory(
                #         {"query_text": query_text, "response_text": response_text},
                #         return_offsets=True,
                #     )
                # )

                # noinspection PyUnresolvedReferences
                data_instance = {
                    k: v for k, v in instance.items() if not k.startswith("_treetune")
                }

                trajectories.append(
                    {
                        "instance_idx": idx,
                        "data_instance": data_instance,
                        "query_text": query_text,
                        "response_text": response_text,
                        # "full_text": full_text,
                        "query_token_ids": query_token_ids,
                        "response_token_ids": response_token_ids,
                        "offsets": offsets,
                        "advantage": advantage,
                        "prover_advantage": prover_advantage,
                        "reward": reward,
                        "leaf": leaf,
                        "logps": logps,
                        "value": value,
                        # "is_unfinished_response": is_unfinished_response,
                        # "steps": steps,
                        # "step_indices": indices,
                        # "step_rewards": step_rewards,
                        # "values": [None] * (len(steps) + 1),  # +1 for the query state
                        # "response_prob": response_prob,
                        # "cutpoints": cutpoints,
                        # "cutpoints_values": cutpoints_values,
                        # "cutpoints_values_std": cutpoints_values_std,
                        # "process_idx": self.distributed_state.process_index,
                        # "response_logprobs_vllm": response_logprob,
                        # "response_probs_vllm": response_prob,
                        # "tokens_vllm": tokens
                    }
                )

                # if len(all_scores) > 0:
                #     once_hit = any([r == 1.0 for r in all_scores])
                #     metrics["once_hit"].append(float(once_hit))

                # if len(all_responses) > 1:
                #     metrics["num_unique_responses"].append(len(set(all_responses)))
                #     if self._bleu_metric is not None:
                #         bleu = self._avg_bleu_of_pairs_of_response(all_responses)
                #         metrics["trajectory_bleu"].append(bleu)

            # noinspection DuplicatedCode
            metrics = {
                k: sum(values) / len(values)
                for k, values in metrics.items()
                if len(values) > 0
            }
            if len(metrics) > 0:
                logs = {f"episodes_metric/{k}": v for k, v in metrics.items()}
                self._cloud_log({**logs, "train/global_iteration": iteration})

            # trajectories = self._create_trajectories(traj_infer_results, 
            #                                          iteration, 
            #                                          results_root_dir)
        except NoTrainingDataException as e:
            # kill_vllm_server()
            # release_memory()
            # vllm_cleanup_fn()
            # release_memory()
            raise   
        # Remove the vllm server, so the next iteration will create a new one
        # self._vllm_server_ptr = []
        # self._guidance_llm_kwargs_ptr = []
        #####################################################################################
        # Estimate the value of each state in the trajectories using Monte Carlo rollouts
        #####################################################################################
        # (
        #     unique_requests,
        #     all_requests,
        #     all_reqs_to_unique_key,
        #     trajectories,
        # ) = self._create_value_estimation_requests(trajectories, results_root_dir, iteration)
        # val_est_result_path = (
        #     results_root_dir.parent / "unique_value_estimation_result_ds"
        # )
        # unique_results = try_loading_inference_results(val_est_result_path)
        # if unique_results is None:
        #     _, guidance_llm_kwargs = get_vllm_server()

        #     t0 = time.time()
        #     self._obtain_inference_results(
        #         inference_strategy_lazy=self.value_inference_strategy_lazy,
        #         requests_ds=unique_requests,
        #         guidance_llm_kwargs=guidance_llm_kwargs,
        #         results_path=results_root_dir / "value_estimation_result_ds_temp",
        #         seed=seed + 1,
        #     )
        #     metrics["timing/episode_generation/value_estimation"] = time.time() - t0

        #     # Merge all results into a single file
        #     self.distributed_state.wait_for_everyone()
        #     if self.distributed_state.is_local_main_process:
        #         shard_paths = list(
        #             results_root_dir.parent.glob(
        #                 "process_*/value_estimation_result_ds_temp"
        #             )
        #         )
        #         shard_paths.sort(key=lambda x: int(x.parent.name.split("process_")[-1]))
        #         merged = concatenate_datasets(
        #             [Dataset.load_from_disk(str(p)) for p in shard_paths]
        #         )
        #         merged.save_to_disk(str(val_est_result_path))
        #         logger.info(f"Created {len(merged)} value estimation results in total.")
        #         del merged
        #         release_memory()

        #     self.distributed_state.wait_for_everyone()
        #     unique_results = Dataset.load_from_disk(str(val_est_result_path))

        # kill_vllm_server()
        # release_memory()
        # vllm_cleanup_fn()
        # release_memory()

        # if len(metrics) > 0:
        #     self._cloud_log(metrics)

        # # Distribute the value estimation results back according to the process index
        # process_idx = self.distributed_state.process_index
        # num_proc = self.distributed_state.num_processes
        # unique_results = unique_results.filter(
        #     lambda x: x["process_idx"] == process_idx,
        #     suffix_template=(
        #         f"_dist{process_idx}_of_{num_proc}__" + "{rank:05d}_of_{num_proc:05d}"
        #     ),
        #     num_proc=None,  # No multiprocessing
        # )
        # assert len(unique_results) == len(set(all_reqs_to_unique_key))

        # # Create a map from unique _treetune__idx to the result index
        # # noinspection PyTypeChecker
        # unique_key_to_result_idx = {
        #     (res["_treetune__idx"], res["process_idx"]): idx
        #     for idx, res in enumerate(unique_results)
        # }
        # assert len(unique_key_to_result_idx) == len(unique_results)

        # # Update all requests with the value estimation results
        # all_results = []
        # for req, unique_key in zip(all_requests, all_reqs_to_unique_key):
        #     result_idx = unique_key_to_result_idx[unique_key]
        #     result = unique_results[result_idx]
        #     assert req["query"] == result["query"]
        #     req.update(
        #         {
        #             k: v
        #             for k, v in result.items()
        #             if k.startswith("_treetune__") and k != "_treetune__idx"
        #         }
        #     )
        #     all_results.append(req)
        # all_results = Dataset.from_list(all_results)
        # all_results.save_to_disk(str(results_root_dir / "value_estimation_results_ds"))
        # del all_results
        # del unique_results
        # release_memory()

        episodes = self._create_episodes(
            # traj_infer_results=traj_infer_results,
            trajectories=trajectories,
            # value_estimation_results=Dataset.load_from_disk(
            #     str(results_root_dir / "value_estimation_results_ds")
            # ),
            iteration=iteration,
            results_root_dir=results_root_dir,
        )

        return episodes

    def _create_episodes(
        self,
        # traj_infer_results: Dataset,
        trajectories: List[Dict[str, Any]],
        # value_estimation_results: Dataset,
        iteration: int,
        results_root_dir: Optional[Path] = None,
    ) -> List[Episode]:
        # Update episodes with the value estimates
        # trajectories = self._update_trajectories_w_values(
        #     traj_infer_results=traj_infer_results,
        #     trajectories=trajectories,
        #     value_estimation_results=value_estimation_results,
        #     iteration=iteration,
        # )

        # metrics = {
        #     # "num_reasoning_steps": [],
        #     "is_unfinished_response": [],
        #     # "values": [],
        # }
        episodes = []
        
        for traj in trajectories:
            values, advantages, prover_advantages = self._compute_token_advantages(traj)
            

            if self.use_pav:
                pav_advantages = [None] * len(advantages)
                for i in range(len(advantages)):
                    pav_advantages[i] = values[i] + 5 * prover_advantages[i]
                advantages = pav_advantages

            episode = Episode(
                question_idx=traj["instance_idx"],
                query_text=traj["query_text"],
                response_text=traj["response_text"],
                query_token_ids=traj["query_token_ids"],
                response_token_ids=traj["response_token_ids"],
                scores=traj["reward"],
                advantages=advantages,
                values=values,
                leaf=traj["leaf"],
                actor_shifted_log_probs=traj["logps"]
                # values_std=values_std,
                # probs=traj["response_prob"]
            )
            episodes.append(episode)

            # metrics["num_reasoning_steps"].append(len(traj["steps"]))
        #     metrics["is_unfinished_response"].append(traj["is_unfinished_response"])
        #     # metrics["values"].extend(traj["values"])

        # if results_root_dir is not None:
        #     with open(results_root_dir / f"trajectories.pkl", "wb") as f:
        #         pickle.dump(trajectories, f)

        # if "is_unfinished_response" in metrics:
        #     metrics["is_unfinished_response"] = sum(
        #         metrics["is_unfinished_response"]
        #     ) / len(metrics["is_unfinished_response"])

        # if "num_reasoning_steps" in metrics:
        #     num_reasoning_steps = np.array(metrics.pop("num_reasoning_steps"))
        #     metrics["num_reasoning_steps/dist"] = num_reasoning_steps
        #     metrics["num_reasoning_steps/mean"] = np.mean(num_reasoning_steps)

        # if "values" in metrics:
        #     values = np.array(metrics.pop("values"))
        #     metrics["mc_values/dist"] = values
        #     metrics["mc_values/mean"] = np.mean(values)

        # if len(metrics) > 0:
        #     logs = {f"episodes_metric/{k}": v for k, v in metrics.items()}
        #     self._cloud_log({**logs, "train/global_iteration": iteration})

        return episodes

    def _update_trajectories_w_values(
        self,
        traj_infer_results: Dataset,
        trajectories: List[Dict[str, Any]],
        value_estimation_results: Dataset,
        iteration: int,
    ) -> List[Dict[str, Any]]:
        metrics = {
            "mc_roll_trunc_frac": [],
            "mc_roll_std": [],
            "mc_avg_num_rolls": [],
            "mc_avg_unique_rolls_frac": [],
            "mc_ci_length": [],
        }
        for res in tqdm(
            value_estimation_results, desc="Updating trajectories with values"
        ):
            process_idx = res["process_idx"]
            if process_idx != self.distributed_state.process_index:
                continue
            data_instance = traj_infer_results[res["instance_idx"]]

            traj_idx = res["traj_idx"]
            value_idx = res["value_idx"]
            traj = trajectories[traj_idx]

            # Perform sanity checks
            # reconst_query = "".join(
            #     ([traj["query_text"]] + traj["steps"])[: value_idx + 1]
            # )
            # assert reconst_query == res["query"]

            reconst_query = traj["query_text"] + self.tokenizer.decode(traj["response_token_ids"][:traj["cutpoints"][value_idx] + 1])
            assert reconst_query == res["query"]

            infer_tree = json.loads(res["_treetune__reasoning_tree"])
            assert reconst_query == infer_tree["text"]
            assert data_instance["query"] in res["query"]

            truncations = [
                c["finish_reason"] == "length" for c in infer_tree["children"]
            ]
            metrics["mc_roll_trunc_frac"].append(sum(truncations) / len(truncations))
            metrics["mc_avg_num_rolls"].append(len(infer_tree["children"]))

            unique_rolls = len(set(c["answer"] for c in infer_tree["children"]))
            metrics["mc_avg_unique_rolls_frac"].append(
                unique_rolls / len(infer_tree["children"])
            )

            if "ci_length" in infer_tree:
                metrics["mc_ci_length"].append(infer_tree["ci_length"])

            value, value_std, rewards = self._compute_mc_value(
                query=res["query"],
                value_estimation_result=res,
                data_instance=data_instance,
                return_rewards=True,
            )

            if rewards is not None and len(rewards) > 1:
                metrics["mc_roll_std"].append(np.std(rewards))

            # Update the value in the trajectory
            # assert trajectories[traj_idx]["values"][value_idx] is None
            # trajectories[traj_idx]["values"][value_idx] = value

            assert trajectories[traj_idx]["cutpoints_values"][value_idx] is None
            trajectories[traj_idx]["cutpoints_values"][value_idx] = value
            trajectories[traj_idx]["cutpoints_values_std"][value_idx] = value_std

        # noinspection DuplicatedCode
        metrics = {
            k: sum(values) / len(values)
            for k, values in metrics.items()
            if len(values) > 0
        }
        if len(metrics) > 0:
            logs = {f"episodes_metric/{k}": v for k, v in metrics.items()}
            self._cloud_log({**logs, "train/global_iteration": iteration})

        return trajectories

    def _compute_token_advantages(
        self,
        trajectory: Dict[str, Any],
    ) -> List[float]:
        query_text = trajectory["query_text"]
        response_text = trajectory["response_text"]
        offsets = trajectory["offsets"]
        query_token_ids = trajectory["query_token_ids"]
        response_token_ids = trajectory["response_token_ids"]

        # reasoning_steps = trajectory["steps"]
        # step_indices = trajectory["step_indices"]
        # assert self.reasoning_step_delimiter.join(reasoning_steps) == response_text

        # advantages = self._compute_step_advantages(trajectory)

        # # Discard the EOS to match the initial response text
        # has_eos = response_token_ids[-1] == self.tokenizer.eos_token_id
        # if has_eos:
        #     response_token_ids = response_token_ids[:-1]

        # # Discard the BOS to match the initial query text
        # has_bos = query_token_ids[0] == self.tokenizer.bos_token_id
        # if has_bos:
        #     query_token_ids = query_token_ids[1:]

        # assert len(offsets) == (len(query_token_ids) + len(response_token_ids))

        # # Map advantages computed for reasoning steps to characters in the response text
        # char_advantages = np.ones(len(response_text)) * -7777777
        # for i, (start, end) in enumerate(zip(step_indices[:-1], step_indices[1:])):
        #     for j in range(start, end):
        #         char_advantages[j] = advantages[i]
        # assert np.all(char_advantages != -7777777)

        # # Find the advantage of response tokens from the advantage of its characters
        # token_advantages = [None] * len(response_token_ids)
        # for i in range(len(token_advantages)):
        #     start_char_pos_of_token = offsets[i + len(query_token_ids)][0]
        #     start_char_pos_of_token -= len(query_text)
        #     token_advantages[i] = char_advantages[start_char_pos_of_token]

        # if has_eos:
        #     token_advantages.append(token_advantages[-1])

        token_values = [trajectory["value"]] * len(response_token_ids)
        token_values_std = [None] * len(response_token_ids)
        advantages = [trajectory["advantage"]] * len(response_token_ids)
        prover_advantages = [trajectory["prover_advantage"]] * len(response_token_ids)

        # score = trajectory["score"]
        # token_values[-1] = score
        # token_values_std[-1] = 0

        # assert trajectory["cutpoints"][0] == 0
        # base = trajectory["cutpoints_values"][0]

        # base = None
        # cutpoints = trajectory["cutpoints"]
        # cutpoints_values = trajectory["cutpoints_values"]
        # cutpoints_values_std = trajectory["cutpoints_values_std"]
        # response_prob = trajectory["response_prob"]

        # if len(cutpoints) == 0:
        #     token_values = [score] * len(response_token_ids)
        #     advantages = [0] * len(response_token_ids)
        #     return token_values, advantages

        # if cutpoints[0] == -1:
        #     base = cutpoints_values[0]
        #     cutpoints = cutpoints[1:]
        #     cutpoints_values = cutpoints_values[1:]
        #     cutpoints_values_std = cutpoints_values_std[1:]
            
        # for idx, value, value_std in zip(cutpoints, cutpoints_values, cutpoints_values_std):
        #     # assert value is not None
        #     token_values[idx] = value
        #     token_values_std[idx] = value_std

        # for i in range(len(token_values)):
        #     if token_values[i] is not None:
        #         if base == None:
        #             advantages[i] = 0
        #         else:
        #             advantages[i] = token_values[i] - base
        #         base = token_values[i]

        # for i in range(len(token_values) - 1, -1, -1):
        #     if advantages[i] is None:
        #         advantages[i] = advantages[i + 1]

        # for i in range(len(token_values)):
        #     if response_prob[i] > 0.9:
        #         advantages[i] = 0


        # for i in range(len(token_values) - 2, -1, -1):
        #     if token_values[i] == None:
        #         token_values[i] = token_values[i + 1]
        #         token_values_std[i] = token_values_std[i + 1]

        # for i in range(len(token_values) - 1, 0, -1):
        #     advantages[i] = token_values[i] - token_values[i - 1]
        
        # if base is not None:
        #     advantages[0] = token_values[0] - base
        # else:
        #     advantages[0] = 0
        # first_conclusion_idx = response_text.find('###')
        # second_conclusion_idx = None
        # if first_conclusion_idx != -1:
        #     second_conclusion_idx = response_text.find('###', first_conclusion_idx + 1)


        # pit = trajectory["pit"]
        # if pit == -1:
        #     for idx in range(len(token_values)):
        #         token_values[idx] = 0
        # else:
        #     for idx in range(len(token_values)):
        #         if second_conclusion_idx and idx >= second_conclusion_idx:
        #             token_values[idx] = -2 # We should try the best to avoid this
        #         elif idx < pit:
        #             token_values[idx] = 0
        #         elif idx == pit and idx > 0: # > 0 because we don't want to penalize the first token
        #             assert trajectory["pit_value"] is not None
        #             token_values[idx] = trajectory["pit_value"]
        #         else:
        #             token_values[idx] = 0

        # noinspection PyTypeChecker
        return token_values, advantages, prover_advantages

    def _compute_step_advantages(self, trajectory: Dict[str, Any]):
        step_rewards = trajectory["step_rewards"]
        values = trajectory["values"]

        # The value of the final/terminating state is by definition 0
        assert values[-1] is None
        # noinspection DuplicatedCode
        values[-1] = 0.0

        # Fill in the missing values from the end
        for i in range(len(values) - 2, -1, -1):
            if values[i] is not None:
                break
            values[i] = step_rewards[i] + values[i + 1]

        # noinspection DuplicatedCode
        assert all(v is not None for v in values)

        advantages = [None] * len(step_rewards)
        assert len(advantages) == len(values) - 1
        assert len(advantages) == len(step_rewards)
        for i in range(len(advantages)):
            advantages[i] = step_rewards[i] + values[i + 1] - values[i]

        return advantages

    def _compute_mc_value(
        self,
        *,
        query: str = None,
        value_estimation_result: Dict[str, Any] = None,
        data_instance: Dict[str, Any] = None,
        return_rewards: bool = False,
    ) -> Union[float, Tuple[float, List[float]]]:
        # noinspection DuplicatedCode
        tree = json.loads(value_estimation_result["_treetune__reasoning_tree"])
        rollouts = [(c["answer"], c["finish_reason"]) for c in tree["children"]]

        rewards = [
            (
                self.reward_function(query, rol, data_instance)[0]
                if finish_reason != "length"
                else self.reward_function.get_unfinished_response_penalty()
            )
            for rol, finish_reason in rollouts
        ]

        if len(rewards) == 0:
            mc_value = 0.0
            mc_std = -1 # Set to -1 to indentify this
        else:
            mc_value = np.mean(rewards)
            mc_std = np.std(rewards)

        if return_rewards:
            return mc_value, mc_std, rewards

        return mc_value, mc_std

    def _rollout_eval_callback(
        self,
        *,
        query: str,
        rollout: str,
        finish_reason: str,
        request_object: Dict[str, Any],
    ) -> float:
        if finish_reason == "length":
            return self.reward_function.get_unfinished_response_penalty()

        data_instance = request_object["data_instance"]
        assert data_instance["problem"] in query
        return self.reward_function(query, rollout, data_instance)[0]

    # def _create_trajectories(
    #     self,
    #     inference_results: Dataset,
    #     iteration: int,
    #     results_root_dir: Path
    # ) -> List[Dict[str, Any]]:
    #     metrics = {
    #         "parse_failed": [],
    #         "once_hit": [],
    #         "is_unfinished_response": [],
    #         "is_truncated_response": [],
    #         "finish_reason_is_length": [],
    #         "trajectory_bleu": [],
    #         "num_unique_responses": [],
    #         "logprobs_mismatch": [],
    #         "num_cutpoints": []
    #     }
    #     trajectories = []

    #     all_edges = []

    #     # Add all the edges
    #     for idx, item in enumerate(inference_results):
    #         # noinspection PyTypeChecker
    #         tree = json.loads(item["_treetune__reasoning_tree"])
    #         edges = self.extract_edges_from_tree(tree, adv_method=self.adv_method)

    #         all_edges.extend(edges)

    #     new_all_edges = []

    #     for edge in all_edges:
    #         query_text = edge["query_text"]
    #         response_text = edge["response_text"]

    #         if len(edge["response_text"]) == 0:
    #             continue # In some rare cases, the response text may be empty, we simply discard these edges
    #         # metrics["finish_reason_is_length"].append(finish_reason == "length")


    #         query_token_ids, response_token_ids, offsets = (
    #             self._tokenize_trajectory(
    #                 {"query_text": query_text, "response_text": response_text},
    #                 return_offsets=True,
    #             )
    #         )

    #         edge["query_token_ids"] = query_token_ids
    #         edge["response_token_ids"] = response_token_ids
    #         edge["offsets"] = offsets
    #         edge["process_idx"] = self.distributed_state.process_index

    #         new_all_edges.append(edge)

    #     # if len(all_edges) == 0:
    #     #     raise NoTrainingDataException()

    #     # Now we got all the edges from the data_shard in this process, we add logprob to these edges
    #     new_all_edges = self._add_logprobs_to_edges(new_all_edges, results_root_dir)

    #     self.replay_buffer.add_edges(new_all_edges, iteration)

    #     # Get edges for this iteration
    #     edges_this_iteration, discard_cnt = self.replay_buffer.get_edges(iteration)
    #     # assert discard_cnt == 0
    #     self._cloud_log({"replay_buffer/discard_cnt": discard_cnt, 
    #                             "replay_buffer/samples": len(edges_this_iteration),  
    #                             "iteration": iteration})

    #     # Create trajectory
    #     for edge in edges_this_iteration:
    #         query_text = edge["query_text"]
    #         response_text = edge["response_text"]
    #         query_token_ids = edge["query_token_ids"]
    #         response_token_ids = edge["response_token_ids"]
    #         offsets = edge["offsets"]
    #         advantage = edge["advantage"]
    #         reward = edge["reward"]
    #         leaf = edge["leaf"]
    #         instance = edge["instance"]
    #         logps = edge["actor_shifted_log_probs"]

    #         if len(edge["response_text"]) == 0:
    #             continue # In some rare cases, the response text may be empty, we simply discard these edges
    #         # metrics["finish_reason_is_length"].append(finish_reason == "length")


    #         # query_token_ids, response_token_ids, offsets = (
    #         #     self._tokenize_trajectory(
    #         #         {"query_text": query_text, "response_text": response_text},
    #         #         return_offsets=True,
    #         #     )
    #         # )

    #         # noinspection PyUnresolvedReferences
    #         data_instance = {
    #             k: v for k, v in instance.items() if not k.startswith("_treetune")
    #         }

    #         trajectories.append(
    #             {
    #                 "instance_idx": idx,
    #                 "data_instance": data_instance,
    #                 "query_text": query_text,
    #                 "response_text": response_text,
    #                 # "full_text": full_text,
    #                 "query_token_ids": query_token_ids,
    #                 "response_token_ids": response_token_ids,
    #                 "offsets": offsets,
    #                 "advantage": advantage,
    #                 "reward": reward,
    #                 "leaf": leaf,
    #                 "logps": logps
    #                 # "is_unfinished_response": is_unfinished_response,
    #                 # "steps": steps,
    #                 # "step_indices": indices,
    #                 # "step_rewards": step_rewards,
    #                 # "values": [None] * (len(steps) + 1),  # +1 for the query state
    #                 # "response_prob": response_prob,
    #                 # "cutpoints": cutpoints,
    #                 # "cutpoints_values": cutpoints_values,
    #                 # "cutpoints_values_std": cutpoints_values_std,
    #                 # "process_idx": self.distributed_state.process_index,
    #                 # "response_logprobs_vllm": response_logprob,
    #                 # "response_probs_vllm": response_prob,
    #                 # "tokens_vllm": tokens
    #             }
    #         )

    #         # if len(all_scores) > 0:
    #         #     once_hit = any([r == 1.0 for r in all_scores])
    #         #     metrics["once_hit"].append(float(once_hit))

    #         # if len(all_responses) > 1:
    #         #     metrics["num_unique_responses"].append(len(set(all_responses)))
    #         #     if self._bleu_metric is not None:
    #         #         bleu = self._avg_bleu_of_pairs_of_response(all_responses)
    #         #         metrics["trajectory_bleu"].append(bleu)

    #     # noinspection DuplicatedCode
    #     metrics = {
    #         k: sum(values) / len(values)
    #         for k, values in metrics.items()
    #         if len(values) > 0
    #     }
    #     if len(metrics) > 0:
    #         logs = {f"episodes_metric/{k}": v for k, v in metrics.items()}
    #         self._cloud_log({**logs, "train/global_iteration": iteration})

    #     return trajectories
    
    def _add_logprobs_to_edges(self, edges: List[Dict[str, Any]], results_root_dir: Path):
        process_idx = self.distributed_state.process_index

        if len(edges) > 0:
            edges_ds = Dataset.from_list(edges)
            edges_ds.save_to_disk(str(results_root_dir / "edges_ds_temp")) # Results_root_dir has iteration in it
            del edges_ds
            release_memory()

        self.distributed_state.wait_for_everyone() # Wait for all processes to finish saving the edges

        shard_paths = list(results_root_dir.parent.glob("process_*/edges_ds_temp"))
        if len(shard_paths) == 0:
            raise NoTrainingDataException()

        # Merge the trajectories from all processes
        if self.distributed_state.is_local_main_process:
            # shard_paths = list(results_root_dir.parent.glob("process_*/edges_ds_temp"))
            # if len(shard_paths) == 0:
            #     raise NoTrainingDataException()
            shard_paths.sort(key=lambda x: int(x.parent.name.split("process_")[-1]))
            merged = concatenate_datasets(
                [Dataset.load_from_disk(str(p)) for p in shard_paths]
            )
            merged.save_to_disk(str(results_root_dir.parent / "merged_edges"))
            logger.info(f"Created {len(merged)} edges in total.")
            del merged
            release_memory()

        self.distributed_state.wait_for_everyone() # Wait for the main process to finish merging the trajectories

        edges_ds = Dataset.load_from_disk(
            str(results_root_dir.parent / "merged_edges")
        ) # Now all the processes have the whole trajectory dataset

        edges_ds = self.trainer.get_episodes_w_actor_logps(edges_ds) # All processes will get a shard of the whole dataset and do inference, all the process still have the whole trajectory dataset

        edges_ds = edges_ds.filter(lambda example: example["process_idx"] == process_idx) # We can filter out the data that is not belong to this process

        edges = edges_ds.to_list()

        del edges_ds
        release_memory()

        return edges

    def _create_cutpoint_query(self, traj: Dict[str, Any], cutpoint: int) -> str:
        query_text = traj["query_text"]
        full_text = traj["full_text"]

        response_up_to_cutpoint = self.tokenizer.decode(traj["response_token_ids"][:cutpoint + 1])
        request_query = query_text + response_up_to_cutpoint
        # assert full_text.startswith(request_query), f"{full_text} != {request_query}"

        return request_query

    # noinspection DuplicatedCode
    def _create_step_query(self, traj: Dict[str, Any], step_idx: int) -> str:
        query_text = traj["query_text"]
        response_text = traj["response_text"]
        full_text = traj["full_text"]

        # Here is the format:
        # indices = [0, ..., len(response_text)]
        #
        # e.g.
        #   step #0: response_text[0:indices[1]],
        #   step #1: response_text[indices[1]:indices[2]],
        #   ...
        indices = traj["step_indices"]
        response_up_to_step = response_text[: indices[step_idx + 1]]

        request_query = query_text + response_up_to_step
        assert full_text.startswith(request_query), f"{full_text} != {request_query}"

        return request_query

    def _create_step_response(self, traj: Dict[str, Any], step_idx: int) -> str:
        response_text = traj["response_text"]
        full_text = traj["full_text"]

        indices = traj["step_indices"]
        response_from_step = response_text[indices[step_idx + 1] :]

        assert full_text.endswith(
            response_from_step
        ), f"{full_text} != {response_from_step}"

        return response_from_step

    def _truncate_response_to_max_length(self, query: str, response: str) -> str:
        if self.max_sequence_length is None:
            return response

        query_token_ids, response_token_ids = self._tokenize_trajectory(
            {"query_text": query, "response_text": response},
        )

        seq_len = len(query_token_ids) + len(response_token_ids)
        if seq_len <= self.max_sequence_length:
            return response

        # Truncate the response token ids
        # Create a new response text
        response_token_ids = response_token_ids[
            : self.max_sequence_length - len(query_token_ids)
        ]

        if response_token_ids[-1] == self.tokenizer.eos_token_id:
            response_token_ids.pop()

        if query_token_ids[0] == self.tokenizer.bos_token_id:
            query_token_ids = query_token_ids[1:]

        new_episode = self.tokenizer.decode(
            query_token_ids + response_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        new_response = new_episode[len(query) :]

        return new_response

    def _split_solution_into_reasoning_steps(self, solution: str) -> List[str]:
        raise NotImplementedError("This method should not be used in this class.")

    def _obtain_inference_results(
        self,
        inference_strategy_lazy: Lazy[InferenceStrategy],
        requests_ds: Dataset,
        guidance_llm_kwargs: Dict,
        results_path: Path,
        seed: int,
    ) -> Dataset:
        # Sanity check
        request_ids = requests_ds["_treetune__idx"]
        assert len(request_ids) == len(set(request_ids)), "Duplicate request ids found."

        # Initialize the inference strategy with the vLLM server URL
        inference_strategy_lazy = copy.deepcopy(inference_strategy_lazy)
        # noinspection PyProtectedMember
        inference_strategy_lazy._params["guidance_llm"].update(guidance_llm_kwargs)
        infer_strategy = inference_strategy_lazy.construct(
            result_dir=results_path.parent / f"{results_path.stem}.infer_strategy",
            seed=seed,
            cloud_logger=None,
            reward_function=self.reward_function,
            log_level=(
                logging.WARNING
                if not self.distributed_state.is_local_main_process
                else None
            ),
        )
        if hasattr(infer_strategy, "node_expander"):
            if hasattr(infer_strategy.node_expander, "set_rollout_eval_callback"):
                infer_strategy.node_expander.set_rollout_eval_callback(
                    self._rollout_eval_callback
                )

        results = infer_strategy.generate(requests_ds)
        results.save_to_disk(str(results_path))
        del results
        del infer_strategy
        del inference_strategy_lazy
        release_memory()

        return Dataset.load_from_disk(str(results_path))

    def _generate_episodes(
        self, inference_results: Dataset, iteration: int
    ) -> List[Union[Dict[str, Any], Episode]]:
        # This is a dummy method.
        # We generate episodes in the `_run_inference` method.
        # i.e., `inference_results` already contains the episodes.
        # noinspection PyTypeChecker
        return inference_results
