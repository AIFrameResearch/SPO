local num_episodes_per_iteration = 512;
local num_rollouts_per_sample = 8;
local num_dataset_samples_per_iteration = num_episodes_per_iteration / num_rollouts_per_sample;

(import 'polIter_qwen1b_spo_chain_point24.jsonnet')
+ {
  episode_generator+: {
    dataset_num_samples_per_iteration: num_dataset_samples_per_iteration,

    type: 'math_episode_generator_w_group_advantages',
    adv_method: 'grpo',
  },
  num_episodes_per_iteration: num_episodes_per_iteration,
  trainer+: {
    params+: {
      use_prob_mask: false,
    },
  },
}
