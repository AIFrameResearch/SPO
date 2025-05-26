from __future__ import annotations
import math
import random
from datasets import Dataset
from treetune.logging_utils import get_logger
from treetune.exceptions import EarlyStop
from treetune.question_samplers.base_question_sampler import QuestionSampler

logger = get_logger(__name__)


@QuestionSampler.register("sticky")
class StickyQuestionSampler(QuestionSampler):
    def __init__(self, 
                 total_num_questions: int,
                 random_proportion: float = 0.25,
                 satisfaction_threshold: float = 0.8,
                 num_rollouts_threshold: int = 1024,
                 early_stop=True) -> None:
        """
        Params:
            num_questions (int):
                The number of questions in the trainset
            random_proportion (float):
                The proportion of random problems
            satisfaction_threshold (float): 
                The threshold for pass@1 that determines if a question is "satisfied"
            num_rollouts_threshold (int): 
                The maximum number of rollouts allowed per question before giving up
            early_stop (bool): 
                Whether the training can stop early once questions are satisfied.
            
        """
        self.total_num_questions = total_num_questions

        assert random_proportion >= 0 and random_proportion <= 1, "random_proportion must be greater than or equal to 0 and less than or equal to 1"
        self.random_proportion = random_proportion

        self.satisfaction_threshold = satisfaction_threshold
        self.num_rollouts_threshold = num_rollouts_threshold
        self.early_stop = early_stop

        self.all_questions = list(range(self.total_num_questions))

        # Initialize the question metrics
        self.question_metrics = {}
        for idx in range(total_num_questions):
            self.question_metrics[idx] = {"num_rollouts": 0, "correct": 0} # Initialize metrics for each question
        
        self.satisfied_questions = [] # Questions whose pass@1 have exceeded the satisfaction threshold
        self.given_up_questions = [] # Questions that have reached the num_rollouts_threshold but not exceed the satisfaction threshold

        self.questions_to_train = [] # Questions twe focus on improving pass@1
    
    def select_question(self, sample_num: int) -> list[int]:
        """
        Half of the questions are randomly selected to avoid overfitting,
        while the other half are chosen from the previous iteration.
        """
        random_sample_size = math.floor(sample_num * self.random_proportion)

        # Randomly select the first half to prevent the model from overfitting on the questions_to_train
        # Additionally, there's a chance that questions already in the satisfied questions list might be sampled again
        # In such cases, we will reevaluate their pass@1 score, and if it's below the threshold, we'll remove them from the satisfied questions list
        random_questions = random.choices(self.all_questions, k=random_sample_size)

        # Reload the questions from the previous training that did not meet the satisfaction threshold
        selected_from_previous = [
            idx for idx in self.questions_to_train
            if idx not in self.satisfied_questions and idx not in self.given_up_questions
        ]

        # If there are new slots, randomly sample unsatisfied questions to fill them, these questions will be trained to reach the satisfaction_threshold
        remaining_sample_size = sample_num - len(random_questions) - len(selected_from_previous)

        new_questions_to_train = []

        if remaining_sample_size > 0:
            question_sampling_pool = list(set(self.all_questions) - set(self.satisfied_questions) - set(self.given_up_questions))
            if len(question_sampling_pool) <= 0:
                if self.early_stop:
                    # We raise an exception to stop the training because the questions in the training set have either met the satisfaction threshold or reached the num_rollouts_threshold
                    raise EarlyStop("No questions to sample to train. All questions in the training set have either met the satisfaction threshold or reached the num_rollouts_threshold")
                else:
                    # If we decide to continue selecting questions, we will use the following strategy:
                    # If there are any discarded problems, we will sample exclusively from them.
                    # Otherwise, all problems are satisfactory, we will randomly sample from the entire set of problems.
                    if len(self.given_up_questions) > 0:
                        new_questions_to_train = random.choices(self.given_up_questions, k=remaining_sample_size)
                    else:
                        new_questions_to_train = random.choices(self.all_questions, k=remaining_sample_size)
            else:
                new_questions_to_train = random.choices(question_sampling_pool, k=remaining_sample_size)

        self.questions_to_train = selected_from_previous + new_questions_to_train

        selected_questions = random_questions + self.questions_to_train

        return selected_questions
    
    def update_question_metrics(self, episodes: Dataset) -> dict:
        """
        Update the question metrics based on the episodes, which have the question indexes and the correctness of the answers
        """
        rewards = {}
        counts = {}
        # Calculate the pass@1 for batch problems
        for instance in episodes:
            question_idx = instance["question_idx"]
            reward = instance["reward"]
            self.question_metrics[question_idx]["num_rollouts"] += 1
            self.question_metrics[question_idx]["correct"] += reward
            if question_idx not in rewards:
                rewards[question_idx] = reward
                counts[question_idx] = 1
            else:
                rewards[question_idx] += reward
                counts[question_idx] += 1

        for question_idx in rewards.keys():
            accuracy = rewards[question_idx] / counts[question_idx]
            if accuracy >= self.satisfaction_threshold and question_idx not in self.satisfied_questions:
                self.satisfied_questions.append(question_idx)
                logger.info(f"Question {question_idx} has passed the satisfaction threshold")
            else:
                if question_idx in self.satisfied_questions:
                    # Since half of the questions are sampled randomly, we might end up including some problems that are already in the satisfied_questions list
                    # If, upon evaluating the problem's pass@1, we find that it falls below the satisfaction threshold, we should remove it from the satisfied questions list, so it can be selected for training later
                    self.satisfied_questions.remove(question_idx)
                    logger.info(f"Question {question_idx} has fallen below the satisfaction threshold")
                if self.question_metrics[question_idx]["num_rollouts"] >= self.num_rollouts_threshold:
                    self.given_up_questions.append(question_idx)
                    logger.warning(f"Question {question_idx}'s pass@1 cannot be satisfied after {self.num_rollouts_threshold} rollouts")
        
        logger.info(f"#Questions satisfied: {len(self.satisfied_questions)}")
        logger.info(f"#Questions given up: {len(self.given_up_questions)}")

        return {
            "satisfied": len(self.satisfied_questions),
            "given_up": len(self.given_up_questions)
        }



