from __future__ import annotations
import numpy as np
from datasets import Dataset
from treetune.logging_utils import get_logger
from treetune.question_samplers.base_question_sampler import QuestionSampler

logger = get_logger(__name__)


@QuestionSampler.register("random")
class RandomQuestionSampler(QuestionSampler):
    def __init__(self, 
                 total_num_questions: int,
                 ) -> None:
        """
        Params:
            total_num_questions (int):
                The number of questions in the trainset
        """
        self.total_num_questions = total_num_questions
        self.all_questions = list(range(self.total_num_questions))
    
    def select_question(self, sample_num: int):
        """
        Randomly sample sample_num questions
        """
        assert sample_num <= self.total_num_questions
        return np.random.choice(self.all_questions, sample_num, replace=False)
    
    def update_question_metrics(self, episodes: Dataset):
        """
        Update the question metrics based on the episodes, which have the question indexes and the correctness of the answers
        """
        return {}



