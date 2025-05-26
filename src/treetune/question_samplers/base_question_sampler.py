from __future__ import annotations
from pathlib import Path
import pickle
from treetune.common import Registrable, Lazy, JsonDict, Params
from datasets import Dataset

class QuestionSampler(Registrable):
    def select_question(self, sample_num: int):
        raise NotImplementedError
    
    def update_question_metrics(self, episodes: Dataset):
        raise NotImplementedError
    
    @staticmethod
    def load_from_disk(path: Path):
        """
        Load the question metrics from disk
        """
        with open(path, "rb") as f:
            return pickle.load(f)
    
    @staticmethod
    def save_to_disk(question_sampler: QuestionSampler, path: Path):
        """
        Save the question metrics to disk
        """
        with open(path, "wb") as f:
            pickle.dump(question_sampler, f)
    