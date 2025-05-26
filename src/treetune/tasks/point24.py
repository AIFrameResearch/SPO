import re
from typing import Dict, List, Optional, Any

import math

from datasets import (
    Dataset,
    DatasetDict,
)

from treetune import logging_utils
from treetune.tasks import Task

logger = logging_utils.get_logger(__name__)


class Point24Checker:
    def check(self, cards: List[int]) -> bool:
        return self._search(cards, 24)

    def _calculate(self, num1, num2, op):
        if op == '+':
            return num1 + num2
        elif op == '-':
            return num1 - num2
        elif op == '*':
            return num1 * num2
        elif op == '/':
            return num1 / num2

    def _search(self, numbers, target):
        if len(numbers) == 1:
            return math.fabs(numbers[0] - target) < 1e-6

        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                num1 = numbers[i]
                num2 = numbers[j]
                for op in '+-*/':
                    try:
                        result = self._calculate(num1, num2, op)
                        new_numbers = numbers[:i] + [result] + numbers[i+1:j] + numbers[j+1:]
                        if self._search(new_numbers, target):
                            return True
                    except ZeroDivisionError:
                        pass

                    try:
                        if op in '-/':
                            result = self._calculate(num2, num1, op)
                            new_numbers = numbers[:i] + [result] + numbers[i+1:j] + numbers[j+1:]
                            if self._search(new_numbers, target):
                                return True
                    except ZeroDivisionError:
                        pass
        return False

@Task.register("point24", exist_ok=True)
class Point24(Task):
    def __init__(
        self,
        answer_pattern: Optional[str] = r"\\boxed\{(.*?)\}",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.answer_pattern = answer_pattern

    def extract_predicted_answer_from_text(self, text: str) -> Optional[str]:
        matches = re.findall(self.answer_pattern, text)
        
        if len(matches) == 1:
            return matches[0]
        else:
            return None

    def grade_answer(
        self,
        instance: dict,
        given_answer: Optional[str] = None,
    ) -> bool:
        if given_answer is None:
            return False
        
        given_answer = re.sub(r'\s*=\s*24\s*$', '', given_answer)

         # Step 1: Check if the expression contains only allowed characters
        if not re.fullmatch(r'^[\d+*/\-()\s\\timesfrac{}]+$', given_answer):
            return False
    
        given_answer = re.sub(r'\\times', '*', given_answer)
        given_answer = re.sub(r'\\frac{([^}]+)}{([^}]+)}', r'(\1)/(\2)', given_answer)

        # Step 2: Extract all numbers from the expression
        expr_numbers = list(map(int, re.findall(r'\d+', given_answer)))
        
        # Step 3: Verify all input numbers are used exactly once
        input_numbers = sorted(list(map(int, instance["puzzle"])))
        expr_sorted = sorted(expr_numbers)
        if input_numbers != expr_sorted:
            return False

        # Step 4: Evaluate the expression safely
        try:
            value = eval(given_answer, {'__builtins__': None}, {})
        except:
            return False

        # Step 5: Check if the result is approximately 24 (handling floating point precision)
        return abs(value - 24) < 1e-6

    # noinspection DuplicatedCode
    def evaluate_predictions(
        self,
        *,
        predictions: List[List[str]] = None,
        references: Dataset = None,
    ) -> Dict[str, float]:
        once_hit_acc = []
        correct_frac = []
        unique_answer_count = []
        none_answer_extracted = []

        for solution_candidates, ref in zip(predictions, references):
            assert len(solution_candidates) > 0
            answer_candidates = [
                self.extract_predicted_answer_from_text(sol)
                for sol in solution_candidates
            ]
            none_answer_extracted.append(
                sum([1 for ans in answer_candidates if ans is None])
                / len(answer_candidates)
            )

            grading_results = [
                self.grade_answer(given_answer=ans, instance=ref)
                for ans in answer_candidates
            ]
            once_hit_acc.append(float(any(grading_results)))
            correct_frac.append(sum(grading_results) / len(grading_results))

            answer_candidates = [
                tuple(ans) if isinstance(ans, list) else ans
                for ans in answer_candidates
            ]

            assert len(answer_candidates) == len(grading_results)

            unique_answer_count.append(len(set(answer_candidates)))

        once_hit = sum(once_hit_acc) / len(once_hit_acc)
        correct_frac = sum(correct_frac) / len(correct_frac)

        return {
            "once_hit": once_hit,
            "exact_match": once_hit,  # for backwards compatibility
            "correct_frac": correct_frac,
            "exact_match_frac": correct_frac,  # for backwards compatibility
            "unique_answer_count": sum(unique_answer_count) / len(unique_answer_count),
            "none_answer_extracted_frac_per_problem": (
                sum(none_answer_extracted) / len(none_answer_extracted)
            ),
        }    

    def build_dataset(
        self,
    ) -> DatasetDict:
        datasets = super().build_dataset()
        datasets = datasets.map(
            self._preprocess_example, num_proc=4, desc="Preprocessing examples"
        )
        return datasets

    def _preprocess_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        puzzle = example["puzzle"]
        output = {
            "puzzle": puzzle,
            "query": str(puzzle)
        }
        return output
