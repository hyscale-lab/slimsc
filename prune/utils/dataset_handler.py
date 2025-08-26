# slimsc/prune/utils/dataset_handler.py

from typing import List, Dict, Tuple, Optional, Union, Any
from .gpqa_utils import (
    load_data_gpqa, 
    create_prompt_gpqa, 
    extract_answer_gpqa, 
    calculate_score_gpqa
)
from .aime_utils import (
    load_data_aime, 
    create_prompt_aime, 
    extract_answer_aime, 
    calculate_score_aime
)
from .math500_utils import (
    load_data_math500,
    create_prompt_math500,
    extract_answer_math500,
    calculate_score_math500
)
from .hmmt_utils import (
    load_data_hmmt,
    create_prompt_hmmt,
    extract_answer_hmmt,
    calculate_score_hmmt
)
from .aqua_rat_utils import (
    load_data_aqua_rat,
    create_prompt_aqua_rat,
    extract_answer_aqua_rat,
    calculate_score_aqua_rat
)
from .math_dapo_utils import (
    load_data_math_dapo,
    create_prompt_math_dapo,
    extract_answer_math_dapo,
    calculate_score_math_dapo
)

class DatasetHandler:
    def __init__(self, dataset_name: str):
        """Initialize dataset handler with specific type.
        
        Args:
            dataset_name (str): Type of dataset ("gpqa_diamond", "aime", "math500", "aqua_rat", "hmmt", "math_dapo")
        """
        if dataset_name not in ["gpqa_diamond", "aime", "math500", "aqua_rat", "hmmt", "math_dapo"]:
            raise ValueError(f"Unknown dataset type: {dataset_name}")
        self.dataset_name = dataset_name

    def load_dataset(self, split: str = "train") -> List[Dict]:
        """Load dataset with optional subset specification."""
        if self.dataset_name == "gpqa_diamond":
            return load_data_gpqa(dataset_name=self.dataset_name, split=split)
        elif self.dataset_name == "math500":
            return load_data_math500(dataset_name=self.dataset_name, split=split)
        elif self.dataset_name == "aime":
            return load_data_aime(dataset_name=self.dataset_name, split=split)
        elif self.dataset_name == "aqua_rat":
            return load_data_aqua_rat()
        elif self.dataset_name == "hmmt":
            return load_data_hmmt(dataset_name=self.dataset_name, split=split)
        elif self.dataset_name == "math_dapo":
            return load_data_math_dapo(split=split)
        else:
            raise ValueError(f"Unhandled dataset name '{self.dataset_name}' in load_dataset method.")

    def create_prompt(self, example: Dict) -> Tuple[str, Union[List[str], str, Tuple[Any, ...]]]:
        """Create prompt from example.
        Returns:
            A tuple: (prompt_string, correct_answer_details).
            For GPQA, correct_answer_details is a tuple: (choices_list, correct_answer_letter).
            For AIME/MATH500/HMMT, correct_answer_details is the correct_answer_string.
        """
        if self.dataset_name == "gpqa_diamond":
            _prompt, _choices, _correct_letter = create_prompt_gpqa(example)
            return _prompt, (_choices, _correct_letter)
        elif self.dataset_name == "math500":
            return create_prompt_math500(example)
        elif self.dataset_name == "aime":
            return create_prompt_aime(example)
        elif self.dataset_name == "aqua_rat":
            return create_prompt_aqua_rat(example)
        elif self.dataset_name == "hmmt":
            return create_prompt_hmmt(example)
        elif self.dataset_name == "math_dapo":
            return create_prompt_math_dapo(example)
        else:
            raise ValueError(f"Unhandled dataset name '{self.dataset_name}' in create_prompt method.")

    def extract_answer(self, content: Optional[str]) -> Optional[str]:
        """Extract answer from model response."""
        if self.dataset_name == "gpqa_diamond":
            return extract_answer_gpqa(content)
        elif self.dataset_name == "math500":
            return extract_answer_math500(content)
        elif self.dataset_name == "aime":
            return extract_answer_aime(content)
        elif self.dataset_name == "aqua_rat":
            return extract_answer_aqua_rat(content)
        elif self.dataset_name == "hmmt":
            return extract_answer_hmmt(content)
        elif self.dataset_name == "math_dapo":
            return extract_answer_math_dapo(content)
        else:
            raise ValueError(f"Unhandled dataset name '{self.dataset_name}' in extract_answer method.")

    def calculate_score(self, extracted_answer: Optional[str], correct_answer: str) -> int:
        """Calculate score for extracted answer."""
        if self.dataset_name == "gpqa_diamond":
            return calculate_score_gpqa(extracted_answer, correct_answer)
        elif self.dataset_name == "math500":
            return calculate_score_math500(extracted_answer, correct_answer)
        elif self.dataset_name == "aime":
            return calculate_score_aime(extracted_answer, correct_answer)
        elif self.dataset_name == "aqua_rat":
            return calculate_score_aqua_rat(extracted_answer, correct_answer)
        elif self.dataset_name == "hmmt":
            return calculate_score_hmmt(extracted_answer, correct_answer)
        elif self.dataset_name == "math_dapo":
            return calculate_score_math_dapo(extracted_answer, correct_answer)
        else:
            raise ValueError(f"Unhandled dataset name '{self.dataset_name}' in calculate_score method.")