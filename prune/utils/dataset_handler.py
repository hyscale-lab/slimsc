from typing import List, Dict, Tuple, Optional
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

class DatasetHandler:
    def __init__(self, dataset_name: str):
        """Initialize dataset handler with specific type.
        
        Args:
            dataset_type (str): Type of dataset ("gpqa_diamond" or "aime")
        """
        if dataset_name not in ["gpqa_diamond", "aime"]:
            raise ValueError(f"Unknown dataset type: {dataset_name}")
        self.dataset_name = dataset_name

    def load_dataset(self, split: str = "train") -> List[Dict]:
        """Load dataset with optional subset specification."""
        if self.dataset_name == "gpqa_diamond":
            return load_data_gpqa(dataset_name=self.dataset_name, split=split)
        elif self.dataset_name == "math500":
            return load_data_math500(dataset_name=self.dataset_name, split=split)
        else:  # aime
            return load_data_aime(dataset_name=self.dataset_name, split=split)

    def create_prompt(self, example: Dict) -> Tuple[str, List[str] | str]:
        """Create prompt from example."""
        if self.dataset_name == "gpqa_diamond":
            return create_prompt_gpqa(example)
        elif self.dataset_name == "math500":
            return create_prompt_math500(example)
        else:
            return create_prompt_aime(example)

    def extract_answer(self, content: Optional[str]) -> Optional[str]:
        """Extract answer from model response."""
        if self.dataset_name == "gpqa_diamond":
            return extract_answer_gpqa(content)
        elif self.dataset_name == "math500":
            return extract_answer_math500(content)
        else:
            return extract_answer_aime(content)

    def calculate_score(self, extracted_answer: Optional[str], correct_answer: str) -> int:
        """Calculate score for extracted answer."""
        if self.dataset_name == "gpqa_diamond":
            return calculate_score_gpqa(extracted_answer, correct_answer)
        elif self.dataset_name == "math500":
            return calculate_score_math500(extracted_answer, correct_answer)
        else:
            return calculate_score_aime(extracted_answer, correct_answer)