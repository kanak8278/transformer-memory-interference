"""
Loader for interleaved category-value datasets following "Unable to Forget" paper format.
"""

import json
import random
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
from jinja2 import Environment, FileSystemLoader

@dataclass
class InterleavedExperimentalPrompt:
    """
    Experimental prompt for interleaved dataset testing.

    For Retroactive Interference (RI) testing:
    - Ground truth = INITIAL (first) appearance of each category
    - Interleaved sequence contains all updates mixed together
    - Query asks for the INITIAL value
    """
    category: str
    baseline_value: str  # Ground truth (first appearance)
    baseline_position: int  # Position of baseline in the interleaved sequence
    interference_level: int  # Number of updates per category
    sequence_length: int  # Total length of interleaved sequence
    interleaved_sequence: List[Dict]  # Full interleaved sequence for context
    query: str
    expected_answer: str

    def to_prompt_string(self) -> str:
        """
        Convert to prompt string following "Unable to Forget" format.

        Structure:
        1. Instruction: Brief directive indicating the task
        2. Update stream: Interleaved category-value pairs
        3. Query: Retrieve INITIAL value for specific category
        """
        # Phase 1: Instruction
        prompt = "Please learn the following information. Pay attention to the INITIAL (first) value for each category.\n\n"

        # Phase 2: Update stream (interleaved sequence)
        prompt += "Information stream:\n"
        for update in self.interleaved_sequence:
            prompt += f"{update['category']}: {update['value']}\n"

        prompt += "\n"

        # Phase 3: Query
        prompt += f"Question: {self.query}\n"
        prompt += "Answer:"

        return prompt

@dataclass
class BatchInterleavedExperimentalPrompt:
    """
    Batch experimental prompt asking about ALL categories at once.
    Matches "Unable to Forget" paper methodology (1 API call total).

    For Retroactive Interference (RI) testing:
    - Ground truth = INITIAL (first) appearance of each category
    - Interleaved sequence contains all updates mixed together
    - Query asks for the INITIAL value of ALL categories at once
    """
    categories: List[str]
    baseline_values: Dict[str, str]  # Ground truth for all categories
    baseline_positions: Dict[str, int]  # Position of baseline for each category
    interference_level: int  # Number of updates per category
    sequence_length: int  # Total length of interleaved sequence
    interleaved_sequence: List[Dict]  # Full interleaved sequence for context

    def to_prompt_string(self) -> str:
        """
        Convert to prompt string asking about ALL categories at once.
        Follows "Unable to Forget" paper format exactly.
        Uses Jinja2 template for consistent prompt generation.

        Structure:
        1. Secretary instruction with keys to track
        2. Text stream: Interleaved category-value pairs
        3. Query: Retrieve INITIAL value for each key with specific format
        """
        # Load Jinja template
        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_dir = os.path.join(os.path.dirname(current_dir), 'prompts')
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template('batch_interleaved_prompt.jinja')

        # Render template with data
        categories_list = ", ".join(self.categories)
        prompt = template.render(
            num_categories=len(self.categories),
            categories_list=categories_list,
            interleaved_sequence=self.interleaved_sequence
        )

        return prompt

    def get_expected_answers(self) -> Dict[str, str]:
        """Get expected answers for all categories"""
        return self.baseline_values.copy()

class InterleavedDatasetLoader:
    """Load and manage interleaved category-value datasets"""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = None
        self._load_dataset()

    def _load_dataset(self):
        """Load the interleaved dataset"""
        with open(self.dataset_path, 'r') as f:
            self.dataset = json.load(f)

        # Support both old and new dataset formats
        if 'categories' in self.dataset:
            # Old format: categories at root level
            categories = self.dataset['categories']
        elif 'metadata' in self.dataset and 'categories' in self.dataset['metadata']:
            # New format: categories in metadata
            categories = self.dataset['metadata']['categories']
        else:
            raise ValueError("Dataset missing 'categories' field")

        print(f"Loaded interleaved dataset:")
        print(f"  Categories: {len(categories)}")
        print(f"  Levels: {self.dataset['metadata']['interference_levels']}")

    def get_categories(self) -> List[str]:
        """Get all categories"""
        # Support both old and new formats
        if 'categories' in self.dataset:
            return self.dataset['categories']
        elif 'metadata' in self.dataset and 'categories' in self.dataset['metadata']:
            return self.dataset['metadata']['categories']
        else:
            raise ValueError("Dataset missing 'categories' field")

    def get_levels(self) -> List[int]:
        """Get all interference levels"""
        return self.dataset['metadata']['interference_levels']

    def create_experiment_prompts(self,
                                 interference_level: int,
                                 categories: Optional[List[str]] = None) -> List[InterleavedExperimentalPrompt]:
        """
        OLD METHOD: Create experimental prompts for testing at a specific interference level.
        Creates ONE prompt per category (46 API calls total).

        DEPRECATED: Use create_batch_experiment_prompt() instead to match "Unable to Forget" paper.

        Args:
            interference_level: Number of updates per category
            categories: List of categories to test (None = all)

        Returns:
            List of InterleavedExperimentalPrompt objects
        """
        if str(interference_level) not in self.dataset['levels']:
            raise ValueError(f"Interference level {interference_level} not found in dataset")

        level_data = self.dataset['levels'][str(interference_level)]

        if categories is None:
            categories = self.get_categories()

        prompts = []

        for category in categories:
            baseline_value = level_data['baseline_values'][category]
            baseline_position = level_data['baseline_positions'][category]
            interleaved_sequence = level_data['interleaved_sequence']

            query = f"What was the INITIAL value for {category}?"
            expected_answer = baseline_value

            prompt = InterleavedExperimentalPrompt(
                category=category,
                baseline_value=baseline_value,
                baseline_position=baseline_position,
                interference_level=interference_level,
                sequence_length=level_data['sequence_length'],
                interleaved_sequence=interleaved_sequence,
                query=query,
                expected_answer=expected_answer
            )

            prompts.append(prompt)

        return prompts

    def create_batch_experiment_prompt(self,
                                      interference_level: int,
                                      categories: Optional[List[str]] = None) -> 'BatchInterleavedExperimentalPrompt':
        """
        NEW METHOD: Create ONE prompt asking about ALL categories at once.
        Matches "Unable to Forget" paper methodology (1 API call total).

        Args:
            interference_level: Number of updates per category
            categories: List of categories to test (None = all)

        Returns:
            Single BatchInterleavedExperimentalPrompt object
        """
        if str(interference_level) not in self.dataset['levels']:
            raise ValueError(f"Interference level {interference_level} not found in dataset")

        level_data = self.dataset['levels'][str(interference_level)]

        if categories is None:
            categories = self.get_categories()

        # Build ground truth mapping
        baseline_values = {cat: level_data['baseline_values'][cat] for cat in categories}
        baseline_positions = {cat: level_data['baseline_positions'][cat] for cat in categories}

        return BatchInterleavedExperimentalPrompt(
            categories=categories,
            baseline_values=baseline_values,
            baseline_positions=baseline_positions,
            interference_level=interference_level,
            sequence_length=level_data['sequence_length'],
            interleaved_sequence=level_data['interleaved_sequence']
        )

    def get_ground_truth(self, interference_level: int, category: str) -> str:
        """Get ground truth (baseline value) for a category at a specific level"""
        level_data = self.dataset['levels'][str(interference_level)]
        return level_data['baseline_values'][category]

    def get_baseline_position(self, interference_level: int, category: str) -> int:
        """Get position of baseline value in the interleaved sequence"""
        level_data = self.dataset['levels'][str(interference_level)]
        return level_data['baseline_positions'][category]

    def get_category_updates(self, interference_level: int, category: str) -> List[Dict]:
        """Get all updates for a specific category from the interleaved sequence"""
        level_data = self.dataset['levels'][str(interference_level)]
        return [
            update for update in level_data['interleaved_sequence']
            if update['category'] == category
        ]

    def validate_interleaving(self, interference_level: int) -> Dict:
        """
        Validate that the interleaving follows "Unable to Forget" constraints:
        - No consecutive updates for same category
        - Each category has exactly 'interference_level' updates
        - First update for each category is tagged as baseline
        """
        level_data = self.dataset['levels'][str(interference_level)]
        sequence = level_data['interleaved_sequence']

        # Check no consecutive repeats
        violations = []
        for i in range(len(sequence) - 1):
            if sequence[i]['category'] == sequence[i + 1]['category']:
                violations.append(i)

        # Check update counts per category
        category_counts = {}
        baseline_counts = {}
        for update in sequence:
            cat = update['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
            if update['is_baseline']:
                baseline_counts[cat] = baseline_counts.get(cat, 0) + 1

        # Check all categories have correct count
        count_errors = []
        for cat in self.get_categories():
            if category_counts.get(cat, 0) != interference_level:
                count_errors.append((cat, category_counts.get(cat, 0)))

        # Check all categories have exactly 1 baseline
        baseline_errors = []
        for cat in self.get_categories():
            if baseline_counts.get(cat, 0) != 1:
                baseline_errors.append((cat, baseline_counts.get(cat, 0)))

        return {
            "valid": len(violations) == 0 and len(count_errors) == 0 and len(baseline_errors) == 0,
            "consecutive_violations": violations,
            "count_errors": count_errors,
            "baseline_errors": baseline_errors,
            "total_updates": len(sequence),
            "expected_total": len(self.get_categories()) * interference_level
        }
