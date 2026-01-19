#!/usr/bin/env python3
"""
Run proactive interference experiments with interleaved category-value datasets.
Following "Unable to Forget" paper methodology with random interleaving.

KEY DIFFERENCE FROM RI:
- RI (run_interleaved_experiment.py): Query FIRST-learned values (baseline)
- PI (this script): Query LAST-learned values (most recent)

DATASET: Uses SAME interleaved sequence as RI for controlled comparison.
"""

import sys
import os
import json
import argparse
from datetime import datetime
from typing import Dict

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from datasets.interleaved_dataset_loader import InterleavedDatasetLoader
from datasets.category_response_parser import CategoryResponseParser
from models import create_model

def run_experiment(model_name: str, interference_level: int, dataset_path: str, sample_size: int = 46):
    """
    Run experiment for a single interference level.
    Uses batch mode (ONE API call) matching "Unable to Forget" paper methodology.

    KEY DIFFERENCE: Queries for LAST-learned values (Proactive Interference)

    Args:
        model_name: Model to use (claude-haiku, claude-sonnet, etc.)
        interference_level: Number of updates per category
        dataset_path: Path to interleaved dataset
        sample_size: Number of categories to test
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {model_name}, Level {interference_level} (PI)")
    print(f"{'='*80}")

    # Load dataset - SAME as RI
    loader = InterleavedDatasetLoader(dataset_path)

    # Get categories to test - SAME as RI
    all_categories = loader.get_categories()
    if sample_size < len(all_categories):
        categories = all_categories[:sample_size]
    else:
        categories = all_categories

    # Create batch experimental prompt - SAME sequence generation as RI
    batch_prompt = loader.create_batch_experiment_prompt(interference_level, categories)

    # Get the interleaved sequence - SAME as RI
    interleaved_sequence = batch_prompt.interleaved_sequence

    # Build PI-specific prompt - DIFFERENT: queries LAST values
    prompt_str = build_pi_prompt(interleaved_sequence, categories)

    # Expected answers for PI - DIFFERENT: LAST appearance instead of FIRST
    expected_answers = get_last_learned_values(interleaved_sequence, categories)

    print(f"\nTesting {len(categories)} categories with {interference_level} updates each")
    print(f"Total sequence length: {batch_prompt.sequence_length}")
    print(f"Using BATCH MODE: 1 API call for all {len(categories)} categories")
    print(f"Query type: LAST-learned values (Proactive Interference)")

    # Initialize model using factory
    model = create_model(model_name)

    # Print the prompt being sent to the LLM
    print(f"\n{'='*80}")
    print(f"PROMPT SENT TO LLM:")
    print(f"{'='*80}")
    print(f"Length: {len(prompt_str)} chars")
    print(f"\nFirst 1000 chars:")
    print(prompt_str[:1000])
    if len(prompt_str) > 1000:
        print(f"\n... [middle section truncated] ...")
        print(f"\nLast 500 chars:")
        print(prompt_str[-500:])
    print(f"{'='*80}\n")

    print(f"Calling LLM API...", end='', flush=True)
    response = model.generate(prompt_str)
    print(" Done!")

    # Print the raw response to debug parsing issues
    print(f"\n{'='*80}")
    print(f"RAW MODEL RESPONSE DIAGNOSTICS:")
    print(f"{'='*80}")
    print(f"Type: {type(response)}")
    print(f"Length: {len(response)} chars")
    print(f"Is empty/whitespace: {len(response.strip()) == 0}")
    if len(response) > 0:
        print(f"First 100 bytes (escaped): {repr(response[:100])}")
    print(f"\nFirst 1500 chars:")
    print(f"{'='*80}")
    print(response[:1500])
    if len(response) > 1500:
        print(f"\n... [showing {1500} of {len(response)} chars] ...")
    print(f"{'='*80}\n")

    # Parse batch response (extract all category values) - SAME parser as RI
    extracted_values = CategoryResponseParser.parse_batch_response(response, categories)

    # Evaluate results - DIFFERENT: compare against LAST values
    results = []
    correct_count = 0
    missing_count = 0

    for category in categories:
        extracted_value = extracted_values.get(category)
        expected_answer = expected_answers[category]

        # Get first and last values for error analysis
        category_updates = [u for u in interleaved_sequence if u['category'] == category]
        first_value = category_updates[0]['value'] if category_updates else None
        last_value = category_updates[-1]['value'] if category_updates else None
        baseline_position = next((i for i, u in enumerate(interleaved_sequence, 1)
                                 if u['category'] == category), 0)

        # Check correctness
        is_correct = False
        is_missing = False

        if extracted_value is None:
            is_missing = True
            missing_count += 1
        elif extracted_value.lower() == expected_answer.lower():
            is_correct = True
            correct_count += 1

        results.append({
            "category": category,
            "expected": expected_answer,  # LAST value (PI)
            "extracted": extracted_value,
            "correct": is_correct,
            "missing": is_missing,
            "baseline_position": baseline_position,
            "first_value": first_value,
            "last_value": last_value,
            "num_updates": len(category_updates)
        })

    # Calculate metrics - SAME calculation as RI
    total = len(results)
    accuracy = (correct_count / total) * 100 if total > 0 else 0
    missing_rate = (missing_count / total) * 100 if total > 0 else 0

    print(f"\nResults:")
    print(f"  Correct: {correct_count}/{total} ({accuracy:.2f}%)")
    print(f"  Missing: {missing_count}/{total} ({missing_rate:.2f}%)")

    # Show some example errors
    errors = [r for r in results if not r['correct']]
    if errors:
        print(f"\nExample errors (showing first 5):")
        for error in errors[:5]:
            print(f"  Category: {error['category']}")
            print(f"    First value: {error['first_value']}")
            print(f"    Last value (expected): {error['last_value']}")
            print(f"    Extracted: {error['extracted']}")
            print(f"    Baseline position: {error['baseline_position']}")

    # Capture token usage and error information from model
    token_info = {
        "input_tokens": model.last_input_tokens,
        "output_tokens": model.last_output_tokens,
        "total_tokens": model.last_total_tokens,
        "was_truncated": model.last_was_truncated,
        "error": model.last_error
    }

    return {
        "model": model_name,
        "interference_level": interference_level,
        "sample_size": len(results),
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total,
        "missing_count": missing_count,
        "missing_rate": missing_rate,
        "sequence_length": batch_prompt.sequence_length,
        "batch_response": response[:10000],  # Store first 10000 chars of batch response
        "token_usage": token_info,
        "results": results
    }

def build_pi_prompt(interleaved_sequence: list, categories: list) -> str:
    """
    Build PI-specific prompt that queries LAST-learned values.
    Uses SAME Jinja2 template structure as RI but changes the query instruction.

    Args:
        interleaved_sequence: List of {category, value} dicts (SAME as RI)
        categories: List of category names to query (SAME as RI)

    Returns:
        Formatted prompt string
    """
    from jinja2 import Environment, FileSystemLoader

    # Load Jinja template - mirrors RI structure exactly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'prompts')
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template('batch_proactive_prompt.jinja')

    # Render template with data - SAME variables as RI
    categories_list = ", ".join(categories)
    prompt = template.render(
        num_categories=len(categories),
        categories_list=categories_list,
        interleaved_sequence=interleaved_sequence
    )

    return prompt

def get_last_learned_values(interleaved_sequence: list, categories: list) -> Dict[str, str]:
    """
    Extract the LAST appearance of each category from the interleaved sequence.

    This is the key difference from RI:
    - RI uses FIRST appearance (baseline_values from batch_prompt)
    - PI uses LAST appearance (this function)

    Args:
        interleaved_sequence: List of {category, value} dicts
        categories: List of category names

    Returns:
        Dict mapping category to its LAST value
    """
    last_values = {}

    # Iterate through sequence and keep updating (last one wins)
    for update in interleaved_sequence:
        if update['category'] in categories:
            last_values[update['category']] = update['value']

    return last_values

def main():
    parser = argparse.ArgumentParser(description="Run interleaved PI experiments")
    parser.add_argument("--model", type=str, default="claude-haiku",
                      help="Model name (claude-haiku, claude-sonnet, gemini-pro, gemini-2.5-pro, gpt-4, gpt-4-turbo, llama-3.1-8b, llama-3.1-70b)")
    parser.add_argument("--levels", type=int, nargs="+", default=[3, 10, 50, 100],
                      help="Interference levels to test")
    parser.add_argument("--sample-size", type=int, default=46,
                      help="Number of categories to test")
    parser.add_argument("--dataset", type=str,
                      default="data/interleaved_dataset_meaningful.json",
                      help="Path to interleaved dataset")
    parser.add_argument("--output-dir", type=str, default="results_pi",
                      help="Output directory for results")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run experiments for each level
    all_results = []

    for level in args.levels:
        result = run_experiment(
            model_name=args.model,
            interference_level=level,
            dataset_path=args.dataset,
            sample_size=args.sample_size
        )
        all_results.append(result)

    # Save consolidated results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{args.output_dir}/{args.model}_pi_levels_{'_'.join(map(str, args.levels))}_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump({
            "experiment": "proactive_interference",
            "model": args.model,
            "levels": args.levels,
            "sample_size": args.sample_size,
            "timestamp": timestamp,
            "results": all_results
        }, f, indent=2)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\nModel: {args.model}")
    print(f"Levels tested: {args.levels}")
    print("\nAccuracy by level:")
    for result in all_results:
        print(f"  Level {result['interference_level']:3d}: {result['accuracy']:6.2f}% "
              f"({result['correct_count']}/{result['total_count']}) "
              f"[Missing: {result['missing_count']}]")

    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
