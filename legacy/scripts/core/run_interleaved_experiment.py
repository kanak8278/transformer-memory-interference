#!/usr/bin/env python3
"""
Run retroactive interference experiments with interleaved category-value datasets.
Following "Unable to Forget" paper methodology with random interleaving.
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

from datasets_v0.interleaved_dataset_loader import InterleavedDatasetLoader
from datasets_v0.category_response_parser import CategoryResponseParser
from models import create_model

def run_experiment(model_name: str, interference_level: int, dataset_path: str, sample_size: int = 46):
    """
    Run experiment for a single interference level.
    Uses batch mode (ONE API call) matching "Unable to Forget" paper methodology.

    Args:
        model_name: Model to use (claude-haiku, claude-sonnet, etc.)
        interference_level: Number of updates per category
        dataset_path: Path to interleaved dataset
        sample_size: Number of categories to test
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {model_name}, Level {interference_level}")
    print(f"{'='*80}")

    # Load dataset
    loader = InterleavedDatasetLoader(dataset_path)

    # Get categories to test
    all_categories = loader.get_categories()
    if sample_size < len(all_categories):
        categories = all_categories[:sample_size]
    else:
        categories = all_categories

    # Create batch experimental prompt (ONE prompt for ALL categories)
    batch_prompt = loader.create_batch_experiment_prompt(interference_level, categories)

    print(f"\nTesting {len(categories)} categories with {interference_level} updates each")
    print(f"Total sequence length: {batch_prompt.sequence_length}")
    print(f"Using BATCH MODE: 1 API call for all {len(categories)} categories")

    # Initialize model using factory
    model = create_model(model_name)

    # Get LLM response (ONE API call for ALL categories)
    prompt_str = batch_prompt.to_prompt_string()

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

    # Parse batch response (extract all category values)
    extracted_values = CategoryResponseParser.parse_batch_response(response, categories)

    # Evaluate results
    results = []
    correct_count = 0
    missing_count = 0
    expected_answers = batch_prompt.get_expected_answers()

    for category in categories:
        extracted_value = extracted_values.get(category)
        expected_answer = expected_answers[category]
        baseline_position = batch_prompt.baseline_positions[category]

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
            "expected": expected_answer,
            "extracted": extracted_value,
            "correct": is_correct,
            "missing": is_missing,
            "baseline_position": baseline_position
        })

    # Calculate metrics
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
            print(f"    Expected: {error['expected']}")
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
        "batch_response": response[:10000],  # Store first 10000 chars of batch response (increased for DeepSeek reasoning tokens)
        "token_usage": token_info,  # NEW: Token and error tracking
        "results": results
    }

def main():
    parser = argparse.ArgumentParser(description="Run interleaved RI experiments")
    parser.add_argument("--model", type=str, default="claude-haiku",
                      help="Model name (claude-haiku, claude-sonnet, gemini-pro, gemini-2.5-pro, gpt-4, gpt-4-turbo, llama-3.1-8b, llama-3.1-70b)")
    parser.add_argument("--levels", type=int, nargs="+", default=[3, 10, 50, 100],
                      help="Interference levels to test")
    parser.add_argument("--sample-size", type=int, default=46,
                      help="Number of categories to test")
    parser.add_argument("--dataset", type=str,
                      default="data/interleaved_dataset_meaningful.json",
                      help="Path to interleaved dataset")
    parser.add_argument("--output-dir", type=str, default="results",
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
    output_file = f"{args.output_dir}/{args.model}_levels_{'_'.join(map(str, args.levels))}_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump({
            "experiment": "interleaved_retroactive_interference",
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
