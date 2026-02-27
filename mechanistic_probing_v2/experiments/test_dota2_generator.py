"""Test script for the Dota 2 narrative interference generator."""

import sys
from pathlib import Path

# Add parent dir to path so we can import core modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.narrative_generator.dota2 import DotaTrialGenerator


def main():
    gen = DotaTrialGenerator()

    print("=" * 80)
    print("Dota 2 Narrative Interference Generator - Test Run")
    print("=" * 80)

    for seed in range(5):
        print(f"\n--- Seed {seed} ---")
        trial = gen.generate_trial(num_keys=5, num_updates=7, condition="RI", seed=seed)
        word_count = len(trial["narrative"].split())
        tracked_values = sum(len(v) for v in trial["entity_tracking"].values())
        ri = trial["questions"]["RI"]["expected_answer"]
        pi = trial["questions"]["PI"]["expected_answer"]

        print(f"Seed {seed}: {word_count} words, "
              f"{tracked_values} tracked values, "
              f"RI={ri}, "
              f"PI={pi}")

        # Show config
        cfg = trial["config"]
        print(f"  Archetype: {cfg['game_archetype']}, "
              f"Tracked: {cfg['tracked_attribute']}, "
              f"Mode: {cfg['attribute_mode']}, "
              f"Filler: {cfg['filler_budget']}")
        print(f"  Teams: {cfg['team1']} vs {cfg['team2']}")
        print(f"  Queried: {trial['questions']['RI']['target_entity']} / "
              f"{trial['questions']['RI']['target_attribute']}")

        # Show entity tracking summary
        for key, vals in trial["entity_tracking"].items():
            print(f"  {key}: {len(vals)} values [{vals[0] if vals else '?'} -> {vals[-1] if vals else '?'}]")

        # Validate
        try:
            gen.validate_trial(trial)
            print(f"  Validation passed")
        except AssertionError as e:
            print(f"  VALIDATION FAILED: {e}")

    # Test determinism
    print("\n" + "=" * 80)
    print("Determinism test: generating same seed twice")
    print("=" * 80)
    trial_a = gen.generate_trial(num_keys=5, num_updates=7, condition="RI", seed=42)
    trial_b = gen.generate_trial(num_keys=5, num_updates=7, condition="RI", seed=42)
    assert trial_a["narrative"] == trial_b["narrative"], "DETERMINISM FAILED!"
    assert trial_a["entity_tracking"] == trial_b["entity_tracking"], "DETERMINISM FAILED (tracking)!"
    print("Determinism: PASSED (identical narratives and tracking)")

    # Test different configs
    print("\n" + "=" * 80)
    print("Config variation tests")
    print("=" * 80)
    configs = [
        (2, 3, "RI", 100),
        (3, 5, "PI", 200),
        (10, 10, "RI", 300),
        (5, 15, "PI", 400),
    ]
    for nk, nu, cond, s in configs:
        print(f"\n  num_keys={nk}, num_updates={nu}, condition={cond}, seed={s}")
        trial = gen.generate_trial(num_keys=nk, num_updates=nu, condition=cond, seed=s)
        word_count = len(trial["narrative"].split())
        tracked_values = sum(len(v) for v in trial["entity_tracking"].values())
        print(f"    {word_count} words, {tracked_values} tracked values")
        try:
            gen.validate_trial(trial)
            print(f"    Validation passed")
        except AssertionError as e:
            print(f"    VALIDATION FAILED: {e}")

    print("\nAll tests complete.")


if __name__ == "__main__":
    main()
