#!/usr/bin/env python3
"""
Test script for the three new narrative generators: wildlife, ICU, ATC.

Tests:
1. Generator imports and initializes without error
2. Generates 10 trials for each (num_keys=3, num_updates=5)
3. Validates ground truth constraints
4. Checks narrative realism (manual + automated checks)
5. Prints example trials for human review

Usage:
    .venv/bin/python v3/scripts/test_narrative_generators.py
"""

import sys
import json
import traceback
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

DOMAINS = []

# ─── Import generators ─────────────────────────────────────────────────────

print("=" * 70)
print("Loading generators...")
print("=" * 70)

try:
    from narrative_generator.wildlife.generator import WildlifeTrialGenerator
    DOMAINS.append(("wildlife", WildlifeTrialGenerator))
    print("✓ WildlifeTrialGenerator loaded")
except Exception as e:
    print(f"✗ WildlifeTrialGenerator failed: {e}")
    traceback.print_exc()

try:
    from narrative_generator.icu.generator import ICUTrialGenerator
    DOMAINS.append(("icu", ICUTrialGenerator))
    print("✓ ICUTrialGenerator loaded")
except Exception as e:
    print(f"✗ ICUTrialGenerator failed: {e}")
    traceback.print_exc()

try:
    from narrative_generator.atc.generator import ATCTrialGenerator
    DOMAINS.append(("atc", ATCTrialGenerator))
    print("✓ ATCTrialGenerator loaded")
except Exception as e:
    print(f"✗ ATCTrialGenerator failed: {e}")
    traceback.print_exc()

print()

# ─── Test configurations ───────────────────────────────────────────────────

TEST_CONFIGS = [
    {"num_keys": 2, "num_updates": 3, "label": "small (2k, 3u)"},
    {"num_keys": 3, "num_updates": 5, "label": "medium (3k, 5u)"},
    {"num_keys": 5, "num_updates": 7, "label": "medium-large (5k, 7u)"},
]

# ─── Validation checks ─────────────────────────────────────────────────────

def validate_trial(trial: dict, domain: str) -> list:
    """Returns list of error strings. Empty = passed."""
    errors = []

    # 1. Required fields
    for field in ["id", "domain", "num_keys", "num_updates", "narrative", "questions", "entity_tracking"]:
        if field not in trial:
            errors.append(f"Missing field: {field}")

    if "questions" in trial:
        q = trial["questions"]
        # 2. Both RI and PI questions exist
        for cond in ["RI", "PI"]:
            if cond not in q:
                errors.append(f"Missing {cond} question")
            else:
                for subfield in ["question", "expected_answer", "target_entity", "target_attribute"]:
                    if subfield not in q[cond]:
                        errors.append(f"Missing questions.{cond}.{subfield}")

        # 3. RI != PI (unless genuinely forced)
        if "RI" in q and "PI" in q:
            ri_ans = q["RI"].get("expected_answer", "")
            pi_ans = q["PI"].get("expected_answer", "")
            if ri_ans == pi_ans:
                errors.append(f"RI == PI (both {ri_ans!r}) — no interference possible")

        # 4. Target entity exists in entity_tracking
        if "RI" in q:
            target = q["RI"].get("target_entity", "")
            target_attr = q["RI"].get("target_attribute", "")
            key = f"{target} / {target_attr}"
            if "entity_tracking" in trial and key not in trial["entity_tracking"]:
                # Try to find any key containing target
                matches = [k for k in trial["entity_tracking"] if target in k]
                if not matches:
                    errors.append(f"Target entity '{target}' not in entity_tracking")

    if "entity_tracking" in trial:
        et = trial["entity_tracking"]
        # 5. Entity tracking has at least 2 distinct values per queried entity
        for key, values in et.items():
            if len(values) < 1:
                errors.append(f"entity_tracking['{key}'] has no values")
            if len(values) < 2:
                errors.append(f"entity_tracking['{key}'] has only 1 value — no interference")

    if "narrative" in trial:
        # 6. Narrative is non-trivial
        narr = trial["narrative"]
        if len(narr) < 100:
            errors.append(f"Narrative too short: {len(narr)} chars")
        # 7. RI answer appears in narrative
        if "questions" in trial and "RI" in trial["questions"]:
            ri_ans = trial["questions"]["RI"].get("expected_answer", "")
            if ri_ans and str(ri_ans) not in narr:
                errors.append(f"RI answer '{ri_ans}' not found in narrative")
        # 8. PI answer appears in narrative
        if "questions" in trial and "PI" in trial["questions"]:
            pi_ans = trial["questions"]["PI"].get("expected_answer", "")
            if pi_ans and str(pi_ans) not in narr:
                errors.append(f"PI answer '{pi_ans}' not found in narrative")

    return errors

# ─── Realism checks ────────────────────────────────────────────────────────

def check_realism(trial: dict, domain: str) -> list:
    """Domain-specific realism checks. Returns list of warnings."""
    warnings = []
    narr = trial.get("narrative", "")

    if domain == "wildlife":
        # Check for wolf terminology in non-wolf trials
        species = trial.get("config", {}).get("species", ["gray_wolf"])
        if "elk" in str(species) and "pup" in narr.lower():
            warnings.append("Elk narrative mentions 'pup' (wolf terminology)")
        if "elk" in str(species) and " pack " in narr.lower():
            warnings.append("Elk narrative mentions 'pack' (wolf terminology)")
        if "grizzly" in str(species) and "rendezvous site" in narr.lower():
            warnings.append("Grizzly narrative mentions 'rendezvous site' (wolf terminology)")
        # Check that weight values are species-appropriate
        if "entity_tracking" in trial:
            for key, vals in trial["entity_tracking"].items():
                if "weight" in key.lower():
                    for v in vals:
                        try:
                            w = float(v)
                            if "elk" in key.lower() and w < 100:
                                warnings.append(f"Elk weight {w}kg suspiciously low")
                            if "wolf" in key.lower() and w > 80:
                                warnings.append(f"Wolf weight {w}kg suspiciously high")
                            if "grizzly" in key.lower() and (w < 100 or w > 400):
                                warnings.append(f"Grizzly weight {w}kg out of range")
                        except (ValueError, TypeError):
                            pass  # non-numeric weight attribute

    elif domain == "icu":
        # Check vital sign ranges
        if "entity_tracking" in trial:
            for key, vals in trial["entity_tracking"].items():
                if "heart_rate" in key:
                    for v in vals:
                        try:
                            hr = float(v)
                            if hr < 30 or hr > 200:
                                warnings.append(f"Heart rate {hr} out of plausible range")
                        except (ValueError, TypeError):
                            pass
                if "temperature_c" in key:
                    for v in vals:
                        try:
                            t = float(v)
                            if t < 34 or t > 42:
                                warnings.append(f"Temperature {t}°C out of plausible range")
                        except (ValueError, TypeError):
                            pass
                if "lactate" in key:
                    for v in vals:
                        try:
                            lac = float(v)
                            if lac < 0.3 or lac > 25:
                                warnings.append(f"Lactate {lac} out of plausible range")
                        except (ValueError, TypeError):
                            pass

    elif domain == "atc":
        # Check altitude values
        if "entity_tracking" in trial:
            for key, vals in trial["entity_tracking"].items():
                if "altitude" in key:
                    for v in vals:
                        # FL format: FL240
                        if str(v).startswith("FL"):
                            try:
                                fl = int(str(v)[2:])
                                if fl < 50 or fl > 600:
                                    warnings.append(f"Flight level {v} out of range")
                            except ValueError:
                                pass
        # Check for correct ATC phraseology markers
        if "cleared" not in narr.lower() and "descend" not in narr.lower() and "climb" not in narr.lower():
            warnings.append("ATC narrative lacks expected phraseology (cleared/descend/climb)")

    return warnings


# ─── Run tests ─────────────────────────────────────────────────────────────

all_results = {}

for domain_name, GeneratorClass in DOMAINS:
    print("=" * 70)
    print(f"DOMAIN: {domain_name.upper()}")
    print("=" * 70)

    try:
        gen = GeneratorClass()
        print(f"  Generator instantiated successfully")
    except Exception as e:
        print(f"  FAILED to instantiate: {e}")
        traceback.print_exc()
        continue

    domain_results = []

    for cfg in TEST_CONFIGS:
        num_keys = cfg["num_keys"]
        num_updates = cfg["num_updates"]
        label = cfg["label"]
        print(f"\n  Config: {label}")

        trials_ok = 0
        total_errors = 0
        total_warnings = 0

        for seed in range(5):  # 5 trials per config
            try:
                trial = gen.generate_trial(
                    num_keys=num_keys,
                    num_updates=num_updates,
                    condition="both",  # generate both RI and PI
                    seed=seed * 100 + num_keys * 10 + num_updates
                )

                errors = validate_trial(trial, domain_name)
                warnings = check_realism(trial, domain_name)

                if errors:
                    total_errors += len(errors)
                    print(f"    seed={seed}: ERRORS: {errors}")
                elif warnings:
                    total_warnings += len(warnings)
                    print(f"    seed={seed}: OK (with warnings: {warnings})")
                    trials_ok += 1
                else:
                    trials_ok += 1

            except Exception as e:
                total_errors += 1
                print(f"    seed={seed}: EXCEPTION: {e}")
                traceback.print_exc()

        print(f"    Result: {trials_ok}/5 OK, {total_errors} errors, {total_warnings} warnings")
        domain_results.append({
            "config": label, "ok": trials_ok, "errors": total_errors, "warnings": total_warnings
        })

    all_results[domain_name] = domain_results

# ─── Print one example trial per domain ────────────────────────────────────

print("\n" + "=" * 70)
print("EXAMPLE TRIALS (for human realism review)")
print("=" * 70)

for domain_name, GeneratorClass in DOMAINS:
    print(f"\n{'─' * 50}")
    print(f"EXAMPLE: {domain_name.upper()} (3 keys, 5 updates)")
    print(f"{'─' * 50}")
    try:
        gen = GeneratorClass()
        trial = gen.generate_trial(num_keys=3, num_updates=5, condition="both", seed=42)
        print("\nNARRATIVE:")
        print(trial.get("narrative", "NO NARRATIVE")[:2000])  # First 2000 chars
        print("\nQUESTIONS:")
        q = trial.get("questions", {})
        if "RI" in q:
            print(f"  RI: {q['RI'].get('question', '?')}")
            print(f"      Answer: {q['RI'].get('expected_answer', '?')}")
        if "PI" in q:
            print(f"  PI: {q['PI'].get('question', '?')}")
            print(f"      Answer: {q['PI'].get('expected_answer', '?')}")
        print("\nENTITY TRACKING:")
        for entity_key, values in list(trial.get("entity_tracking", {}).items())[:3]:
            print(f"  {entity_key}: {values}")
        print(f"\nCONFIG: {json.dumps(trial.get('config', {}), indent=2)}")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()

# ─── Summary ───────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for domain_name, results in all_results.items():
    total_ok = sum(r["ok"] for r in results)
    total_trials = sum(5 for r in results)
    total_errors = sum(r["errors"] for r in results)
    print(f"  {domain_name}: {total_ok}/{total_trials} OK, {total_errors} total errors")
print()
