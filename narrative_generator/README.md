# Narrative Generator

Template-based narrative trial generators for memory interference experiments. Each domain produces naturalistic stories with embedded key-value state changes, enabling RI (retroactive interference) and PI (proactive interference) evaluation on LLMs.

## Structure

```
narrative_generator/
├── __init__.py          # Package exports
├── base.py              # Abstract base class (NarrativeTrialGenerator)
├── README.md            # This file
└── dota2/               # Dota 2 match commentary domain
    ├── __init__.py
    ├── generator.py     # DotaTrialGenerator implementation
    └── data/
        ├── dota2_heroes.json     # 127 heroes with roles
        ├── dota2_items.json      # Items by tier + role builds
        ├── dota2_templates.json  # 90+ narrative templates
        └── dota2_names.json      # Teams, tournaments, locations
```

## Quick Start

```python
from narrative_generator import DotaTrialGenerator

gen = DotaTrialGenerator()

# Single trial
trial = gen.generate_trial(
    num_keys=3,       # heroes to track
    num_updates=10,   # target value mentions per hero
    condition="RI",   # "RI" or "PI"
    seed=42,          # fully deterministic
)

print(trial["narrative"])        # the story text
print(trial["question"])         # the recall question
print(trial["expected_answer"])  # ground truth answer
print(trial["entity_tracking"])  # {hero: [val1, val2, ...]} for all tracked values

# Validate ground truth
gen.validate_trial(trial)  # raises AssertionError on failure
```

## Batch Generation

```python
gen = DotaTrialGenerator()

# Generate full grid: 5 key levels × 8 update levels × 2 conditions × 30 trials
batch = gen.generate_batch(
    key_levels=[2, 3, 5, 7, 10],
    update_levels=[1, 3, 5, 10, 20, 30, 40, 50],
    trials_per_cell=30,
    seed_start=100000,
    tracked_attribute="gold",   # optional: fix attribute across all trials
    attribute_mode="same",      # optional: all heroes track same attribute
)

# Save to data/narrative_interference/dota2/
path = gen.save_batch(batch, tag="gold_same")
# → data/narrative_interference/dota2/dota2_gold_same_20260228_120000.json
```

## generate_trial() Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `num_keys` | int | Yes | Number of heroes tracked (2-10) |
| `num_updates` | int | Yes | Target tracked-value mentions per hero (1-50) |
| `condition` | str | Yes | `"RI"` (recall first value) or `"PI"` (recall last value) |
| `seed` | int | Yes | Random seed — same seed = same trial |

### Optional Config Overrides (**kwargs)

Pass any of these to override the auto-sampled config:

| Override | Values | Default | Description |
|----------|--------|---------|-------------|
| `tracked_attribute` | See list below | Random | Which stat to track and query |
| `attribute_mode` | `"same"`, `"mixed"` | Random | All heroes track same attribute, or each gets role-appropriate one |
| `filler_budget` | `"minimal"`, `"light"`, `"medium"`, `"heavy"` | Random | Amount of non-tracked filler text between events |
| `voice` | `"analyst"`, `"caster"` | Random | Narration style — analytical vs play-by-play |
| `match_scope` | `"full"`, `"window"` | Random | Full game (0 min) or mid-game window |
| `game_archetype` | See list below | Random | Match narrative arc |
| `queried_hero_idx` | int (0 to num_keys-1) | Random | Which hero gets the recall question |

### Tracked Attributes

```
gold, net_worth, kills, deaths, assists, kda_string,
last_hits, level, gpm, xpm, total_damage_dealt, total_damage_taken
```

### Game Archetypes

| Archetype | Description |
|-----------|-------------|
| `stomp` | One team dominates from start |
| `comeback` | Early leader loses control, other team rallies |
| `close` | Back-and-forth, even game |
| `farmfest` | Low-kill, high-economy game |
| `bloodbath` | High-kill, aggressive game |
| `split_push` | Tower-focused, map control game |

### Filler Budget

Controls narrative density — more filler = longer stories, harder recall:

| Budget | Insert Probability | Sentences per Insert |
|--------|-------------------|---------------------|
| `minimal` | 5% | 1 |
| `light` | 25% | 1 |
| `medium` | 45% | 1-2 |
| `heavy` | 65% | 1-3 |

## generate_batch() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `key_levels` | list[int] | `[2, 3, 5, 7, 10]` | Grid: num_keys values |
| `update_levels` | list[int] | `[1, 3, 5, 10, 20, 30, 40, 50]` | Grid: num_updates values |
| `trials_per_cell` | int | `30` | Trials per (keys, updates, condition) cell |
| `seed_start` | int | `100000` | First seed (increments by 1 per trial) |
| `**kwargs` | | | Config overrides (same as generate_trial) |

Total trials = `len(key_levels) × len(update_levels) × 2 × trials_per_cell`

Default grid: 5 × 8 × 2 × 30 = **2,400 trials**

## Output Schema

Each trial is a dict with:

```python
{
    "narrative": str,           # The full story text
    "question": str,            # "What was the INITIAL/LAST reported {attr} for {hero}?"
    "expected_answer": str,     # Ground truth value
    "condition": "RI" | "PI",
    "num_keys": int,
    "num_updates": int,
    "seed": int,

    # Ground truth tracking
    "entity_tracking": {        # {hero: [v1, v2, ..., vN]} — values in narrative order
        "Anti-Mage": ["2,450", "5,100", "8,300"],
        ...
    },
    "queried_entity": str,      # Which hero the question asks about
    "ri_answer": str,           # First tracked value for queried hero
    "pi_answer": str,           # Last tracked value for queried hero

    # Reproducibility
    "config": {
        "tracked_attribute": str,
        "attribute_mode": str,
        "filler_budget": str,
        "voice": str,
        "match_scope": str,
        "game_archetype": str,
        "queried_hero_idx": int,
        "start_time": float,
        "team1": str,
        "team2": str,
        "tournament": str,
        "radiant_heroes": [[hero, role], ...],
        "dire_heroes": [[hero, role], ...],
    },

    # Diagnostics
    "mention_counts": {hero: int, ...},  # Times each hero is mentioned
    "word_count": int,
    "full_state_log": [...]              # Every state change event
}
```

## Adding a New Domain

1. Create `narrative_generator/<domain>/`
2. Subclass `NarrativeTrialGenerator` from `base.py`
3. Implement `generate_trial()` and `validate_trial()`
4. Add data files under `<domain>/data/`
5. Export from `<domain>/__init__.py` and the top-level `__init__.py`

Domain specs for future generators: `data/narrative_interference/domain_specs/`

## Validation

Every trial passes `validate_trial()` which checks:
- All values in `entity_tracking` appear in the narrative text
- `ri_answer != pi_answer` (interference is real)
- Queried hero has >= 2 tracked value mentions
- `expected_answer` matches `ri_answer` (for RI) or `pi_answer` (for PI)
