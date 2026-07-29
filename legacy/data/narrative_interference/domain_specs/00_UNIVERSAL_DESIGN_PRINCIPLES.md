# Universal Design Principles for Narrative Interference Dataset

These principles apply to ALL domains (Dota 2, PUBG, Hospital ICU, etc.).
Learned from the Dota 2 design discussion. Update as we refine.

---

## 1. Mention Counts ≠ Equal

Mention counts do NOT need to be equal across entities. The narrative naturally mentions
some entities more than others. We track how many times each entity's attribute was
mentioned so we have ground truth. The `num_updates` in config is a TARGET/BUDGET.

However: don't make it trivially imbalanced (e.g., hero A mentioned 10 times, hero B
mentioned once). Roughly similar is fine, exact equality is not required.

## 2. Tracked Attributes — Sample Randomly from Large Pool

Each domain should have 15-25+ trackable attributes. When sampling a trial, randomly
pick which attribute(s) to track. More options = more variety = harder for model to
develop shortcuts.

For maximum interference: track the SAME attribute type for all entities (all gold
values look similar → hard to distinguish). For variety: mix attribute types.

Sample 50/50 same vs mixed.

## 3. Voice Style — 50/50

- **Analyst voice** (past tense, structured, post-game recap)
- **Caster voice** (present tense, dramatic, play-by-play)

Sample 50/50. Both are valid narrative styles.

## 4. Filler Budgets — 4 Levels

| Budget | Filler ratio | Description |
|--------|-------------|-------------|
| `minimal` | 0-10% | Almost every sentence has a tracked value |
| `light` | 20-30% | Some context between updates |
| `medium` | 40-50% | Realistic commentary with analysis |
| `heavy` | 60-70% | Dense narrative, lots of non-tracked details |

Sample randomly per trial. This creates a built-in difficulty axis.

## 5. Ground Truth — Track EVERYTHING

Every generated trial must have complete ground truth:

```json
{
    "config": { ... },           // Full config for reproducibility
    "narrative": "...",          // What the model sees
    "questions": { "RI": ..., "PI": ... },
    "entity_tracking": {         // ONLY values that appear in narrative
        "Entity A / attribute": ["v1", "v2", "v3"],
        "Entity B / attribute": ["v1", "v2"]
    },
    "full_state_log": [          // ALL state changes (even unmentioned)
        {"time": ..., "event": ..., "entity": ..., "attr": ...,
         "old_value": ..., "new_value": ..., "mentioned_in_narrative": true/false}
    ],
    "mention_counts": {          // How many times each entity appears
        "Entity A": 7,
        "Entity B": 3
    }
}
```

The `full_state_log` enables analysis beyond just RI/PI accuracy:
- Does mention frequency affect recall?
- Does position in narrative matter?
- Are earlier-mentioned entities easier?
- Does filler between mentions affect interference?

## 6. Reproducibility — Seed Controls Everything

`config + seed` must be sufficient to regenerate the exact same trial.
All randomization derives from a single `Random(seed)` instance.
Save the full config with every trial so experiments can be reproduced.

## 7. Validation — Automated Checks

Every generated trial must pass:
1. Every value in `entity_tracking` appears EXACTLY in the narrative text
2. RI expected answer = first value in tracking for queried entity
3. PI expected answer = last value in tracking for queried entity
4. RI ≠ PI (different first and last values — otherwise no discrimination)
5. `entity_tracking` keys match expected entities
6. No duplicate consecutive values (if gold is 5000 twice in a row, that's not an update)

## 8. Question Generation — Decouple from Narrative

**Key insight: Questions are CHEAP, narratives are EXPENSIVE.**

Once a narrative exists with full ground truth (entity_tracking + full_state_log),
we can generate UNLIMITED questions from it after the fact:

- Different entities: "What was Anti-Mage's gold?" vs "What was Invoker's gold?"
- Different attributes: "What was Anti-Mage's gold?" vs "What was Anti-Mage's level?"
- RI vs PI: first vs last mention
- Different phrasings: "when first mentioned" vs "at the earliest reference"
- Multi-hop: "Which hero had the highest gold at first mention?"
- Comparison: "Did Anti-Mage or Invoker have more gold at last mention?"

Each combination = separate trial/sample. One narrative → dozens of data points.

**Design decision: focus on generating rich narratives with complete ground truth.
Question generation is a separate, lightweight step that runs AFTER narrative generation.
Don't over-design question format now.**

For the basic experiments, simple format works:
- RI: "What was {entity}'s {attribute} when first mentioned?"
- PI: "What was {entity}'s most recent {attribute}?"

## 9. Domain-Specific State Machine

Each domain needs its own state machine that:
1. Initializes entity states realistically for the chosen time window
2. Generates plausible events (weighted by game phase / time of day / etc.)
3. Updates states according to domain rules (constraints)
4. Filters impossible events (can't kill dead heroes, can't sell items you don't have)
5. Produces an event log that the narrative renderer converts to prose

The state machine and the narrative renderer are SEPARATE concerns.
State machine produces events. Renderer converts events to text.

## 10. Template Variety

Each event type needs 8-10+ narrative templates to avoid repetition within a single
story. Templates are parameterized with entity names, values, locations, times.

The template system should support:
- Multiple sentence structures for the same event
- Variable-length descriptions (short for minimal filler, long for heavy filler)
- Filler sentences that mention non-tracked attributes for realism

## 11. What Applies Across Domains

These choices are UNIVERSAL — same for Dota 2, PUBG, Hospital ICU, etc.:
- Mention count tracking (not forced equality)
- Filler budget levels (minimal/light/medium/heavy)
- Voice styles (analyst/caster or equivalent per domain)
- Ground truth structure (entity_tracking + full_state_log)
- Seed-based reproducibility
- Automated validation
- Separate state machine and narrative renderer

What's DOMAIN-SPECIFIC:
- Entity types and names
- Attribute lists and value ranges
- State machine constraints and event types
- Narrative templates and domain vocabulary
- Game phase / time progression model
