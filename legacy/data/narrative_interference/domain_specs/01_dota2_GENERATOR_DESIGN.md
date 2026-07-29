# Dota 2 Narrative Interference Generator — System Design

## 1. What We're Building

A function: `generate_dota_trial(num_keys, num_updates, condition, seed) → (narrative, question, expected_answer, entity_tracking)`

Where:
- `num_keys` = number of heroes tracked in the narrative (2-10)
- `num_updates` = number of times each hero's tracked attribute changes (3-50)
- `condition` = "RI" (ask about first value) or "PI" (ask about last value)
- `seed` = random seed for reproducibility

## 2. Core Design Decisions

### What is the "tracked attribute"?

For maximum interference, we track **the SAME attribute type** for all heroes. If we track gold for hero A but kills for hero B, the model can trivially distinguish them by value type. We want all values to look similar so the model must track WHICH entity had WHICH value at WHICH time.

**Best tracked attribute candidates (all numeric, similar ranges, frequently changing):**
- **Gold / Net worth** — numbers in 1000-30000 range, changes every event
- **Kill score (K/D/A)** — small numbers, changes on kills/deaths/assists
- **Last hits (CS)** — numbers in 0-400 range, monotonically increasing
- **Level** — 1-30, monotonically increasing
- **Items** — categorical (item names), changes on purchase

**Recommended:** Use gold/net_worth as the primary tracked attribute. It changes most frequently, has a wide range, and the numbers are sufficiently similar across heroes to create interference (all heroes have gold in the thousands).

But we should ALSO support other attributes as options — the generator should let you pick which attribute to track.

### How do other attributes fit in?

Non-tracked attributes appear in the narrative as **filler/context**. They make the narrative realistic but aren't queried. Example:

> "At 14:32, Anti-Mage completed his Battle Fury and his **net worth jumped to 8,400**. He'd been farming efficiently with **186 last hits** at this point. Meanwhile, Invoker had just hit **level 12** in the mid lane, and his **net worth stood at 7,100** after picking up a kill on the enemy support."

Here gold/net_worth (bold) is the tracked attribute. Last hits and level are filler.

## 3. State Machine Design

### 3.1 Match Initialization

```
Input: num_keys heroes, seed
Output: initial state for all heroes

Steps:
1. Pick num_keys heroes from the 127-hero pool (no duplicates)
2. Split into two teams (Radiant/Dire): ceil(num_keys/2) vs floor(num_keys/2)
3. Assign roles:
   - If team has 5 heroes: pos1 (carry), pos2 (mid), pos3 (offlane), pos4 (soft sup), pos5 (hard sup)
   - If team has fewer: assign highest-priority roles first
4. Initialize each hero's state:
   - gold = 600 (starting gold)
   - level = 1
   - kills = 0, deaths = 0, assists = 0
   - last_hits = 0
   - items = ["tango", "healing_salve", "iron_branch", "iron_branch"] (varies by role)
   - alive = True
   - net_worth = ~600
```

### 3.2 Event Types

The match is a sequence of events. Each event modifies 1-3 heroes' states.

```
EVENT TYPES:
├── farming_update      — hero's gold/CS increases from farming (passive)
├── solo_kill           — hero A kills hero B (gold transfer, K/D/A change)
├── teamfight           — multiple kills in one engagement
├── item_purchase       — hero buys item (gold decreases, net_worth stays or increases)
├── tower_kill          — team destroys tower (team gold distributed)
├── roshan_kill         — team kills Roshan (gold + Aegis)
├── level_up            — hero gains a level
├── death_and_respawn   — hero dies, respawn timer
└── barracks_kill       — team destroys barracks (super creeps)
```

### 3.3 Event Generation (the state machine)

```
def generate_next_event(state, game_time, rng):
    """Generate a plausible next event based on current game state and time."""

    # Phase determines event probabilities
    if game_time < 10:  # Laning
        weights = {
            "farming_update": 45,
            "solo_kill": 15,
            "item_purchase": 25,
            "level_up": 15,
            # No towers/Roshan/teamfights in laning phase
        }
    elif game_time < 25:  # Mid game
        weights = {
            "farming_update": 15,
            "solo_kill": 20,
            "teamfight": 15,
            "item_purchase": 20,
            "tower_kill": 10,
            "roshan_kill": 5,
            "level_up": 10,
            "death_and_respawn": 5,
        }
    else:  # Late game
        weights = {
            "teamfight": 25,
            "solo_kill": 10,
            "item_purchase": 15,
            "tower_kill": 15,
            "roshan_kill": 10,
            "barracks_kill": 10,
            "level_up": 5,
            "death_and_respawn": 10,
        }

    # Filter out impossible events
    # Can't kill Roshan if already killed recently
    # Can't destroy towers that are already destroyed
    # Can't level up heroes already at max level
    # Dead heroes can't do anything

    event_type = weighted_random_choice(weights, rng)
    event = create_event(event_type, state, game_time, rng)
    return event
```

### 3.4 Event Handlers (Constraints)

Each event handler enforces game rules:

```
FARMING_UPDATE:
  - Pick a living hero (weighted by role — pos1 farms most, pos5 least)
  - Advance game_time by 1-3 minutes
  - Add gold: pos1 gets 400-700/min, pos2 gets 350-600/min, pos3 gets 250-450/min, etc.
  - Add last_hits proportional to gold
  - Constraints: gold only increases from farming

SOLO_KILL:
  - Pick killer (living hero, weighted by role and game phase)
  - Pick victim (living hero on opposite team)
  - Constraints: killer ≠ victim, both must be alive
  - Gold change:
    - Killer gains: 125 + (victim_level × 8) + streak_bonus
    - Victim loses: net_worth / 40
  - XP change: killer gains kill XP
  - K/D/A: killer.kills++, victim.deaths++
  - Victim dies: alive = False, respawn_timer = 5 + (3.8 × level)
  - Nearby allies get assist gold/XP

TEAMFIGHT:
  - Pick 3-8 involved heroes (from both teams)
  - Simulate 2-5 kills in sequence
  - One team "wins" the fight (kills more)
  - Apply kill gold/XP to all participants
  - Constraints: can't kill dead heroes, can't have 6 kills in a 5v5

ITEM_PURCHASE:
  - Pick a hero with enough gold for a plausible next item
  - Item choice depends on:
    - Role (carry buys damage, support buys utility)
    - Current items (follow common build paths)
    - Available gold
  - Gold decreases by item cost
  - Net worth stays same (gold → item value)
  - Constraints: need enough gold, max 6 item slots

TOWER_KILL:
  - Pick which tower falls (must be destroyable — T1 before T2 before T3)
  - Team gold: 100-160 per player (depends on tier)
  - Constraints: towers fall in order (can't destroy T2 before T1 in same lane)

ROSHAN_KILL:
  - One team kills Roshan
  - Gold: ~225 per player + killer bonus
  - Drops: Aegis (+ Cheese/Shard on 2nd/3rd kills)
  - Constraints: Roshan respawn timer (8-11 min between kills)

LEVEL_UP:
  - Hero gains enough XP to reach next level
  - Constraints: sequential levels, max 30
  - Level timing depends on role (mid hits 6 first, support hits 6 last)
```

### 3.5 Gold Trajectory Constraints (Critical for Realism)

Gold trajectories must be plausible:

```
POSITION 1 (Carry) gold trajectory:
  5 min:  1800-2500
  10 min: 3500-5000
  15 min: 5500-8000
  20 min: 8000-12000
  25 min: 11000-16000
  30 min: 14000-22000
  35 min: 17000-28000
  40 min: 20000-35000

POSITION 5 (Hard Support) gold trajectory:
  5 min:  800-1200
  10 min: 1500-2500
  15 min: 2500-4000
  20 min: 3500-5500
  25 min: 4500-7000
  30 min: 6000-9000
  35 min: 7000-11000
  40 min: 8000-13000

Key constraint: Gold can DECREASE (death penalty, buyback) but generally trends up.
Item purchases don't decrease net_worth (gold converts to item value).
Deaths decrease unreliable gold by (net_worth / 40).
```

## 4. Narrative Generation

### 4.1 Story Structure

A trial narrative has this structure:

```
[MATCH HEADER]
  - Teams, hero picks, roles

[TIME SEGMENT 1: Early Game]
  - 2-5 event narrations
  - Each mentions 1-3 heroes' tracked attribute values

[TIME SEGMENT 2: Mid Game]
  - 3-8 event narrations

[TIME SEGMENT 3: Late Game]
  - 2-5 event narrations

[Optional: MATCH OUTCOME]
```

The number of segments and events per segment scales with num_updates. For num_updates=5, we might have 2 segments with 2-3 events each. For num_updates=30, we'd have 5-6 segments with 5-6 events each.

### 4.2 Narrative Templates

Each event type has multiple templates. Templates are parameterized with hero names, values, locations, and times.

```python
# ── MATCH HEADER TEMPLATES ──

HEADER_TEMPLATES = [
    "Game {game_id} of the {tournament_name}: {team1_name} ({team1_heroes_str}) "
    "versus {team2_name} ({team2_heroes_str}). {team1_name} secured first pick with "
    "{first_pick}, while {team2_name} responded with a {response_description}.",

    "A {match_adjective} matchup in {tournament_name} as {team1_name} draft "
    "{team1_heroes_str} against {team2_name}'s {team2_heroes_str}. "
    "The draft gives {favored_team} a slight edge in {advantage_type}.",
]

# ── FARMING UPDATE TEMPLATES ──

FARMING_TEMPLATES = [
    "By the {time}-minute mark, {hero}'s efficient farming in the {location} had "
    "pushed {pronoun} net worth to {gold} gold.",

    "{hero} continued to find farm in the {location}, sitting at {gold} gold with "
    "{cs} last hits at {time} minutes.",

    "With {cs} creeps secured by {time} minutes, {hero}'s gold stood at {gold} — "
    "{comparison} for a {role} at this stage of the game.",

    "The {location} was proving fruitful for {hero}, who had accumulated {gold} gold "
    "by the {time}-minute mark. {filler_detail}",
]

# ── KILL TEMPLATES ──

KILL_TEMPLATES = [
    "At {time}:{seconds}, {killer} found {victim} alone near {location} and secured "
    "the kill. The {kill_gold}-gold bounty pushed {killer}'s net worth to {killer_gold}. "
    "{victim}'s gold dropped to {victim_gold} with a {respawn}s respawn timer.",

    "A rotation from {killer} caught {victim} off-guard at {location}. The kill "
    "brought {killer}'s score to {k}/{d}/{a} and a net worth of {killer_gold} gold. "
    "{victim} fell to {victim_gold} gold, facing {respawn} seconds on the sideline.",

    "{killer} executed a {kill_description} on {victim} at {location}, earning "
    "{kill_gold} gold for the takedown. {killer} now has {killer_gold} gold while "
    "{victim}'s death penalty left {pronoun} at {victim_gold}.",
]

# ── ITEM PURCHASE TEMPLATES ──

ITEM_TEMPLATES = [
    "{hero} returned to base and completed {item} for {cost} gold, leaving "
    "{pronoun} with {remaining_gold} gold in reserve. Net worth: {net_worth}.",

    "A quick trip to the shop saw {hero} pick up {item}. After spending {cost} gold, "
    "{pronoun} net worth sat at {net_worth}. {item_significance}.",

    "With {item} now in {pronoun} inventory, {hero}'s net worth reached {net_worth}. "
    "The {cost}-gold purchase was funded by {funding_source}.",
]

# ── TEAMFIGHT TEMPLATES ──

TEAMFIGHT_TEMPLATES = [
    "A massive engagement broke out at {location} at the {time}-minute mark. "
    "{winning_team} came out ahead, trading {win_kills} for {loss_kills}. "
    "{mvp} was the standout performer, netting {mvp_kills} kills and climbing "
    "to {mvp_gold} gold. On the losing side, {worst_performer}'s net worth "
    "dropped to {worst_gold} after the death penalty.",

    "Chaos erupted near {location} at {time} minutes. {engagement_description}. "
    "When the dust settled, {results_description}. "
    "{hero1} emerged with {hero1_gold} gold, while {hero2} fell to {hero2_gold}.",
]

# ── TOWER KILL TEMPLATES ──

TOWER_TEMPLATES = [
    "{team} pushed down the {lane} tier {tier} tower at {time} minutes. "
    "The {tower_gold}-gold team bounty boosted everyone — notably {hero}, "
    "whose gold rose to {hero_gold}.",

    "The {lane} lane tier {tier} tower fell to {team} at the {time}-minute mark, "
    "distributing {tower_gold} gold to each player. {hero}'s net worth climbed "
    "to {hero_gold}.",
]

# ── ROSHAN TEMPLATES ──

ROSHAN_TEMPLATES = [
    "{team} secured Roshan at {time} minutes, with {carrier} claiming the Aegis. "
    "The bonus gold pushed {hero}'s net worth to {hero_gold}. "
    "{other_hero} reached {other_gold} from the shared bounty.",

    "Roshan fell to {team} at the {time}-minute mark — their {nth} Roshan kill. "
    "{drops_description}. {hero} climbed to {hero_gold} gold.",
]
```

### 4.3 Location Pool

Dota 2 map locations for flavor:

```python
LOCATIONS = [
    # Lanes
    "the top lane", "the mid lane", "the bottom lane",
    "the safe lane", "the offlane",

    # Jungle areas
    "the Radiant jungle", "the Dire jungle",
    "the Radiant ancients", "the Dire ancients",
    "the triangle", "the large camp",

    # Key landmarks
    "the Roshan pit", "the river", "the Dire shrine",
    "the Radiant shrine", "the Radiant outpost", "the Dire outpost",
    "the top rune spot", "the bottom rune spot",

    # Tower areas
    "the tier 1 tower", "the tier 2 tower", "the high ground",

    # Specific areas
    "the ward cliff", "the secret shop",
]
```

### 4.4 Filler / Context Sentences

Between tracked-attribute updates, we add context that mentions non-tracked attributes:

```python
FILLER_TEMPLATES = [
    "{hero} had reached level {level} by this point, with {ability} now maxed out.",
    "The draft advantage was showing — {hero}'s {item} timing was {comparison} minutes {direction} of average.",
    "{hero}'s score line read {k}/{d}/{a}, {assessment} for a {role}.",
    "Ward vision from {support} revealed {info}, giving {team} the information advantage.",
    "{hero} had already completed {item1} and was working toward {item2}.",
    "The gold graph showed {team} with a {gold_lead}-gold lead at this point.",
]
```

## 5. Controlling num_keys and num_updates

### num_keys (number of heroes mentioned)

- num_keys=2: 1v1 mid lane duel narrative
- num_keys=3-4: one team's core heroes
- num_keys=5: one full team
- num_keys=6-8: both teams' cores
- num_keys=10: full 5v5

Each hero appears roughly equally in the narrative.

### num_updates (state changes per hero)

This determines story length and complexity:

| num_updates | Events per hero | ~Word count | Game time covered |
|-------------|----------------|-------------|-------------------|
| 3 | 3 | 200-400 | ~10-15 min window |
| 5 | 5 | 400-700 | ~15-25 min window |
| 7 | 7 | 600-1000 | ~20-30 min window |
| 10 | 10 | 900-1500 | Full match |
| 15 | 15 | 1500-2500 | Detailed full match |
| 20+ | 20+ | 2000-4000 | Very detailed match |

### Interleaving Strategy

Events naturally interleave because different events affect different heroes:

```
Event 1: Anti-Mage farms (AM gold: 1800 → 2400)
Event 2: Invoker gets a kill (Invoker gold: 2100 → 2800)
Event 3: Axe dies (Axe gold: 1200 → 1050)
Event 4: Anti-Mage buys item (AM gold: 2400 → 2600 net worth, but gold drops)
Event 5: Lina farms (Lina gold: 1500 → 2000)
Event 6: Invoker buys item (Invoker gold: 2800 → 1200 remaining)
...
```

The constraint: no hero should be updated twice in a row (same as the KV dataset interleaving).

## 6. Distinguishing Setups

To generate many unique trials at the same (num_keys, num_updates):

### Variation dimensions:
1. **Hero draft** — 127 choose 10 = astronomically many combinations
2. **Team composition** — same heroes, different role assignments
3. **Game narrative** — different event sequences (kill-heavy vs farm-heavy vs push-heavy)
4. **Game pace** — stomp (one-sided) vs close (back-and-forth)
5. **Queried hero** — which hero the RI/PI question asks about
6. **Queried attribute** — gold vs kills vs level vs items
7. **Tournament/team names** — purely cosmetic variation
8. **Timestamp window** — early game only vs full match vs late game focus

### Example distinguishing setups:

```python
GAME_ARCHETYPES = [
    "stomp_radiant",      # Radiant wins decisively, pos1 snowballs
    "stomp_dire",         # Dire wins decisively
    "comeback_radiant",   # Dire leads early, Radiant comes back
    "comeback_dire",      # Radiant leads early, Dire comes back
    "close_game",         # Back and forth, close gold graphs
    "farmfest",           # Few kills, high CS, late game focused
    "bloodbath",          # Many kills, aggressive playstyle
    "split_push",         # One team plays rat dota, split pushing
    "roshan_centric",     # Multiple Roshan fights define the game
    "base_race",          # Both teams pushing simultaneously
]
```

Each archetype produces different gold curves and event distributions.

## 7. Team & Tournament Name Pools

```python
TEAM_NAMES = [
    "Storm Vanguard", "Eclipse Gaming", "Phantom Regiment", "Iron Crown",
    "Nebula Esports", "Shadow Crest", "Arctic Wolves", "Solar Flux",
    "Crimson Tide", "Void Reapers", "Thunder Legion", "Crystal Dominion",
    "Obsidian Order", "Frost Sentinel", "Blaze Horizon", "Neon Dynasty",
    "Titan Force", "Dark Meridian", "Apex Legends", "Nova Strike",
    "Steel Phoenix", "Lunar Eclipse", "Storm Breakers", "Ember Rising",
    # ... 50+ more
]

TOURNAMENT_NAMES = [
    "The Meridian Cup", "Vanguard Championship", "Apex Invitational",
    "The Crown Series", "Horizon Masters", "Eclipse League Season 4",
    "The Nexus Tournament", "Crimson Open", "Steel Summit",
    "Dynasty Championship", "The Forge Invitational",
    # ... 20+ more
]
```

## 8. Question Templates

```python
RI_TEMPLATES = [
    "What was {hero}'s net worth when first mentioned in this match?",
    "What was {hero}'s gold at the first reference in the game?",
    "At {hero}'s first appearance in the narrative, what was {pronoun} net worth?",
]

PI_TEMPLATES = [
    "What was {hero}'s net worth at the most recent update?",
    "In the last mention of {hero}'s gold, what was the value?",
    "What was {hero}'s final recorded net worth?",
]
```

## 9. Output Format

Same as the existing narrative dataset:

```json
{
    "id": "dota2_001",
    "domain": "dota2",
    "num_keys": 5,
    "num_updates": 7,
    "narrative": "Game 1 of the Meridian Cup: Storm Vanguard (Anti-Mage, Lina, ...) ...",
    "questions": {
        "RI": {
            "question": "What was Anti-Mage's net worth when first mentioned?",
            "expected_answer": "1850",
            "target_entity": "Anti-Mage",
            "target_attribute": "net_worth"
        },
        "PI": {
            "question": "What was Anti-Mage's most recent net worth?",
            "expected_answer": "18400",
            "target_entity": "Anti-Mage",
            "target_attribute": "net_worth"
        }
    },
    "entity_tracking": {
        "Anti-Mage / net_worth": ["1850", "3400", "5200", "8100", "11300", "14700", "18400"],
        "Lina / net_worth": ["2100", "2800", "4200", "5600", "7100", "8900", "10500"],
        "Tidehunter / net_worth": ["1200", "2400", "3100", "4800", "5600", "7200", "8900"],
        "Invoker / net_worth": ["2300", "3500", "5800", "7400", "9200", "11000", "13600"],
        "Crystal Maiden / net_worth": ["900", "1400", "2100", "2800", "3500", "4200", "5100"]
    }
}
```

## 10. Implementation Order

1. **Hero database** — names, roles, typical item builds
2. **Item database** — names, costs, categories (starting/early/core/luxury)
3. **State class** — HeroState with all mutable attributes
4. **Event generators** — one function per event type, each enforcing constraints
5. **Event scheduler** — decides which event happens next based on game phase
6. **Narrative renderer** — converts event sequence into prose using templates
7. **Trial generator** — orchestrates everything into a single trial output
8. **Validation** — verify all tracked values appear in narrative, RI/PI answers correct

## 11. Design Decisions (Resolved)

### Q1: Track net_worth or gold?
**DECISION: Randomly sample.** Sometimes gold (volatile — drops on item purchase and death), sometimes net_worth (smoother — only drops on death). The state machine tracks BOTH internally; the narrative renderer picks which to mention based on the sampled config. Gold is harder for the LLM because it goes up AND down.

### Q2: Same attribute for all heroes, or different per hero?
**DECISION: Randomly sample.**
- ~50% of trials: ALL heroes track the SAME attribute (pure interference — all values look similar, maximum confusion)
- ~50% of trials: heroes track MIXED attributes (more realistic narrative, but easier because value types differ — gold numbers vs kill counts vs item names)

When mixed, attribute assignment should fit the role: carries track gold/net_worth, supports track assists/wards, mids track kills/level.

**IMPORTANT:** Mention counts do NOT need to be equal across heroes. The narrative naturally mentions some heroes more than others. We track how many times each hero's attribute was mentioned so we have ground truth. The `num_updates` in config is a TARGET/BUDGET, not a hard constraint. Natural variation is fine — what matters is the ground truth entity_tracking records exactly what appeared in text.

### Q3: Filler budget?
**DECISION: Sample from 4 budget levels.**

| Budget | Filler ratio | Description |
|--------|-------------|-------------|
| `minimal` | 0-10% | Almost every sentence has a tracked value |
| `light` | 20-30% | Some game context between updates |
| `medium` | 40-50% | Realistic commentary with analysis |
| `heavy` | 60-70% | Dense narrative, lots of non-tracked details |

Each trial randomly samples a budget. This creates a built-in difficulty axis: minimal filler = easy (like KV pairs with prose), heavy filler = hard (realistic but buried values).

### Q4: Full match or window?
**DECISION: Determined by num_updates.**

| num_updates | Coverage | Narrative scope |
|-------------|----------|-----------------|
| 3-5 | ~10-15 min window | Early game recap or teamfight sequence |
| 7-10 | ~20-30 min | Half-match or mid-game focus |
| 15-20 | Full match | Complete match recap |
| 30+ | Very detailed full match | Minute-by-minute commentary |

For windows: initialize hero states to realistic values for the starting timestamp (not 600 gold). The state machine uses role-specific gold trajectories from the mechanics doc to set plausible initial conditions.

### Q5: Caster or analyst voice?
**DECISION: Sample both, 50/50.**
- **Analyst** (past tense, structured): post-game recap style
- **Caster** (present tense, dramatic): play-by-play, teamfight sequences

## 12. Trial Config Schema (Final)

```python
@dataclass
class DotaTrialConfig:
    num_keys: int              # 2-10 heroes
    num_updates: int           # 3-50 state changes per hero
    condition: str             # "RI" or "PI"
    seed: int                  # reproducibility
    tracked_attribute: str     # "gold" | "net_worth" | "kills" | "deaths" | "kda" |
                               # "last_hits" | "level" | "gpm" | "latest_item" | "mixed"
    attribute_mode: str        # "same" (all heroes same attr) | "mixed" (per-hero)
    filler_budget: str         # "minimal" | "light" | "medium" | "heavy"
    voice: str                 # "caster" | "analyst"
    match_scope: str           # "full" | "window"
    game_archetype: str        # "stomp" | "comeback" | "close" | "farmfest" | "bloodbath"
    queried_hero_idx: int      # which hero to ask about (0 to num_keys-1)
```

When `seed` is provided, all other fields can be auto-sampled deterministically:
```python
def auto_config(num_keys, num_updates, condition, seed):
    rng = Random(seed)
    return DotaTrialConfig(
        num_keys=num_keys,
        num_updates=num_updates,
        condition=condition,
        seed=seed,
        tracked_attribute=rng.choice(["gold", "net_worth", "kills", "last_hits", "level", "gpm"]),
        attribute_mode=rng.choice(["same", "mixed"]),
        filler_budget=rng.choice(["minimal", "light", "medium", "heavy"]),
        voice=rng.choices(["analyst", "caster"], weights=[70, 30])[0],
        match_scope="full" if num_updates >= 15 else rng.choice(["full", "window"]),
        game_archetype=rng.choice(["stomp", "comeback", "close", "farmfest", "bloodbath"]),
        queried_hero_idx=rng.randint(0, num_keys - 1),
    )
```
