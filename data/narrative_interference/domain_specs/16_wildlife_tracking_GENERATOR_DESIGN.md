# Wildlife Tracking Narrative Interference Generator — System Design

> **AUTHORITATIVE SOURCE NOTE**: For species biological data (weight ranges, movement rates,
> seasonal patterns, mortality rates, reproductive timing), the file
> `16_wildlife_tracking_SPECIES_PARAMETERS.md` is the authoritative reference and supersedes
> any species data in the mechanics file or inline values in this design document. When the
> two conflict, SPECIES_PARAMETERS wins. This design file is authoritative for generator
> architecture, event logic, narrative templates, and config schemas.

## 1. What We're Building

A function: `generate_wildlife_trial(num_keys, num_updates, condition, seed) → (narrative, question, expected_answer, entity_tracking)`

Where:
- `num_keys` = number of tagged animals tracked in the narrative (2-15)
- `num_updates` = number of times each animal's tracked attribute changes across observations (3-30)
- `condition` = "RI" (ask about first value) or "PI" (ask about last value)
- `seed` = random seed for reproducibility

## 2. Core Design Decisions

### What is the "tracked attribute"?

Same principle as Dota 2: for maximum interference, track the SAME attribute type across all animals. All weight values look similar → model must track WHICH animal had WHICH weight at WHICH observation.

**Tracked attribute candidates (pool of 20+):**

**Numeric, continuous:**
- **weight_kg** — best primary candidate. Within-species range is tight (wolf: 32-65kg). Changes gradually across weeks. Similar to Dota's gold: the "money" of wildlife studies.
- **daily_movement_km** — varies 3-50km depending on season/activity. Good overlap across species.
- **elevation_m** — 1500-3500m in mountain study areas. Similar ranges across all animals.
- **distance_from_den_km** — 0-80km. Seasonally variable. Only meaningful for denning species.
- **territory_size_km2** — 50-1300km² for wolves. Changes when pack splits/merges.
- **collar_battery_pct** — 0-100%. Only decreases (or jumps to 100 on replacement). Monotonic decay.
- **distance_traveled_today_km** — 0-50km. High variance day to day.

**Numeric, integer/count:**
- **pup_count** — 0-8. Changes in spring (birth) and through summer (mortality). Only relevant for breeding females.
- **kill_count_observed** — cumulative integer. Increases only.
- **days_since_last_sighting** — integer, resets on each observation.
- **scent_post_count** — territorial marking frequency. Integer per survey.

**Ordinal:**
- **body_condition_score** — 1 to 9 scale (Heinze system). Assessed visually. Changes slowly.

**Categorical (named values) — each pool must have 15-25+ distinct values for interference:**

- **location_zone** — named geographic areas (15-30 per study area). Changes frequently. Pool size is large by design (see Section 4.4).

- **activity_state** (~22 values):
  resting at den site, resting in open meadow, bedded in timber, bedded on ridge,
  traveling on ridgeline, traveling through drainage, traveling along river corridor,
  hunting (stalking), hunting (active chase), hunting (ambush),
  feeding at fresh kill site, feeding at old carcass (scavenging),
  scent-marking territory boundary, howling (stationary), howling while traveling,
  at rendezvous site with pups, playing/socializing with pack,
  swimming/crossing river, digging at rodent burrow,
  patrolling territory perimeter, on exposed ridge scanning,
  crossing road corridor, at water source (drinking)

- **health_status** (~20 values):
  healthy (excellent condition), healthy (average condition),
  minor limp (left hind), minor limp (right fore), minor limp (left fore),
  bite wound (shoulder, healing), bite wound (flank, fresh), bite wound (neck, healing),
  sarcoptic mange (early — patchy hair loss <20%), sarcoptic mange (moderate — 20-50% hair loss),
  sarcoptic mange (advanced — >50% hair loss, emaciated),
  laceration (healing, scabbed), emaciated (ribs and spine prominent),
  porcupine quills in muzzle, eye injury (left, clouded), eye injury (right, swollen shut),
  snare/trap injury (leg, scarred), frostbite on ears (healed), dental issues (broken canine),
  recovering post-whelping (thin but improving)

- **pack_membership** — named packs (25+ pack names in pool, see Section 7) + "disperser (solo)" + "disperser (with unnamed companion)" + "new pair (forming pack)." Pool grows with study area.

- **reproductive_status** (~14 values):
  non-breeding yearling, non-breeding subordinate adult, non-breeding (aged/post-reproductive),
  courting (pair bonding observed), breeding confirmed (observed tie),
  pregnant (early — no visible change), pregnant (late — visibly gravid),
  denning (pre-whelp, restricted movement), nursing (pups in den, <3 weeks),
  nursing (pups at den entrance, 3-6 weeks), nursing (pups at rendezvous site, 6-12 weeks),
  post-weaning (pups independent, >16 weeks),
  failed den (pups lost — cause unknown), failed den (pups lost — infanticide confirmed)

- **habitat_type** (~22 values):
  alpine meadow (above treeline), subalpine meadow (near treeline),
  lodgepole pine forest (dense), lodgepole pine forest (open/burned),
  spruce-fir forest (dense canopy), Douglas fir forest (old growth),
  mixed conifer-aspen (transitional), aspen grove (mature),
  dense willow thicket (riparian), riparian cottonwood gallery,
  sagebrush steppe (open), sagebrush-grassland mosaic,
  open grassland (valley floor), recent burn scar (<5 years, sparse regrowth),
  recovering burn (5-15 years, dense shrub), talus/scree slope,
  rocky outcrop (cliff face), wetland/marsh (sedge meadow),
  old-growth forest (mixed conifer, >200yr), clearcut edge (recent logging),
  road corridor margin (within 100m of road), agricultural field edge (ranch land)

- **overnight_location** (~18 values):
  primary den site, secondary/backup den site, rendezvous site (meadow),
  rendezvous site (creek junction), bedding area (south-facing slope in timber),
  bedding area (ridgetop with wind protection), bedding area (dense willows),
  kill site (elk carcass), kill site (bison carcass),
  river crossing (gravel bar), rocky overhang/cave, exposed ridge (vantage point),
  creek bottom (cottonwood stand), avalanche chute (open), trail junction,
  thermal feature area (Yellowstone specific), abandoned homestead/structure,
  roadside pullout area (human-adjacent)

- **prey_association** (~16 values):
  elk herd (large, 50+ head), elk herd (moderate, 20-50 head), elk herd (small, 5-20 head),
  elk cow-calf group (nursery), lone bull elk (rutting season), elk carcass (old, scavenging),
  bison group (large, 30+), bison group (small, 5-15), lone bison (bull),
  mule deer group (5-15), whitetail deer (solitary or paired),
  moose (solitary adult), moose cow-calf pair,
  pronghorn herd (15-40), bighorn sheep band (8-20),
  no prey observed (area appears empty)

- **social_context** (~18 values) — NEW compound attribute:
  alone (no other collared animals within 2 km),
  with breeding partner only, with breeding partner and 2-3 yearlings,
  with full pack (4-6 members detected), with full pack (7-12 members detected),
  with 1-2 subordinate pack members, with pups at den,
  with pups at rendezvous site, near rival pack (<3 km, same drainage),
  near rival pack (<1 km, boundary confrontation), with non-pack disperser (potential mate),
  trailing elk herd at 200-500m, within 500m of collared animal from different pack,
  with unmarked/uncollared companion(s), separated from pack by >10 km,
  at shared kill site with another pack, near human activity (ranch, road, campground),
  with mixed-species group (e.g., ravens, magpies at carcass)

**Total: 20 tracked attribute options.** When sampling a trial, randomly pick which attribute(s) to track. 50/50 same vs mixed mode per universal design principles.

### 2.1 TRACKABLE_ATTRIBUTES Specification

> **WARNING — Monotonic attributes**: Attributes with direction `monotonic_increase` or
> `monotonic_decrease` (e.g., `collar_battery_pct`, `kill_count_observed`) produce values
> that only move in one direction. This makes RI/PI trivially solvable (first value is
> always min or max). These are marked `interference_quality: "low"` and MUST be
> downweighted in sampling. The `auto_config` function uses `interference_quality` weights
> to ensure high-quality attributes dominate trial generation.

```python
TRACKABLE_ATTRIBUTES = [
    {
        "name": "weight_kg",
        "type": "numeric_continuous",
        "range": (3.0, 900.0),  # red fox 3kg to bison 900kg; per-species range is tighter
        "format_str": "{value} kg",
        "precision": 1,  # e.g., "38.2 kg"
        "direction": "bidirectional",
        "update_events": ["capture_workup", "visual_observation", "seasonal_shift", "health_change"],
        "interference_quality": "high",  # tight within-species ranges, gradual changes, high confusability
        "species_applicable": "all",
        "notes": "Best primary attribute. Within-species range is tight (wolf: 32-65kg). Similar to Dota's gold.",
    },
    {
        "name": "body_condition_score",
        "type": "numeric_integer",
        "range": (1, 9),  # Heinze body condition scoring system
        "format_str": "body condition {value}/9",
        "precision": 0,
        "direction": "bidirectional",
        "update_events": ["capture_workup", "visual_observation", "health_change", "seasonal_shift"],
        "interference_quality": "high",  # 9-point scale, gradual changes, high confusability
        "species_applicable": "all",
        "notes": "Assessed visually or at capture. Changes max +/-1 per 2-4 weeks.",
    },
    {
        "name": "location_zone",
        "type": "categorical",
        "range": "from STUDY_AREAS zone pools (15-30 per area)",
        "format_str": "{value}",
        "precision": None,
        "direction": "bidirectional",
        "update_events": ["relocation_observation", "visual_observation", "capture_workup",
                          "kill_site_investigation", "dispersal_event", "seasonal_shift",
                          "territorial_event"],
        "interference_quality": "high",  # large pool of confusable zone names
        "species_applicable": "all",
        "notes": "Named geographic zones. Adjacent-constrained movement. Pool size 15-30.",
    },
    {
        "name": "daily_movement_km",
        "type": "numeric_continuous",
        "range": (0.0, 50.0),
        "format_str": "{value} km",
        "precision": 1,
        "direction": "bidirectional",
        "update_events": ["relocation_observation", "seasonal_shift"],
        "interference_quality": "high",  # overlapping ranges across species
        "species_applicable": "all",
        "notes": "Varies 3-50km depending on species, season, and activity.",
    },
    {
        "name": "elevation_m",
        "type": "numeric_continuous",
        "range": (180.0, 3500.0),  # Isle Royale 180m to alpine peaks 3500m
        "format_str": "{value} m",
        "precision": 0,
        "direction": "bidirectional",
        "update_events": ["relocation_observation", "visual_observation", "seasonal_shift"],
        "interference_quality": "medium",  # ranges overlap but elevation is partly determined by zone
        "species_applicable": "all",
        "notes": "Tied to zone elevation. Seasonal vertical migration creates good variance.",
    },
    {
        "name": "distance_from_den_km",
        "type": "numeric_continuous",
        "range": (0.0, 80.0),
        "format_str": "{value} km from den",
        "precision": 1,
        "direction": "bidirectional",
        "update_events": ["relocation_observation", "denning_event", "seasonal_shift"],
        "interference_quality": "medium",  # only meaningful during denning season
        "species_applicable": ["gray_wolf", "grizzly_bear", "black_bear", "mountain_lion", "red_fox", "coyote", "canada_lynx"],
        "notes": "Only meaningful for denning species during denning season. Zero outside denning.",
    },
    {
        "name": "territory_size_km2",
        "type": "numeric_continuous",
        "range": (5.0, 4000.0),
        "format_str": "{value} km²",
        "precision": 0,
        "direction": "bidirectional",
        "update_events": ["territorial_event", "pack_interaction", "dispersal_event", "seasonal_shift"],
        "interference_quality": "medium",  # changes slowly, fewer updates per study
        "species_applicable": ["gray_wolf", "mountain_lion", "grizzly_bear", "black_bear",
                               "coyote", "wolverine", "canada_lynx", "red_fox"],
        "notes": "Changes when pack splits/merges or territory expands/contracts. Slow-changing.",
    },
    {
        "name": "collar_battery_pct",
        "type": "numeric_continuous",
        "range": (0.0, 100.0),
        "format_str": "{value}%",
        "precision": 0,
        "direction": "monotonic_decrease",  # only decreases (or jumps to 100 on replacement)
        "update_events": ["collar_event", "capture_workup"],
        "interference_quality": "low",  # MONOTONIC — defeats interference. Downweight in sampling.
        "species_applicable": "all",
        "notes": "WARNING: Monotonic decay makes RI/PI trivially solvable. Only use as filler, not tracked.",
    },
    {
        "name": "distance_traveled_today_km",
        "type": "numeric_continuous",
        "range": (0.0, 50.0),
        "format_str": "{value} km today",
        "precision": 1,
        "direction": "bidirectional",
        "update_events": ["relocation_observation"],
        "interference_quality": "medium",  # high day-to-day variance, but independent per observation
        "species_applicable": "all",
        "notes": "High variance. Each observation is somewhat independent (not cumulative).",
    },
    {
        "name": "pup_count",
        "type": "numeric_integer",
        "range": (0, 9),
        "format_str": "{value} pups",
        "precision": 0,
        "direction": "bidirectional",  # increases at birth, decreases with mortality
        "update_events": ["denning_event", "visual_observation", "mortality_event"],
        "interference_quality": "medium",  # small range (0-9), changes infrequently
        "species_applicable": ["gray_wolf", "coyote", "red_fox", "canada_lynx", "mountain_lion"],
        "notes": "Only relevant for breeding females. Changes in spring (birth) and summer (mortality).",
    },
    {
        "name": "kill_count_observed",
        "type": "numeric_integer",
        "range": (0, 50),
        "format_str": "{value} observed kills",
        "precision": 0,
        "direction": "monotonic_increase",  # cumulative, only increases
        "update_events": ["kill_site_investigation"],
        "interference_quality": "low",  # MONOTONIC — defeats interference. Downweight in sampling.
        "species_applicable": ["gray_wolf", "mountain_lion", "grizzly_bear", "black_bear",
                               "coyote", "wolverine", "canada_lynx"],
        "notes": "WARNING: Monotonic increase makes RI/PI trivially solvable. Only use as filler.",
    },
    {
        "name": "days_since_last_sighting",
        "type": "numeric_integer",
        "range": (0, 120),
        "format_str": "{value} days since last sighting",
        "precision": 0,
        "direction": "bidirectional",  # resets on each observation
        "update_events": ["relocation_observation", "visual_observation", "capture_workup"],
        "interference_quality": "medium",  # resets create variance, but somewhat predictable
        "species_applicable": "all",
        "notes": "Resets to 0 at each observation, then climbs. Creates sawtooth pattern.",
    },
    {
        "name": "scent_post_count",
        "type": "numeric_integer",
        "range": (0, 30),
        "format_str": "{value} scent posts",
        "precision": 0,
        "direction": "bidirectional",
        "update_events": ["territorial_event", "visual_observation"],
        "interference_quality": "medium",  # integer, moderate range
        "species_applicable": ["gray_wolf", "mountain_lion", "coyote", "red_fox", "wolverine"],
        "notes": "Territorial marking frequency per survey. Seasonal variation.",
    },
    {
        "name": "activity_state",
        "type": "categorical",
        "range": "22 values (see Section 2 pool)",
        "format_str": "{value}",
        "precision": None,
        "direction": "bidirectional",
        "update_events": ["relocation_observation", "visual_observation", "kill_site_investigation",
                          "pack_interaction", "dispersal_event", "denning_event", "seasonal_shift"],
        "interference_quality": "high",  # large pool of confusable activity descriptions
        "species_applicable": "all",
        "notes": "22 distinct values. Season-constrained but high confusability.",
    },
    {
        "name": "health_status",
        "type": "categorical",
        "range": "20 values (see Section 2 pool)",
        "format_str": "{value}",
        "precision": None,
        "direction": "bidirectional",
        "update_events": ["capture_workup", "visual_observation", "health_change", "mortality_event"],
        "interference_quality": "high",  # many similar-sounding conditions
        "species_applicable": "all",
        "notes": "20 values with similar-sounding injury/condition descriptions.",
    },
    {
        "name": "pack_membership",
        "type": "categorical",
        "range": "25+ pack names + disperser states (see Section 7)",
        "format_str": "{value}",
        "precision": None,
        "direction": "bidirectional",
        "update_events": ["pack_interaction", "dispersal_event", "mortality_event"],
        "interference_quality": "high",  # many confusable pack names
        "species_applicable": ["gray_wolf", "coyote"],
        "notes": "Pack names are geographically named, creating confusable strings.",
    },
    {
        "name": "reproductive_status",
        "type": "categorical",
        "range": "14 values (see Section 2 pool)",
        "format_str": "{value}",
        "precision": None,
        "direction": "bidirectional",
        "update_events": ["denning_event", "visual_observation", "seasonal_shift"],
        "interference_quality": "medium",  # constrained progression, fewer valid transitions
        "species_applicable": "all",
        "notes": "Season-locked progression. Some transitions are one-way within a season.",
    },
    {
        "name": "habitat_type",
        "type": "categorical",
        "range": "22 values (see Section 2 pool)",
        "format_str": "{value}",
        "precision": None,
        "direction": "bidirectional",
        "update_events": ["relocation_observation", "visual_observation", "seasonal_shift"],
        "interference_quality": "high",  # many similar-sounding habitat names
        "species_applicable": "all",
        "notes": "22 values. Dense forest types are easily confused.",
    },
    {
        "name": "overnight_location",
        "type": "categorical",
        "range": "18 values (see Section 2 pool)",
        "format_str": "{value}",
        "precision": None,
        "direction": "bidirectional",
        "update_events": ["relocation_observation", "visual_observation"],
        "interference_quality": "high",  # many confusable location types
        "species_applicable": "all",
        "notes": "18 values. Bedding/resting site descriptions overlap heavily.",
    },
    {
        "name": "prey_association",
        "type": "categorical",
        "range": "16 values (see Section 2 pool)",
        "format_str": "{value}",
        "precision": None,
        "direction": "bidirectional",
        "update_events": ["kill_site_investigation", "visual_observation", "relocation_observation"],
        "interference_quality": "medium",  # moderate pool, some values very distinct
        "species_applicable": ["gray_wolf", "mountain_lion", "grizzly_bear", "black_bear",
                               "coyote", "wolverine", "canada_lynx"],
        "notes": "Only for predator species. Prey group sizes create some confusability.",
    },
    {
        "name": "social_context",
        "type": "categorical",
        "range": "18 values (see Section 2 pool)",
        "format_str": "{value}",
        "precision": None,
        "direction": "bidirectional",
        "update_events": ["relocation_observation", "visual_observation", "pack_interaction",
                          "dispersal_event", "denning_event"],
        "interference_quality": "high",  # many similar pack-size/companion descriptions
        "species_applicable": "all",
        "notes": "18 values. Pack-size variants create high confusability.",
    },
]

# Interference quality weights for sampling in auto_config
INTERFERENCE_QUALITY_WEIGHTS = {
    "high": 3,    # 3x more likely to be sampled
    "medium": 1,  # baseline
    "low": 0.1,   # 10x less likely — monotonic attributes nearly excluded
}
```

### Entity heterogeneity: same-species vs mixed-species

**Same-species mode (~60% of trials):** All entities are the same species (e.g., 5-12 wolves). Value ranges overlap heavily → maximum interference. This is the hard condition.

Species options for same-species mode (15 species):

**Predators:**
- **Gray wolf** — best candidate. Pack structure = natural entity grouping. 5-12 animals per pack. Weight M:40-65kg, F:32-50kg. Extensively studied (Yellowstone data).
- **Mountain lion / cougar** — solitary, territorial. Weight M:55-100kg, F:30-65kg. Hard to observe → sparse GPS data. Large home ranges (150-800 km²).
- **Grizzly bear** — solitary but overlapping ranges. Weight M:180-360kg, F:130-200kg. Hibernation Nov-Mar creates seasonal data gaps.
- **Black bear** — smaller, more abundant. Weight M:60-200kg, F:40-80kg. Arboreal, den in hollow trees. Less aggressive than grizzly.
- **Coyote** — pack or pair structure. Weight 8-20kg. Urban-wildland interface. Much smaller than wolf (distinct weight range).
- **Red fox** — solitary/pair. Weight 3-7kg. Wide habitat tolerance. Smallest canid in pool.
- **Wolverine** — solitary, vast ranges (200-1500 km²). Weight 8-18kg. Alpine/subalpine specialist. Very hard to observe.
- **Canada lynx** — solitary. Weight 8-14kg. Tied to snowshoe hare cycle. Boreal forest specialist.

**Ungulates / Herbivores:**
- **Elk / wapiti** — herd animal. Weight M:300-450kg, F:225-300kg. Seasonal migration. Most abundant large herbivore in Yellowstone.
- **Mule deer** — smaller herds or solitary. Weight M:55-120kg, F:45-75kg. Migratory in mountains.
- **Whitetail deer** — forest edge. Weight M:60-130kg, F:40-75kg. Less migratory than mule deer.
- **Moose** — solitary or cow-calf. Weight M:380-600kg, F:270-400kg. Largest cervid. Wetland/riparian specialist.
- **Pronghorn** — herd, highly migratory. Weight M:40-65kg, F:35-50kg (overlaps wolf range!). Fastest land animal in N. America.
- **Bighorn sheep** — band structure (8-30). Weight M:60-125kg, F:35-70kg. Alpine/cliff specialist. Seasonal altitudinal migration.
- **Bison** — large herds. Weight M:450-900kg, F:320-545kg. Yellowstone icon. Seasonal range shifts.

**Mixed-species mode (~40% of trials):** Mix of 2-3 species from the same ecosystem. Some attributes still overlap (elevation, movement, body condition, location), others don't (weight). Creates a natural difficulty gradient.

Typical mixes:
- 3 wolves + 3 elk (predator-prey, same valley)
- 2 wolves + 2 grizzlies + 2 elk (Yellowstone triad)
- 4 mountain lions + 3 mule deer (predator-prey, mountain terrain)
- 3 pronghorn + 3 wolves (similar weight ranges! hard even in mixed mode)
- 2 grizzlies + 2 black bears + 2 elk (co-occurring large mammals)
- 3 lynx + 3 wolverines (elusive carnivore study, boreal forest)
- 4 bighorn sheep + 3 mountain lions (alpine predator-prey)
- 2 wolves + 2 coyotes + 2 elk (mesopredator-apex-prey dynamics)
- 3 moose + 3 wolves (Isle Royale classic)
- 2 bison + 2 elk + 2 wolves (Yellowstone full community)

### How do non-tracked attributes fit in?

Same as Dota 2: non-tracked attributes appear as **filler/context**. They make the narrative realistic but aren't queried. Example:

> "March 14: Wolf F-2847 was relocated via aerial telemetry to the **Copper Creek drainage**, 12 km northeast of her last known position. She appeared to be **traveling alone** — no other pack members detected within 2 km. Body condition appeared fair, estimated **4 out of 9**. Snow cover in the drainage was patchy at this elevation."

Here location_zone (bold) is the tracked attribute. Distance, social context, body condition, and snow cover are filler.

## 3. State Machine Design

### 3.1 Study Initialization

```
Input: num_keys animals, seed
Output: initial state for all animals + study metadata

Steps:
1. Pick species mode: same-species (60%) or mixed-species (40%)
2. If same-species: pick one species from pool, generate num_keys animals
3. If mixed-species: pick 2-3 species, distribute num_keys across them
4. Assign each animal:
   - collar_id: species letter + gender + 4-digit number (e.g., WF-2847, WM-1203, EM-0456)
   - name (optional common name for known individuals): "Scarface", "Old Blue", etc.
   - species, sex, age_years
   - pack_or_group membership (wolves/elk) or solitary (bears/lions)
   - initial state: weight, location_zone, body_condition, activity_state, etc.
5. Pick study_area from geographic pools (Yellowstone, Greater Yellowstone Ecosystem,
   Northern Cascades, Northern Rockies generic, Great Lakes generic)
6. Pick season_start (determines initial conditions and event weights)
7. Generate location_zone pool from study_area (15-30 named zones)
```

### 3.2 Event Types

The study is a sequence of observations/events. Each event updates 1-3 animals' states.

```
EVENT TYPES:
├── relocation_observation    — animal detected at new location (GPS collar ping / aerial survey)
├── capture_workup            — animal captured, weighed, measured, collar checked (rare, high-info)
├── visual_observation        — field sighting, body condition noted, activity recorded
├── kill_site_investigation   — predation event found, prey identified, animal states updated
├── pack_interaction          — inter-pack encounter, aggression, displacement, or merger
├── dispersal_event           — animal leaves pack/group, begins solo movement
├── denning_event             — animal enters/exits den, pups born, pups emerge
├── mortality_event           — animal dies (predation, starvation, human-caused, disease)
├── seasonal_shift            — behavior changes with season (migration start, denning, rut)
├── collar_event              — collar battery check, replacement, signal loss/recovery
├── territorial_event         — boundary patrol, scent marking survey, territory expansion/contraction
└── health_change             — injury detected, recovery noted, parasite load change
```

### 3.2.1 Event Dict Schema

Every event handler returns a dict with this exact structure:

```python
@dataclass
class EventDict:
    """Returned by every event handler. This is the contract between event generation and narrative rendering."""
    event_type: str              # One of the 12 event types from Section 3.2
    timestamp: str               # ISO date string, e.g. "2024-03-14"
    study_day: int               # Day number within the study (0-indexed)
    primary_animal: str          # Collar ID of the main animal, e.g. "WF-2847"
    secondary_animals: list      # List of collar IDs of other involved animals (may be empty)
                                 # e.g. ["WM-1203", "WF-3041"] for pack_interaction
    attribute_changes: dict      # Dict of {attribute_name: new_value} for primary_animal
                                 # e.g. {"weight_kg": 38.2, "location_zone": "Copper Creek drainage"}
                                 # Only includes attributes that CHANGED in this event.
    secondary_attribute_changes: dict  # Dict of {collar_id: {attr: new_value}} for secondary animals
                                       # e.g. {"WM-1203": {"location_zone": "Lamar Valley"}}
    narrative_data: dict         # Template variables for narrative rendering. Includes:
                                 # - "date": formatted date string
                                 # - "animal_id": collar ID (or common name if set)
                                 # - "pronoun", "pronoun_pos", "pronoun_cap": gendered pronouns
                                 # - "zone": location zone name
                                 # - all event-type-specific variables (see template sections)
                                 # e.g. {"date": "March 14", "animal_id": "WF-2847",
                                 #        "zone": "Copper Creek drainage", "distance_km": 12,
                                 #        "direction": "northeast", "pronoun": "she", ...}
    filler_data: dict            # Context/filler variables for non-tracked attributes
                                 # e.g. {"snow_depth_cm": 45, "elk_count": 120,
                                 #        "weather_condition": "overcast skies"}
    tracked_value_mention: dict  # Exactly which tracked attribute value appears in this event's narrative
                                 # {"attribute": "weight_kg", "value": 38.2, "animal": "WF-2847"}
                                 # This is the value that counts toward num_updates.
                                 # None if this event doesn't mention a tracked value.

# Example return from a relocation_observation handler:
# {
#     "event_type": "relocation_observation",
#     "timestamp": "2024-03-14",
#     "study_day": 73,
#     "primary_animal": "WF-2847",
#     "secondary_animals": [],
#     "attribute_changes": {"location_zone": "Copper Creek drainage", "elevation_m": 2100},
#     "secondary_attribute_changes": {},
#     "narrative_data": {
#         "date": "March 14", "animal_id": "WF-2847", "zone": "Copper Creek drainage",
#         "distance_km": 12, "direction": "northeast", "previous_zone": "Slough Creek",
#         "elevation_m": 2100, "pronoun": "she", "pronoun_pos": "her", "pronoun_cap": "She",
#         "context_detail": "Snow cover in the drainage was patchy at this elevation."
#     },
#     "filler_data": {"snow_depth_cm": 45, "social_note": "no other pack members within 2 km"},
#     "tracked_value_mention": {"attribute": "location_zone", "value": "Copper Creek drainage",
#                                "animal": "WF-2847"}
# }
```

### 3.3 Event Generation (the state machine)

```python
# ALL 12 event types present in EVERY season. Weight=0 means impossible for that season.
# This ensures consistent dict keys across all seasons for programmatic access.

SEASONAL_EVENT_WEIGHTS = {
    "winter": {  # December-February
        "relocation_observation": 30,    # collar pings continue; fewer field visits
        "visual_observation": 10,         # hard to access in winter, deep snow
        "capture_workup": 2,              # rarely trap in winter — too stressful for animals
        "kill_site_investigation": 20,    # wolves hunt elk heavily in winter
        "pack_interaction": 5,            # packs encounter each other at shared kill sites
        "dispersal_event": 0,             # dispersal does not occur in winter (energetically prohibitive)
        "denning_event": 0,               # denning is a spring event (April-May for wolves)
        "mortality_event": 8,             # starvation, exposure, winter-kill
        "seasonal_shift": 5,              # mid-winter movement pattern changes
        "collar_event": 5,                # battery depletes faster in cold
        "territorial_event": 10,          # winter territorial patrols along boundaries
        "health_change": 5,               # winter stress, mange progression, injuries
    },
    "spring": {  # March-May
        "relocation_observation": 20,
        "visual_observation": 15,          # more field access as snow melts
        "capture_workup": 8,               # spring trapping season begins
        "kill_site_investigation": 15,     # hunting continues; elk calving = vulnerable prey
        "pack_interaction": 5,
        "dispersal_event": 5,              # yearlings start dispersing in spring
        "denning_event": 15,               # pups born April-May (wolves); bears emerge from dens
        "mortality_event": 5,              # post-winter mortality, weakened animals succumb
        "seasonal_shift": 4,               # spring green-up, migration onset
        "collar_event": 3,
        "territorial_event": 0,            # territory patrols minimal during denning focus
        "health_change": 5,                # post-winter recovery, injury healing
    },
    "summer": {  # June-August
        "relocation_observation": 25,
        "visual_observation": 20,          # best field season — long days, accessible terrain
        "capture_workup": 10,              # summer trapping operations
        "kill_site_investigation": 10,     # prey more dispersed, lower kill density
        "pack_interaction": 5,
        "dispersal_event": 10,             # peak dispersal season for yearlings/young adults
        "denning_event": 8,                # pups at rendezvous sites, den emergence
        "mortality_event": 0,              # summer mortality very rare for adults
        "seasonal_shift": 0,               # no major seasonal transitions mid-summer
        "collar_event": 4,
        "territorial_event": 5,            # some boundary maintenance
        "health_change": 3,                # low — summer is easiest season
    },
    "fall": {  # September-November
        "relocation_observation": 25,
        "visual_observation": 15,
        "capture_workup": 5,
        "kill_site_investigation": 15,     # pre-winter hunting increases; elk rut = easy prey
        "pack_interaction": 10,            # fall territorial disputes peak
        "dispersal_event": 8,              # second dispersal window (young adults)
        "denning_event": 0,                # no denning in fall (bears enter dens but that's seasonal_shift)
        "mortality_event": 4,              # hunting season outside parks; inter-pack conflict
        "seasonal_shift": 8,               # migration, pre-denning movements, rut behavior
        "collar_event": 0,                 # collar maintenance done in summer; no fall checks
        "territorial_event": 10,           # fall boundary defense intensifies before winter
        "health_change": 0,                # health changes absorbed into seasonal_shift for fall
    },
}

def generate_next_event(state, study_day, season, rng):
    """Generate a plausible next observation based on current state and season."""

    weights = dict(SEASONAL_EVENT_WEIGHTS[season])  # copy to allow filtering

    # Filter impossible events for the current animal pool:
    # - Can't have denning for males, non-breeding females, or non-denning species
    # - Can't disperse if already a disperser/solitary
    # - Dead animals generate no events
    # - Can't do capture_workup if animal hasn't been located recently
    # - Mortality removes animal from future events
    # After filtering, re-normalize weights (remove 0-weight keys for sampling)

    filtered = {k: v for k, v in weights.items() if v > 0}
    event_type = weighted_random_choice(filtered, rng)
    event = create_event(event_type, state, study_day, season, rng)
    return event
```

### 3.4 Event Handlers (Constraints)

Each event handler enforces ecological rules:

```
RELOCATION_OBSERVATION:
  - Pick a living animal (weighted by collar reliability — some collars ping more often)
  - Advance study_day by 1-14 days (observation frequency varies)
  - Update location_zone (movement constrained by distance from last location)
  - Update elevation based on zone
  - Update distance_from_den if denning season
  - Constraints:
    - Movement per day: wolves 15-30km, elk 5-20km, bears 3-15km, lions 5-25km
    - Total displacement since last observation = daily_rate × days_elapsed (roughly)
    - Can't jump to location 200km away in 3 days for a wolf
    - Zone transitions must be geographically adjacent or connected
    - Season affects which zones are accessible (high alpine closed in winter)

CAPTURE_WORKUP:
  - Pick a living animal (weighted toward animals not recently captured)
  - This is the HIGH-INFORMATION event — many attributes updated at once
  - Record: weight_kg, body_condition_score, collar_battery_pct, health_status
  - Take measurements: body length, chest girth
  - Possibly replace collar
  - Constraints:
    - Weight change since last capture: max ±15% per month for adults
    - Body condition changes max ±2 points per month
    - Collar battery decreases ~2-5% per month (GPS collars)
    - Can't capture dead animals
    - Capture frequency: typically 1-2 times per year per animal

VISUAL_OBSERVATION:
  - Pick a living animal (weighted by visibility — elk/pronghorn easier than wolves/lions)
  - Update: location_zone, activity_state, estimated body_condition (less precise than capture)
  - Note social context: alone, with pack, with pups, near other tagged animals
  - Constraints:
    - Visual estimates of body condition are ±1 point compared to capture
    - Can't see weight directly (only estimate from body condition)
    - Some species are nocturnal/crepuscular — observation timing matters

KILL_SITE_INVESTIGATION:
  - Pick 1-3 predator animals (wolves/lions/bears that were near the kill site)
  - Update: location_zone (at kill site), activity_state → "feeding"
  - Optionally update prey animal status if prey is also tagged
  - Constraints:
    - Only predator species initiate kills
    - Kill frequency: wolves average 1 ungulate per 10-14 days per pack
    - Bears scavenge more than hunt (especially in spring on winter-killed elk)
    - Kill sites cluster near prey concentrations

PACK_INTERACTION:
  - Pick 2-4 animals from different packs/groups
  - Update: location_zone (both packs at boundary area), activity_state
  - Possible outcomes: avoidance, howling exchange, aggressive chase, mortality
  - Constraints:
    - Only social species (wolves, elk herds) have meaningful group interactions
    - Inter-pack aggression is the #1 cause of natural wolf mortality
    - Interactions happen at territory boundaries

DISPERSAL_EVENT:
  - Pick one yearling or young adult (typically 1-3 years old for wolves)
  - Update: pack_membership → "disperser", activity_state → "migrating"
  - Subsequent events show long-distance movement
  - Constraints:
    - Dispersal distance: wolves average 60-100km, up to 1000+km
    - Timing: peak in spring (yearlings) and fall (young adults)
    - Breeding pair never disperses (pack dissolution is different)
    - Once dispersed, animal may join existing pack or find mate to form new pack

DENNING_EVENT:
  - Pick a breeding female
  - Update: location_zone → den site, activity_state → "denning", pup_count
  - Constraints:
    - Only breeding females den
    - Den entry: late March to mid-April (wolves)
    - Pup birth: mid-April (wolves), litter size 4-6 average
    - Den emergence: 2-3 weeks after birth
    - Rendezvous site move: 6-8 weeks after birth
    - Pup mortality: 40-60% in first year

MORTALITY_EVENT:
  - Pick a living animal
  - Update: status → "dead", record cause and location
  - Constraints:
    - Causes: inter-pack aggression (wolves), starvation (winter), human (vehicle/hunting),
      disease, predation (for prey species)
    - Seasonal mortality: highest in late winter (February-March)
    - Age-dependent: pups have highest mortality, prime adults lowest
    - Dead animals stop generating events (but may appear in investigation events)

SEASONAL_SHIFT:
  - Affects multiple animals simultaneously
  - Update: activity_state, movement patterns, location preferences
  - Constraints:
    - Winter → spring: movement increases, weight recovery begins
    - Spring → summer: denning females emerge, pups at rendezvous sites
    - Summer → fall: increased movement, territorial activity, pre-breeding behavior
    - Fall → winter: elk migration begins, wolves follow prey, bears enter hibernation
    - Bears: hibernate November-March (no observations possible)

COLLAR_EVENT:
  - Pick an animal
  - Update: collar_battery_pct (decrease), possibly signal_status
  - Constraints:
    - Battery depletes faster in cold weather
    - GPS fix rate: 4-12 fixes/day typical
    - Battery life: 1-3 years depending on fix rate
    - When battery dies, animal becomes untrackable (potential narrative gap)

TERRITORIAL_EVENT:
  - Pick 1-2 animals (usually breeding pair)
  - Update: location_zone (at territory boundary), territory_size_km2
  - Describe boundary patrol, scent marking, or howling
  - Constraints:
    - Territory size changes gradually (expansion/contraction over months)
    - Boundary patrols happen every 2-3 weeks
    - Scent posts refreshed every 2-4 weeks
    - Territory overlap between packs: usually < 5%

HEALTH_CHANGE:
  - Pick an animal
  - Update: health_status, possibly body_condition_score and weight_kg
  - Constraints:
    - Mange: slow progression over months, weight loss, hair loss
    - Injury: from inter-pack fight, trap injury, or vehicle strike
    - Recovery: takes weeks to months depending on severity
    - Sarcoptic mange: 50-80% case fatality in wolves without treatment
```

### 3.5 Attribute Trajectory Constraints (Critical for Realism)

Weight trajectories must be ecologically plausible:

```
GRAY WOLF weight trajectory (adult male, 50kg baseline):
  Jan:  42-50 kg (post-winter low, losing weight)
  Feb:  40-48 kg (lowest point, prey hardest to catch)
  Mar:  42-50 kg (recovery begins, elk calving soon)
  Apr:  44-52 kg (spring recovery)
  May:  48-55 kg (abundant prey, elk calves)
  Jun:  50-58 kg (peak prey availability)
  Jul:  52-60 kg (summer peak)
  Aug:  50-58 kg (pups weaning, competition for food)
  Sep:  48-56 kg (fall maintenance)
  Oct:  50-58 kg (pre-winter buildup, elk rut = easier prey)
  Nov:  48-55 kg (winter onset)
  Dec:  45-52 kg (winter decline begins)

Key constraint: Weight changes max ±1-2 kg per week for adults.
Pups: birth weight ~500g, reach ~15-20kg by 6 months, ~30-40kg by 12 months.

GRAY WOLF body condition score trajectory:
  Late winter (Feb-Mar): 2-4 (lean)
  Spring (Apr-May): 3-5 (recovering)
  Summer (Jun-Aug): 5-7 (good to excellent)
  Fall (Sep-Nov): 4-7 (variable)
  Early winter (Dec-Jan): 3-5 (declining)

Key constraint: BCS changes max ±1 point per 2-4 weeks.

COLLAR BATTERY trajectory:
  Start: 100%
  Depletion rate: 2-5% per month (GPS mode)
  Cold weather: +50% depletion (winter months)
  Replace at: 15-20% remaining
```

## 3.6 Seasonal Coherence Engine (Ecological Correlation Rules)

This is the wildlife equivalent of the ICU physiological correlation engine. Every attribute change must be ecologically coherent with the season, species, and individual trajectory. The state machine does NOT independently drift attributes — it moves the animal along a **seasonal trajectory curve** and enforces cross-attribute correlations.

### 3.6.1 Seasonal Trajectory Curves by Species

Each species has a master seasonal curve. The state machine reads values from this curve (with noise) rather than randomly generating them. The curve encodes the correlations implicitly.

```python
# GRAY WOLF — Annual trajectory (breeding adult female, Yellowstone)
WOLF_FEMALE_ANNUAL = {
    "jan": {
        "weight_kg": (38, 46), "bcs": (3, 4),
        "activity": ["traveling through drainage", "hunting (stalking)", "feeding at fresh kill site", "resting in open meadow"],
        "habitat": ["lodgepole pine forest (dense)", "sagebrush-grassland mosaic", "riparian cottonwood gallery"],
        "elevation_m": (1700, 2200),  # lower elevations in winter, following elk
        "movement_km_day": (15, 30),  # high movement, hunting
        "prey": ["elk herd (large, 50+ head)", "elk herd (moderate, 20-50 head)", "elk carcass (old, scavenging)"],
        "reproductive": "non-breeding subordinate adult",  # or "courting (pair bonding observed)" if alpha
        "social": ["with full pack (4-6 members detected)", "with full pack (7-12 members detected)"],
    },
    "feb": {
        "weight_kg": (36, 44), "bcs": (2, 4),  # lowest weight
        "activity": ["hunting (active chase)", "traveling on ridgeline", "feeding at fresh kill site", "scent-marking territory boundary"],
        "habitat": ["sagebrush-grassland mosaic", "open grassland (valley floor)", "riparian cottonwood gallery"],
        "elevation_m": (1600, 2100),
        "movement_km_day": (15, 35),
        "prey": ["elk herd (large, 50+ head)", "elk cow-calf group"],
        "reproductive": "breeding confirmed (observed tie)",  # breeding season Feb
        "social": ["with breeding partner only", "with full pack (4-6 members detected)"],
    },
    "mar": {
        "weight_kg": (37, 46), "bcs": (2, 4),
        "activity": ["traveling through drainage", "hunting (stalking)", "resting at den site", "scent-marking territory boundary"],
        "habitat": ["lodgepole pine forest (dense)", "spruce-fir forest (dense canopy)", "dense willow thicket (riparian)"],
        "elevation_m": (1700, 2300),
        "movement_km_day": (10, 25),  # decreasing if approaching denning
        "prey": ["elk herd (moderate, 20-50 head)", "elk cow-calf group", "mule deer group (5-15)"],
        "reproductive": "pregnant (early — no visible change)",
        "social": ["with breeding partner and 2-3 yearlings", "with full pack (4-6 members detected)"],
    },
    "apr": {
        "weight_kg": (38, 48), "bcs": (3, 5),  # recovery starting, prey more available
        "activity": ["resting at den site", "denning (pre-whelp, restricted movement)", "hunting (ambush)"],
        "habitat": ["spruce-fir forest (dense canopy)", "lodgepole pine forest (dense)"],  # den sites typically in dense forest
        "elevation_m": (1800, 2400),
        "movement_km_day": (2, 10),  # restricted movement near den
        "prey": ["elk cow-calf group", "elk herd (small, 5-20 head)"],
        "reproductive": "denning (pre-whelp, restricted movement)",  # or "nursing (pups in den, <3 weeks)" late April
        "social": ["with breeding partner only", "alone (no other collared animals within 2 km)"],
    },
    "may": {
        "weight_kg": (36, 45), "bcs": (3, 5),  # post-whelp dip for females
        "activity": ["resting at den site", "nursing (pups at den entrance)", "hunting (stalking)"],
        "habitat": ["spruce-fir forest (dense canopy)", "lodgepole pine forest (dense)", "subalpine meadow (near treeline)"],
        "elevation_m": (1900, 2500),
        "movement_km_day": (2, 8),  # very restricted, near den
        "prey": ["elk cow-calf group", "mule deer group (5-15)", "elk herd (small, 5-20 head)"],
        "reproductive": "nursing (pups in den, <3 weeks)",
        "social": ["with breeding partner only", "with pups at den"],
    },
    "jun": {
        "weight_kg": (40, 52), "bcs": (4, 6),  # recovery, abundant prey
        "activity": ["at rendezvous site with pups", "hunting (active chase)", "feeding at fresh kill site", "playing/socializing with pack"],
        "habitat": ["subalpine meadow (near treeline)", "alpine meadow (above treeline)", "open grassland (valley floor)"],
        "elevation_m": (2000, 2800),  # moving upslope
        "movement_km_day": (5, 20),  # increasing from den
        "prey": ["elk cow-calf group", "elk herd (moderate, 20-50 head)", "mule deer group (5-15)"],
        "reproductive": "nursing (pups at den entrance, 3-6 weeks)",
        "social": ["with pups at rendezvous site", "with full pack (4-6 members detected)"],
    },
    "jul": {
        "weight_kg": (44, 56), "bcs": (5, 7),  # peak prey availability
        "activity": ["at rendezvous site with pups", "hunting (active chase)", "playing/socializing with pack", "traveling through drainage"],
        "habitat": ["subalpine meadow (near treeline)", "alpine meadow (above treeline)", "mixed conifer-aspen (transitional)"],
        "elevation_m": (2100, 3000),
        "movement_km_day": (8, 25),
        "prey": ["elk cow-calf group", "elk herd (moderate, 20-50 head)", "mule deer group (5-15)", "moose cow-calf pair"],
        "reproductive": "nursing (pups at rendezvous site, 6-12 weeks)",
        "social": ["with pups at rendezvous site", "with full pack (7-12 members detected)"],
    },
    "aug": {
        "weight_kg": (44, 56), "bcs": (5, 7),  # summer peak
        "activity": ["traveling on ridgeline", "hunting (stalking)", "at rendezvous site with pups", "patrolling territory perimeter"],
        "habitat": ["alpine meadow (above treeline)", "subalpine meadow (near treeline)", "spruce-fir forest (dense canopy)"],
        "elevation_m": (2000, 3000),
        "movement_km_day": (10, 30),  # pups more mobile, pack ranges wider
        "prey": ["elk herd (moderate, 20-50 head)", "mule deer group (5-15)", "bighorn sheep band (8-20)"],
        "reproductive": "post-weaning (pups independent, >16 weeks)",
        "social": ["with full pack (7-12 members detected)", "with pups at rendezvous site"],
    },
    "sep": {
        "weight_kg": (42, 54), "bcs": (4, 7),
        "activity": ["traveling on ridgeline", "hunting (active chase)", "scent-marking territory boundary", "patrolling territory perimeter"],
        "habitat": ["mixed conifer-aspen (transitional)", "aspen grove (mature)", "sagebrush-grassland mosaic"],
        "elevation_m": (1900, 2800),
        "movement_km_day": (15, 35),  # fall territory defense
        "prey": ["elk herd (large, 50+ head)", "lone bull elk (rutting season)", "mule deer group (5-15)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with full pack (7-12 members detected)", "near rival pack (<3 km, same drainage)"],
    },
    "oct": {
        "weight_kg": (44, 56), "bcs": (4, 7),  # pre-winter buildup, elk rut = easy prey
        "activity": ["hunting (active chase)", "feeding at fresh kill site", "traveling through drainage", "scent-marking territory boundary"],
        "habitat": ["sagebrush-grassland mosaic", "open grassland (valley floor)", "riparian cottonwood gallery"],
        "elevation_m": (1800, 2500),  # moving downslope following elk
        "movement_km_day": (15, 30),
        "prey": ["lone bull elk (rutting season)", "elk herd (large, 50+ head)", "elk cow-calf group"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with full pack (7-12 members detected)"],
    },
    "nov": {
        "weight_kg": (42, 52), "bcs": (4, 6),  # winter decline starting
        "activity": ["hunting (stalking)", "traveling through drainage", "resting in open meadow", "feeding at fresh kill site"],
        "habitat": ["sagebrush-grassland mosaic", "lodgepole pine forest (open/burned)", "open grassland (valley floor)"],
        "elevation_m": (1700, 2300),  # lower
        "movement_km_day": (15, 30),
        "prey": ["elk herd (large, 50+ head)", "elk herd (moderate, 20-50 head)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with full pack (4-6 members detected)", "with full pack (7-12 members detected)"],
    },
    "dec": {
        "weight_kg": (40, 50), "bcs": (3, 5),  # declining
        "activity": ["hunting (stalking)", "traveling on ridgeline", "resting in open meadow", "feeding at old carcass (scavenging)"],
        "habitat": ["sagebrush-grassland mosaic", "open grassland (valley floor)", "lodgepole pine forest (dense)"],
        "elevation_m": (1600, 2200),
        "movement_km_day": (15, 30),
        "prey": ["elk herd (large, 50+ head)", "elk herd (moderate, 20-50 head)", "elk carcass (old, scavenging)"],
        "reproductive": "non-breeding subordinate adult",  # or "courting" late Dec for alpha
        "social": ["with full pack (4-6 members detected)", "with full pack (7-12 members detected)"],
    },
}
```

**Additional species trajectory curves** (same structure as wolf female above, derived from SPECIES_PARAMETERS file):

```python
# GRAY WOLF — Annual trajectory (adult male, Northern Rockies)
# Anchor values from SPECIES_PARAMETERS: M 32-66 kg, avg 50 kg. Movement 16-32 km/day.
# Monthly values DERIVED by interpolation from seasonal anchors. Marked with # DERIVED.
WOLF_MALE_ANNUAL = {
    "jan": {
        "weight_kg": (44, 54), "bcs": (3, 5),  # DERIVED: post-peak decline, above female
        "activity": ["traveling through drainage", "hunting (active chase)", "feeding at fresh kill site", "scent-marking territory boundary"],
        "habitat": ["lodgepole pine forest (dense)", "sagebrush-grassland mosaic", "riparian cottonwood gallery"],
        "elevation_m": (1700, 2200),  # following elk to low elevation
        "movement_km_day": (18, 35),  # DERIVED: higher than female, territorial patrols
        "prey": ["elk herd (large, 50+ head)", "elk herd (moderate, 20-50 head)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with full pack (4-6 members detected)", "with full pack (7-12 members detected)"],
    },
    "feb": {
        "weight_kg": (42, 52), "bcs": (3, 4),  # DERIVED: lowest point, energy deficit
        "activity": ["hunting (active chase)", "scent-marking territory boundary", "traveling on ridgeline", "howling (stationary)"],
        "habitat": ["sagebrush-grassland mosaic", "open grassland (valley floor)", "riparian cottonwood gallery"],
        "elevation_m": (1600, 2100),
        "movement_km_day": (18, 35),  # DERIVED: high movement, breeding season competition
        "prey": ["elk herd (large, 50+ head)", "elk cow-calf group"],
        "reproductive": "breeding confirmed (observed tie)",  # breeding season
        "social": ["with breeding partner only", "with full pack (4-6 members detected)"],
    },
    "mar": {
        "weight_kg": (43, 54), "bcs": (3, 5),  # DERIVED: recovery starting
        "activity": ["hunting (stalking)", "scent-marking territory boundary", "patrolling territory perimeter", "traveling through drainage"],
        "habitat": ["lodgepole pine forest (dense)", "spruce-fir forest (dense canopy)", "sagebrush-grassland mosaic"],
        "elevation_m": (1700, 2300),
        "movement_km_day": (16, 32),  # DERIVED: broad territory patrols while female dens
        "prey": ["elk herd (moderate, 20-50 head)", "mule deer group (5-15)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with full pack (4-6 members detected)", "with 1-2 subordinate pack members"],
    },
    "apr": {
        "weight_kg": (46, 56), "bcs": (4, 5),  # DERIVED: recovery, providing for denning female
        "activity": ["hunting (active chase)", "traveling through drainage", "feeding at fresh kill site", "patrolling territory perimeter"],
        "habitat": ["spruce-fir forest (dense canopy)", "lodgepole pine forest (dense)", "subalpine meadow (near treeline)"],
        "elevation_m": (1800, 2400),
        "movement_km_day": (15, 30),  # DERIVED: food provisioning to den
        "prey": ["elk cow-calf group", "elk herd (small, 5-20 head)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with 1-2 subordinate pack members", "with full pack (4-6 members detected)"],
    },
    "may": {
        "weight_kg": (48, 58), "bcs": (4, 6),  # DERIVED: abundant prey, elk calves vulnerable
        "activity": ["hunting (active chase)", "feeding at fresh kill site", "traveling through drainage"],
        "habitat": ["subalpine meadow (near treeline)", "lodgepole pine forest (dense)", "mixed conifer-aspen (transitional)"],
        "elevation_m": (1900, 2600),
        "movement_km_day": (15, 30),  # DERIVED
        "prey": ["elk cow-calf group", "elk herd (moderate, 20-50 head)", "mule deer group (5-15)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with full pack (4-6 members detected)"],
    },
    "jun": {
        "weight_kg": (50, 62), "bcs": (5, 7),  # DERIVED: peak prey availability
        "activity": ["hunting (active chase)", "feeding at fresh kill site", "playing/socializing with pack", "at rendezvous site with pups"],
        "habitat": ["subalpine meadow (near treeline)", "alpine meadow (above treeline)", "open grassland (valley floor)"],
        "elevation_m": (2000, 2800),
        "movement_km_day": (10, 25),  # DERIVED: some time at rendezvous
        "prey": ["elk cow-calf group", "elk herd (moderate, 20-50 head)", "mule deer group (5-15)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with pups at rendezvous site", "with full pack (7-12 members detected)"],
    },
    "jul": {
        "weight_kg": (52, 64), "bcs": (5, 7),  # DERIVED: summer peak approaching
        "activity": ["hunting (active chase)", "playing/socializing with pack", "traveling through drainage"],
        "habitat": ["subalpine meadow (near treeline)", "alpine meadow (above treeline)", "mixed conifer-aspen (transitional)"],
        "elevation_m": (2100, 3000),
        "movement_km_day": (12, 28),  # DERIVED
        "prey": ["elk cow-calf group", "elk herd (moderate, 20-50 head)", "mule deer group (5-15)", "moose cow-calf pair"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with full pack (7-12 members detected)", "with pups at rendezvous site"],
    },
    "aug": {
        "weight_kg": (54, 66), "bcs": (6, 8),  # DERIVED: near peak, upper end of species range
        "activity": ["traveling on ridgeline", "hunting (stalking)", "patrolling territory perimeter"],
        "habitat": ["alpine meadow (above treeline)", "subalpine meadow (near treeline)", "spruce-fir forest (dense canopy)"],
        "elevation_m": (2000, 3000),
        "movement_km_day": (15, 32),  # DERIVED: wider ranging
        "prey": ["elk herd (moderate, 20-50 head)", "mule deer group (5-15)", "bighorn sheep band (8-20)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with full pack (7-12 members detected)"],
    },
    "sep": {
        "weight_kg": (52, 64), "bcs": (5, 7),  # DERIVED: slight decline from peak
        "activity": ["scent-marking territory boundary", "patrolling territory perimeter", "hunting (active chase)", "traveling on ridgeline"],
        "habitat": ["mixed conifer-aspen (transitional)", "aspen grove (mature)", "sagebrush-grassland mosaic"],
        "elevation_m": (1900, 2800),
        "movement_km_day": (18, 35),  # DERIVED: fall territory defense
        "prey": ["elk herd (large, 50+ head)", "lone bull elk (rutting season)", "mule deer group (5-15)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with full pack (7-12 members detected)", "near rival pack (<3 km, same drainage)"],
    },
    "oct": {
        "weight_kg": (54, 66), "bcs": (5, 7),  # DERIVED: pre-winter buildup, easy elk prey
        "activity": ["hunting (active chase)", "feeding at fresh kill site", "scent-marking territory boundary"],
        "habitat": ["sagebrush-grassland mosaic", "open grassland (valley floor)", "riparian cottonwood gallery"],
        "elevation_m": (1800, 2500),
        "movement_km_day": (18, 32),  # DERIVED
        "prey": ["lone bull elk (rutting season)", "elk herd (large, 50+ head)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with full pack (7-12 members detected)"],
    },
    "nov": {
        "weight_kg": (50, 62), "bcs": (4, 6),  # DERIVED: winter decline starting
        "activity": ["hunting (stalking)", "traveling through drainage", "feeding at fresh kill site"],
        "habitat": ["sagebrush-grassland mosaic", "lodgepole pine forest (open/burned)", "open grassland (valley floor)"],
        "elevation_m": (1700, 2300),
        "movement_km_day": (18, 32),  # DERIVED
        "prey": ["elk herd (large, 50+ head)", "elk herd (moderate, 20-50 head)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with full pack (4-6 members detected)", "with full pack (7-12 members detected)"],
    },
    "dec": {
        "weight_kg": (46, 58), "bcs": (3, 6),  # DERIVED: declining toward winter low
        "activity": ["hunting (stalking)", "traveling on ridgeline", "feeding at old carcass (scavenging)"],
        "habitat": ["sagebrush-grassland mosaic", "open grassland (valley floor)", "lodgepole pine forest (dense)"],
        "elevation_m": (1600, 2200),
        "movement_km_day": (18, 32),  # DERIVED
        "prey": ["elk herd (large, 50+ head)", "elk carcass (old, scavenging)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with full pack (4-6 members detected)", "with full pack (7-12 members detected)"],
    },
}

# ELK — Annual trajectory (adult cow, Yellowstone Northern Range)
# Anchor values from SPECIES_PARAMETERS: cows ~227 kg avg. Calving late May-June.
# Migration: winter low elevation (1600-2000m) → summer high (2500-3200m).
# Monthly values DERIVED by interpolation from seasonal anchors.
ELK_COW_ANNUAL = {
    "jan": {
        "weight_kg": (200, 230), "bcs": (3, 5),  # DERIVED: winter range, weight declining
        "activity": ["resting in open meadow", "traveling through drainage", "feeding at old carcass (scavenging)"],
        "habitat": ["sagebrush-grassland mosaic", "open grassland (valley floor)"],
        "elevation_m": (1600, 2000),  # winter range, low elevation
        "movement_km_day": (3, 10),  # DERIVED: conserving energy
        "prey": [],  # herbivore — no prey association
        "reproductive": "non-breeding subordinate adult",
        "social": ["with full pack (7-12 members detected)"],  # large winter herds
    },
    "feb": {
        "weight_kg": (190, 220), "bcs": (2, 4),  # DERIVED: lowest weight, late winter stress
        "activity": ["resting in open meadow", "traveling through drainage"],
        "habitat": ["sagebrush-grassland mosaic", "open grassland (valley floor)"],
        "elevation_m": (1600, 2000),
        "movement_km_day": (3, 8),  # DERIVED: minimal movement, energy conservation
        "prey": [],
        "reproductive": "pregnant (early — no visible change)",  # bred Nov-Dec
        "social": ["with full pack (7-12 members detected)"],
    },
    "mar": {
        "weight_kg": (195, 225), "bcs": (2, 4),  # DERIVED: slight recovery starting
        "activity": ["traveling through drainage", "resting in open meadow"],
        "habitat": ["sagebrush-grassland mosaic", "open grassland (valley floor)", "riparian cottonwood gallery"],
        "elevation_m": (1700, 2200),  # DERIVED: beginning upslope movement
        "movement_km_day": (5, 12),  # DERIVED: spring migration starting
        "prey": [],
        "reproductive": "pregnant (late — visibly gravid)",
        "social": ["with full pack (4-6 members detected)"],
    },
    "apr": {
        "weight_kg": (200, 235), "bcs": (3, 5),  # DERIVED: green-up, recovery
        "activity": ["traveling through drainage", "resting in open meadow"],
        "habitat": ["mixed conifer-aspen (transitional)", "subalpine meadow (near treeline)", "open grassland (valley floor)"],
        "elevation_m": (1900, 2500),  # DERIVED: spring migration upslope
        "movement_km_day": (5, 15),  # DERIVED: active migration
        "prey": [],
        "reproductive": "pregnant (late — visibly gravid)",
        "social": ["with full pack (4-6 members detected)"],
    },
    "may": {
        "weight_kg": (195, 230), "bcs": (3, 5),  # DERIVED: pre-calving, some weight to calf
        "activity": ["resting in open meadow", "traveling through drainage"],
        "habitat": ["subalpine meadow (near treeline)", "mixed conifer-aspen (transitional)", "dense willow thicket (riparian)"],
        "elevation_m": (2100, 2700),  # DERIVED: approaching summer range
        "movement_km_day": (3, 8),  # DERIVED: restricted movement near calving
        "prey": [],
        "reproductive": "denning (pre-whelp, restricted movement)",  # calving late May
        "social": ["alone (no other collared animals within 2 km)"],  # cows isolate to calve
    },
    "jun": {
        "weight_kg": (190, 225), "bcs": (3, 5),  # DERIVED: post-calving dip, lactation demand
        "activity": ["resting in open meadow", "traveling through drainage"],
        "habitat": ["subalpine meadow (near treeline)", "alpine meadow (above treeline)"],
        "elevation_m": (2300, 2900),  # DERIVED: summer range
        "movement_km_day": (3, 10),  # DERIVED: calf at heel
        "prey": [],
        "reproductive": "nursing (pups at den entrance, 3-6 weeks)",  # nursing calf
        "social": ["with pups at rendezvous site"],  # cow-calf groups
    },
    "jul": {
        "weight_kg": (210, 250), "bcs": (5, 7),  # DERIVED: recovery, peak forage
        "activity": ["resting in open meadow", "traveling through drainage"],
        "habitat": ["alpine meadow (above treeline)", "subalpine meadow (near treeline)"],
        "elevation_m": (2500, 3200),  # summer high elevation
        "movement_km_day": (5, 12),  # DERIVED
        "prey": [],
        "reproductive": "nursing (pups at rendezvous site, 6-12 weeks)",
        "social": ["with pups at rendezvous site", "with full pack (4-6 members detected)"],
    },
    "aug": {
        "weight_kg": (220, 260), "bcs": (5, 7),  # DERIVED: near peak condition
        "activity": ["resting in open meadow", "traveling on ridgeline"],
        "habitat": ["alpine meadow (above treeline)", "subalpine meadow (near treeline)"],
        "elevation_m": (2400, 3100),
        "movement_km_day": (5, 15),  # DERIVED
        "prey": [],
        "reproductive": "post-weaning (pups independent, >16 weeks)",
        "social": ["with full pack (4-6 members detected)"],
    },
    "sep": {
        "weight_kg": (225, 265), "bcs": (5, 7),  # DERIVED: peak condition for cows
        "activity": ["traveling through drainage", "resting in open meadow"],
        "habitat": ["subalpine meadow (near treeline)", "mixed conifer-aspen (transitional)"],
        "elevation_m": (2200, 2800),  # DERIVED: beginning downslope
        "movement_km_day": (5, 15),  # DERIVED
        "prey": [],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with full pack (4-6 members detected)"],
    },
    "oct": {
        "weight_kg": (220, 260), "bcs": (5, 7),  # DERIVED: maintaining condition
        "activity": ["traveling through drainage", "resting in open meadow"],
        "habitat": ["mixed conifer-aspen (transitional)", "sagebrush-grassland mosaic"],
        "elevation_m": (1900, 2500),  # DERIVED: fall migration downslope
        "movement_km_day": (5, 15),  # DERIVED
        "prey": [],
        "reproductive": "non-breeding subordinate adult",  # pre-rut
        "social": ["with full pack (7-12 members detected)"],  # herds consolidating
    },
    "nov": {
        "weight_kg": (215, 250), "bcs": (4, 6),  # DERIVED: rut, early winter
        "activity": ["traveling through drainage", "resting in open meadow"],
        "habitat": ["sagebrush-grassland mosaic", "open grassland (valley floor)"],
        "elevation_m": (1700, 2200),  # DERIVED: approaching winter range
        "movement_km_day": (5, 12),  # DERIVED
        "prey": [],
        "reproductive": "courting (pair bonding observed)",  # elk rut Nov-Dec
        "social": ["with full pack (7-12 members detected)"],
    },
    "dec": {
        "weight_kg": (205, 240), "bcs": (4, 5),  # DERIVED: early winter decline
        "activity": ["resting in open meadow", "traveling through drainage"],
        "habitat": ["sagebrush-grassland mosaic", "open grassland (valley floor)"],
        "elevation_m": (1600, 2000),  # winter range
        "movement_km_day": (3, 10),  # DERIVED
        "prey": [],
        "reproductive": "breeding confirmed (observed tie)",
        "social": ["with full pack (7-12 members detected)"],
    },
}

# ELK — Annual trajectory (adult bull, Yellowstone)
# Anchor: bulls ~318 kg avg, up to 408 kg. Lose 20% body weight during Sep-Oct rut.
# Key difference from cow: OPPOSITE weight trajectory in fall (bulls lose, cows maintain).
ELK_BULL_ANNUAL = {
    "jan": {
        "weight_kg": (260, 310), "bcs": (3, 4),  # DERIVED: post-rut low, still recovering
        "activity": ["resting in open meadow", "traveling through drainage"],
        "habitat": ["sagebrush-grassland mosaic", "open grassland (valley floor)"],
        "elevation_m": (1600, 2000),
        "movement_km_day": (3, 8),  # DERIVED: conserving energy post-rut
        "prey": [],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with 1-2 subordinate pack members"],  # bachelor groups
    },
    "feb": {
        "weight_kg": (250, 300), "bcs": (2, 4),  # DERIVED: lowest, post-rut + winter
        "activity": ["resting in open meadow"],
        "habitat": ["sagebrush-grassland mosaic", "open grassland (valley floor)"],
        "elevation_m": (1600, 2000),
        "movement_km_day": (3, 8),  # DERIVED
        "prey": [],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with 1-2 subordinate pack members"],
    },
    "mar": {
        "weight_kg": (260, 310), "bcs": (3, 4),  # DERIVED: recovery begins
        "activity": ["traveling through drainage", "resting in open meadow"],
        "habitat": ["sagebrush-grassland mosaic", "open grassland (valley floor)"],
        "elevation_m": (1700, 2200),  # DERIVED
        "movement_km_day": (5, 12),  # DERIVED
        "prey": [],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with 1-2 subordinate pack members"],
    },
    "apr": {
        "weight_kg": (280, 330), "bcs": (4, 5),  # DERIVED: green-up, good forage
        "activity": ["traveling through drainage", "resting in open meadow"],
        "habitat": ["mixed conifer-aspen (transitional)", "subalpine meadow (near treeline)"],
        "elevation_m": (1900, 2500),  # DERIVED
        "movement_km_day": (5, 15),  # DERIVED
        "prey": [],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with 1-2 subordinate pack members"],
    },
    "may": {
        "weight_kg": (290, 350), "bcs": (4, 6),  # DERIVED: gaining, antlers growing
        "activity": ["resting in open meadow", "traveling on ridgeline"],
        "habitat": ["subalpine meadow (near treeline)", "alpine meadow (above treeline)"],
        "elevation_m": (2100, 2700),  # DERIVED
        "movement_km_day": (5, 15),  # DERIVED
        "prey": [],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with 1-2 subordinate pack members"],
    },
    "jun": {
        "weight_kg": (310, 370), "bcs": (5, 7),  # DERIVED: rapid weight gain
        "activity": ["resting in open meadow", "traveling on ridgeline"],
        "habitat": ["subalpine meadow (near treeline)", "alpine meadow (above treeline)"],
        "elevation_m": (2300, 3000),  # DERIVED
        "movement_km_day": (5, 12),  # DERIVED
        "prey": [],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with 1-2 subordinate pack members"],
    },
    "jul": {
        "weight_kg": (330, 400), "bcs": (6, 8),  # DERIVED: approaching peak
        "activity": ["resting in open meadow", "traveling on ridgeline"],
        "habitat": ["alpine meadow (above treeline)", "subalpine meadow (near treeline)"],
        "elevation_m": (2500, 3200),
        "movement_km_day": (5, 12),  # DERIVED
        "prey": [],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with 1-2 subordinate pack members"],  # bachelor herds
    },
    "aug": {
        "weight_kg": (350, 420), "bcs": (7, 9),  # DERIVED: peak condition, pre-rut
        "activity": ["resting in open meadow", "on exposed ridge scanning"],
        "habitat": ["alpine meadow (above treeline)", "subalpine meadow (near treeline)"],
        "elevation_m": (2400, 3100),
        "movement_km_day": (5, 15),  # DERIVED: increasing as rut approaches
        "prey": [],
        "reproductive": "non-breeding subordinate adult",  # antlers hardening
        "social": ["with 1-2 subordinate pack members"],
    },
    "sep": {
        "weight_kg": (320, 390), "bcs": (5, 7),  # DERIVED: rut begins, weight DROPPING fast
        "activity": ["howling (stationary)", "traveling on ridgeline", "on exposed ridge scanning"],  # bugling
        "habitat": ["subalpine meadow (near treeline)", "mixed conifer-aspen (transitional)"],
        "elevation_m": (2200, 2800),  # DERIVED
        "movement_km_day": (8, 25),  # DERIVED: rut activity, harem defense
        "prey": [],
        "reproductive": "courting (pair bonding observed)",
        "social": ["with full pack (7-12 members detected)"],  # harem groups
    },
    "oct": {
        "weight_kg": (280, 350), "bcs": (4, 6),  # DERIVED: 20% weight loss during peak rut
        "activity": ["howling (stationary)", "traveling through drainage"],  # bugling
        "habitat": ["mixed conifer-aspen (transitional)", "sagebrush-grassland mosaic"],
        "elevation_m": (1900, 2500),  # DERIVED
        "movement_km_day": (8, 20),  # DERIVED: still active in rut
        "prey": [],
        "reproductive": "breeding confirmed (observed tie)",
        "social": ["with full pack (7-12 members detected)"],
    },
    "nov": {
        "weight_kg": (265, 320), "bcs": (3, 5),  # DERIVED: post-rut exhaustion
        "activity": ["resting in open meadow", "traveling through drainage"],
        "habitat": ["sagebrush-grassland mosaic", "open grassland (valley floor)"],
        "elevation_m": (1700, 2200),  # DERIVED
        "movement_km_day": (3, 10),  # DERIVED: exhausted
        "prey": [],
        "reproductive": "non-breeding subordinate adult",
        "social": ["alone (no other collared animals within 2 km)"],  # solitary post-rut
    },
    "dec": {
        "weight_kg": (260, 315), "bcs": (3, 4),  # DERIVED: early winter, still depleted
        "activity": ["resting in open meadow"],
        "habitat": ["sagebrush-grassland mosaic", "open grassland (valley floor)"],
        "elevation_m": (1600, 2000),
        "movement_km_day": (3, 8),  # DERIVED
        "prey": [],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with 1-2 subordinate pack members"],  # reforming bachelor groups
    },
}

# GRIZZLY BEAR — Annual trajectory (adult, Greater Yellowstone)
# Anchor values from SPECIES_PARAMETERS: M 136-318 kg avg ~200-230 kg; F 91-181 kg.
# Using mid-range adult male (~220 kg baseline). HIBERNATION Nov-Mar.
# Nov-Mar values are HIBERNATION — marked as such, no observations generated.
GRIZZLY_BEAR_ANNUAL = {
    "jan": "HIBERNATION",   # No observations. Bears in den.
    "feb": "HIBERNATION",   # No observations. Bears in den.
    "mar": "HIBERNATION",   # No observations. Most bears still denned (males emerge late March).
    "apr": {
        "weight_kg": (150, 200), "bcs": (2, 3),  # DERIVED: post-hibernation, lost 15-30% from fall
        "activity": ["resting in open meadow", "digging at rodent burrow", "traveling through drainage"],
        "habitat": ["riparian cottonwood gallery", "dense willow thicket (riparian)", "open grassland (valley floor)"],
        "elevation_m": (1600, 2200),  # low elevation, near streams
        "movement_km_day": (2, 8),  # DERIVED: sluggish post-hibernation
        "prey": ["elk carcass (old, scavenging)"],  # scavenging winter-killed elk
        "reproductive": "non-breeding subordinate adult",
        "social": ["alone (no other collared animals within 2 km)"],  # solitary
    },
    "may": {
        "weight_kg": (160, 215), "bcs": (3, 4),  # DERIVED: recovery, digging roots/eating vegetation
        "activity": ["digging at rodent burrow", "traveling through drainage", "feeding at old carcass (scavenging)"],
        "habitat": ["subalpine meadow (near treeline)", "dense willow thicket (riparian)", "open grassland (valley floor)"],
        "elevation_m": (1800, 2500),  # DERIVED: moving upslope
        "movement_km_day": (3, 12),  # DERIVED
        "prey": ["elk cow-calf group", "elk carcass (old, scavenging)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["alone (no other collared animals within 2 km)"],
    },
    "jun": {
        "weight_kg": (175, 230), "bcs": (3, 5),  # DERIVED: steady recovery
        "activity": ["traveling through drainage", "feeding at fresh kill site", "digging at rodent burrow"],
        "habitat": ["subalpine meadow (near treeline)", "alpine meadow (above treeline)", "mixed conifer-aspen (transitional)"],
        "elevation_m": (2000, 2800),  # DERIVED: following green-up upslope
        "movement_km_day": (5, 15),  # DERIVED: active foraging
        "prey": ["elk cow-calf group", "mule deer group (5-15)"],
        "reproductive": "courting (pair bonding observed)",  # mating season Jun-Jul
        "social": ["alone (no other collared animals within 2 km)", "with non-pack disperser (potential mate)"],
    },
    "jul": {
        "weight_kg": (190, 250), "bcs": (4, 6),  # DERIVED: berries starting, weight climbing
        "activity": ["feeding at fresh kill site", "traveling on ridgeline", "swimming/crossing river"],
        "habitat": ["alpine meadow (above treeline)", "subalpine meadow (near treeline)", "dense willow thicket (riparian)"],
        "elevation_m": (2200, 3000),
        "movement_km_day": (5, 18),  # DERIVED
        "prey": ["mule deer group (5-15)", "elk herd (small, 5-20 head)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["alone (no other collared animals within 2 km)"],
    },
    "aug": {
        "weight_kg": (210, 280), "bcs": (5, 7),  # DERIVED: hyperphagia starting, berry season
        "activity": ["feeding at fresh kill site", "traveling through drainage", "digging at rodent burrow"],
        "habitat": ["alpine meadow (above treeline)", "subalpine meadow (near treeline)", "recent burn scar (<5 years, sparse regrowth)"],
        "elevation_m": (2000, 2800),  # DERIVED: following berry patches
        "movement_km_day": (5, 20),  # DERIVED: covering ground for food
        "prey": ["elk herd (moderate, 20-50 head)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["alone (no other collared animals within 2 km)"],
    },
    "sep": {
        "weight_kg": (230, 300), "bcs": (6, 8),  # DERIVED: peak hyperphagia, gaining 1-2 kg/day
        "activity": ["feeding at fresh kill site", "traveling through drainage"],
        "habitat": ["subalpine meadow (near treeline)", "riparian cottonwood gallery", "dense willow thicket (riparian)"],
        "elevation_m": (1800, 2600),  # DERIVED: dropping toward den elevation
        "movement_km_day": (3, 15),  # DERIVED: focused on eating, less exploring
        "prey": ["elk herd (large, 50+ head)", "elk carcass (old, scavenging)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["alone (no other collared animals within 2 km)"],
    },
    "oct": {
        "weight_kg": (250, 320), "bcs": (7, 9),  # DERIVED: near peak, pre-den weight
        "activity": ["feeding at fresh kill site", "traveling through drainage", "resting in open meadow"],
        "habitat": ["spruce-fir forest (dense canopy)", "subalpine meadow (near treeline)"],
        "elevation_m": (2000, 2800),  # DERIVED: seeking den site at higher elevation
        "movement_km_day": (2, 10),  # DERIVED: slowing down, preparing to den
        "prey": ["elk carcass (old, scavenging)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["alone (no other collared animals within 2 km)"],
    },
    "nov": "HIBERNATION",   # Females denned by late November. Males may still be active early Nov.
    "dec": "HIBERNATION",   # All bears in den.
}

# MOUNTAIN LION — Annual trajectory (adult, Northern Rockies)
# Anchor values from SPECIES_PARAMETERS: M 52-100 kg; F 29-64 kg. Using mid-range adult.
# No hibernation. No strong seasonal weight cycle. Year-round hunter.
# Key: nocturnal/crepuscular activity, sparse observations (GPS only).
MOUNTAIN_LION_ANNUAL = {
    "jan": {
        "weight_kg": (55, 80), "bcs": (4, 6),  # DERIVED: stable, year-round hunter
        "activity": ["hunting (ambush)", "resting at den site", "traveling through drainage"],
        "habitat": ["spruce-fir forest (dense canopy)", "rocky outcrop (cliff face)", "lodgepole pine forest (dense)"],
        "elevation_m": (1500, 2400),  # DERIVED: follows deer to lower elevations in winter
        "movement_km_day": (3, 10),  # DERIVED: from species params ~3.7 km/day avg
        "prey": ["mule deer group (5-15)", "elk herd (small, 5-20 head)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["alone (no other collared animals within 2 km)"],  # solitary
    },
    "feb": {
        "weight_kg": (55, 80), "bcs": (4, 6),  # DERIVED: stable
        "activity": ["hunting (ambush)", "traveling through drainage", "resting at den site"],
        "habitat": ["spruce-fir forest (dense canopy)", "rocky outcrop (cliff face)"],
        "elevation_m": (1500, 2300),  # DERIVED
        "movement_km_day": (3, 10),  # DERIVED
        "prey": ["mule deer group (5-15)", "elk herd (small, 5-20 head)"],
        "reproductive": "courting (pair bonding observed)",  # breeding can occur any time, peaks vary
        "social": ["alone (no other collared animals within 2 km)", "with non-pack disperser (potential mate)"],
    },
    "mar": {
        "weight_kg": (55, 80), "bcs": (4, 6),  # DERIVED
        "activity": ["hunting (ambush)", "traveling through drainage"],
        "habitat": ["spruce-fir forest (dense canopy)", "rocky outcrop (cliff face)", "lodgepole pine forest (dense)"],
        "elevation_m": (1600, 2400),  # DERIVED
        "movement_km_day": (3, 10),  # DERIVED
        "prey": ["mule deer group (5-15)", "elk herd (small, 5-20 head)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["alone (no other collared animals within 2 km)"],
    },
    "apr": {
        "weight_kg": (55, 82), "bcs": (4, 6),  # DERIVED: slight improvement with spring prey
        "activity": ["hunting (ambush)", "hunting (stalking)", "traveling through drainage"],
        "habitat": ["mixed conifer-aspen (transitional)", "spruce-fir forest (dense canopy)", "rocky outcrop (cliff face)"],
        "elevation_m": (1700, 2600),  # DERIVED: following deer upslope
        "movement_km_day": (3, 12),  # DERIVED
        "prey": ["mule deer group (5-15)", "elk cow-calf group"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["alone (no other collared animals within 2 km)"],
    },
    "may": {
        "weight_kg": (55, 85), "bcs": (4, 7),  # DERIVED: abundant fawns/calves
        "activity": ["hunting (ambush)", "hunting (stalking)", "resting at den site"],
        "habitat": ["mixed conifer-aspen (transitional)", "spruce-fir forest (dense canopy)"],
        "elevation_m": (1800, 2700),  # DERIVED
        "movement_km_day": (3, 12),  # DERIVED
        "prey": ["mule deer group (5-15)", "elk cow-calf group"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["alone (no other collared animals within 2 km)"],
    },
    "jun": {
        "weight_kg": (55, 85), "bcs": (5, 7),  # DERIVED: peak prey availability (fawns)
        "activity": ["hunting (ambush)", "feeding at fresh kill site", "resting at den site"],
        "habitat": ["subalpine meadow (near treeline)", "mixed conifer-aspen (transitional)", "rocky outcrop (cliff face)"],
        "elevation_m": (2000, 2900),  # DERIVED
        "movement_km_day": (3, 10),  # DERIVED
        "prey": ["mule deer group (5-15)", "elk cow-calf group", "bighorn sheep band (8-20)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["alone (no other collared animals within 2 km)"],
    },
    "jul": {
        "weight_kg": (55, 85), "bcs": (5, 7),  # DERIVED: stable, good hunting
        "activity": ["hunting (ambush)", "resting at den site", "traveling on ridgeline"],
        "habitat": ["subalpine meadow (near treeline)", "rocky outcrop (cliff face)", "alpine meadow (above treeline)"],
        "elevation_m": (2100, 3000),  # DERIVED
        "movement_km_day": (3, 10),  # DERIVED
        "prey": ["mule deer group (5-15)", "bighorn sheep band (8-20)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["alone (no other collared animals within 2 km)"],
    },
    "aug": {
        "weight_kg": (55, 85), "bcs": (5, 7),  # DERIVED
        "activity": ["hunting (ambush)", "traveling on ridgeline", "resting at den site"],
        "habitat": ["subalpine meadow (near treeline)", "rocky outcrop (cliff face)"],
        "elevation_m": (2000, 2900),  # DERIVED
        "movement_km_day": (3, 12),  # DERIVED
        "prey": ["mule deer group (5-15)", "elk herd (moderate, 20-50 head)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["alone (no other collared animals within 2 km)"],
    },
    "sep": {
        "weight_kg": (55, 85), "bcs": (5, 7),  # DERIVED
        "activity": ["hunting (ambush)", "traveling through drainage", "resting at den site"],
        "habitat": ["mixed conifer-aspen (transitional)", "spruce-fir forest (dense canopy)"],
        "elevation_m": (1900, 2700),  # DERIVED: following deer downslope
        "movement_km_day": (3, 12),  # DERIVED
        "prey": ["mule deer group (5-15)", "elk herd (moderate, 20-50 head)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["alone (no other collared animals within 2 km)"],
    },
    "oct": {
        "weight_kg": (55, 82), "bcs": (4, 6),  # DERIVED
        "activity": ["hunting (ambush)", "traveling through drainage"],
        "habitat": ["spruce-fir forest (dense canopy)", "mixed conifer-aspen (transitional)", "rocky outcrop (cliff face)"],
        "elevation_m": (1700, 2500),  # DERIVED
        "movement_km_day": (3, 12),  # DERIVED
        "prey": ["mule deer group (5-15)", "elk herd (moderate, 20-50 head)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["alone (no other collared animals within 2 km)"],
    },
    "nov": {
        "weight_kg": (55, 80), "bcs": (4, 6),  # DERIVED
        "activity": ["hunting (ambush)", "traveling through drainage", "resting at den site"],
        "habitat": ["spruce-fir forest (dense canopy)", "rocky outcrop (cliff face)"],
        "elevation_m": (1500, 2400),  # DERIVED
        "movement_km_day": (3, 10),  # DERIVED
        "prey": ["mule deer group (5-15)", "elk herd (small, 5-20 head)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["alone (no other collared animals within 2 km)"],
    },
    "dec": {
        "weight_kg": (55, 80), "bcs": (4, 6),  # DERIVED
        "activity": ["hunting (ambush)", "resting at den site", "traveling through drainage"],
        "habitat": ["spruce-fir forest (dense canopy)", "rocky outcrop (cliff face)", "lodgepole pine forest (dense)"],
        "elevation_m": (1500, 2300),  # DERIVED
        "movement_km_day": (3, 10),  # DERIVED
        "prey": ["mule deer group (5-15)", "elk herd (small, 5-20 head)"],
        "reproductive": "non-breeding subordinate adult",
        "social": ["alone (no other collared animals within 2 km)"],
    },
}

# PRONGHORN — Annual trajectory (adult, Greater Yellowstone / sagebrush steppe)
# Anchor values from SPECIES_PARAMETERS: M 40-65 kg, F 34-48 kg. Using mid-range adult.
# Highly migratory: summer high elevation → winter low sagebrush steppe.
# Fawning late May-early June. Twins typical.
PRONGHORN_ANNUAL = {
    "jan": {
        "weight_kg": (36, 50), "bcs": (3, 4),  # DERIVED: winter, weight declining
        "activity": ["resting in open meadow", "traveling through drainage"],
        "habitat": ["sagebrush steppe (open)", "sagebrush-grassland mosaic", "open grassland (valley floor)"],
        "elevation_m": (1500, 2000),  # DERIVED: winter range, low elevation sagebrush
        "movement_km_day": (3, 10),  # DERIVED: conserving energy
        "prey": [],  # herbivore
        "reproductive": "non-breeding subordinate adult",
        "social": ["with full pack (7-12 members detected)"],  # winter aggregations
    },
    "feb": {
        "weight_kg": (34, 48), "bcs": (2, 4),  # DERIVED: late winter low
        "activity": ["resting in open meadow"],
        "habitat": ["sagebrush steppe (open)", "sagebrush-grassland mosaic"],
        "elevation_m": (1500, 1900),  # DERIVED
        "movement_km_day": (3, 8),  # DERIVED
        "prey": [],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with full pack (7-12 members detected)"],
    },
    "mar": {
        "weight_kg": (35, 50), "bcs": (3, 4),  # DERIVED: recovery starting
        "activity": ["traveling through drainage", "resting in open meadow"],
        "habitat": ["sagebrush steppe (open)", "sagebrush-grassland mosaic"],
        "elevation_m": (1500, 2000),  # DERIVED
        "movement_km_day": (5, 15),  # DERIVED: spring migration starting
        "prey": [],
        "reproductive": "pregnant (early — no visible change)",
        "social": ["with full pack (4-6 members detected)"],  # herds breaking up
    },
    "apr": {
        "weight_kg": (38, 52), "bcs": (3, 5),  # DERIVED: spring green-up
        "activity": ["traveling through drainage", "resting in open meadow"],
        "habitat": ["sagebrush-grassland mosaic", "open grassland (valley floor)"],
        "elevation_m": (1600, 2200),  # DERIVED: migrating upslope
        "movement_km_day": (8, 25),  # DERIVED: active migration
        "prey": [],
        "reproductive": "pregnant (late — visibly gravid)",
        "social": ["with full pack (4-6 members detected)"],
    },
    "may": {
        "weight_kg": (36, 50), "bcs": (3, 5),  # DERIVED: pre-fawning weight
        "activity": ["resting in open meadow", "traveling through drainage"],
        "habitat": ["sagebrush-grassland mosaic", "open grassland (valley floor)", "subalpine meadow (near treeline)"],
        "elevation_m": (1800, 2400),  # DERIVED
        "movement_km_day": (3, 10),  # DERIVED: restricted near fawning
        "prey": [],
        "reproductive": "denning (pre-whelp, restricted movement)",  # fawning late May
        "social": ["alone (no other collared animals within 2 km)"],  # females isolate to fawn
    },
    "jun": {
        "weight_kg": (35, 48), "bcs": (3, 5),  # DERIVED: post-fawning, lactation demand
        "activity": ["resting in open meadow", "traveling through drainage"],
        "habitat": ["open grassland (valley floor)", "sagebrush-grassland mosaic", "subalpine meadow (near treeline)"],
        "elevation_m": (2000, 2600),  # DERIVED: summer range
        "movement_km_day": (3, 10),  # DERIVED: fawns at heel
        "prey": [],
        "reproductive": "nursing (pups at den entrance, 3-6 weeks)",
        "social": ["with pups at den"],  # nursery bands
    },
    "jul": {
        "weight_kg": (40, 55), "bcs": (4, 6),  # DERIVED: recovery, good summer forage
        "activity": ["resting in open meadow", "traveling on ridgeline"],
        "habitat": ["subalpine meadow (near treeline)", "open grassland (valley floor)", "sagebrush-grassland mosaic"],
        "elevation_m": (2000, 2800),  # DERIVED: challenging traditional low-elevation assumption
        "movement_km_day": (5, 15),  # DERIVED
        "prey": [],
        "reproductive": "nursing (pups at rendezvous site, 6-12 weeks)",
        "social": ["with pups at rendezvous site", "with full pack (4-6 members detected)"],
    },
    "aug": {
        "weight_kg": (42, 58), "bcs": (5, 7),  # DERIVED: near peak condition
        "activity": ["traveling on ridgeline", "resting in open meadow"],
        "habitat": ["subalpine meadow (near treeline)", "sagebrush-grassland mosaic"],
        "elevation_m": (2000, 2800),  # DERIVED
        "movement_km_day": (5, 15),  # DERIVED
        "prey": [],
        "reproductive": "post-weaning (pups independent, >16 weeks)",
        "social": ["with full pack (4-6 members detected)"],
    },
    "sep": {
        "weight_kg": (44, 60), "bcs": (5, 7),  # DERIVED: peak condition
        "activity": ["traveling through drainage", "resting in open meadow"],
        "habitat": ["sagebrush-grassland mosaic", "open grassland (valley floor)"],
        "elevation_m": (1800, 2500),  # DERIVED: beginning fall migration
        "movement_km_day": (5, 15),  # DERIVED
        "prey": [],
        "reproductive": "non-breeding subordinate adult",  # pre-rut
        "social": ["with full pack (4-6 members detected)"],
    },
    "oct": {
        "weight_kg": (42, 58), "bcs": (5, 6),  # DERIVED: fall migration
        "activity": ["traveling through drainage", "resting in open meadow"],
        "habitat": ["sagebrush-grassland mosaic", "sagebrush steppe (open)"],
        "elevation_m": (1600, 2200),  # DERIVED: migrating to winter range
        "movement_km_day": (8, 25),  # DERIVED: active migration
        "prey": [],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with full pack (7-12 members detected)"],  # herds forming
    },
    "nov": {
        "weight_kg": (40, 55), "bcs": (4, 6),  # DERIVED
        "activity": ["resting in open meadow", "traveling through drainage"],
        "habitat": ["sagebrush steppe (open)", "sagebrush-grassland mosaic"],
        "elevation_m": (1500, 2000),  # DERIVED: on winter range
        "movement_km_day": (3, 10),  # DERIVED
        "prey": [],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with full pack (7-12 members detected)"],
    },
    "dec": {
        "weight_kg": (38, 52), "bcs": (3, 5),  # DERIVED: early winter decline
        "activity": ["resting in open meadow"],
        "habitat": ["sagebrush steppe (open)", "sagebrush-grassland mosaic"],
        "elevation_m": (1500, 1900),  # DERIVED
        "movement_km_day": (3, 8),  # DERIVED
        "prey": [],
        "reproductive": "non-breeding subordinate adult",
        "social": ["with full pack (7-12 members detected)"],
    },
}

# Master lookup for trajectory curves by species key
SPECIES_TRAJECTORY_CURVES = {
    "gray_wolf_female": WOLF_FEMALE_ANNUAL,
    "gray_wolf_male": WOLF_MALE_ANNUAL,
    "elk_cow": ELK_COW_ANNUAL,
    "elk_bull": ELK_BULL_ANNUAL,
    "grizzly_bear": GRIZZLY_BEAR_ANNUAL,      # Active season only; HIBERNATION months = no events
    "mountain_lion": MOUNTAIN_LION_ANNUAL,
    "pronghorn": PRONGHORN_ANNUAL,
}
```

**Additional species trajectory summaries** (prose form for species without full curves yet — implement curves before coding):

```
BLACK BEAR:
  Similar to grizzly but smaller (M 60-200kg, F 40-80kg).
  Hibernation Nov-Apr (longer than grizzly). Active May-Oct.
  More arboreal. Less aggressive. Urban-interface encounters.

COYOTE:
  Small (8-20 kg). Pair or small pack structure.
  No hibernation. Year-round activity.
  Breeding Jan-Mar, pups born Apr-May.
  Weight relatively stable year-round.

RED FOX:
  Very small (3-7 kg). Solitary or pair.
  Breeding Dec-Feb, kits born Mar-Apr.
  Wide habitat tolerance. Nocturnal/crepuscular.

WOLVERINE:
  Solitary, vast ranges (200-1500 km²). Weight 8-18 kg.
  Alpine/subalpine specialist. Very hard to observe.
  Denning Feb-May. Year-round activity in snow.

CANADA LYNX:
  Solitary. Weight 8-14 kg. Tied to snowshoe hare cycle.
  Boreal forest specialist. Breeding Mar-Apr, kittens born May-Jun.

MULE DEER:
  M 55-120 kg, F 45-75 kg. Winter weight loss 20-25%.
  Migration 32-390 km. Fawning late May-June.
  Similar trajectory shape to elk cow but smaller scale.

WHITETAIL DEER:
  Similar to mule deer. M 60-130 kg, F 40-75 kg.
  Less migratory. Forest edge specialist.

MOOSE:
  Largest cervid. M 542-725 kg, F 364-591 kg.
  Solitary. Wetland/riparian specialist.
  Rut Sep-Oct. Calving mid-May to early June.

BIGHORN SHEEP:
  Rams 73-113 kg, ewes 53-91 kg. Band structure 8-30.
  Seasonal altitudinal migration. Rut Oct-Jan.
  Lambing Apr-Jun. Alpine/cliff specialist.

BISON:
  M 450-900 kg, F 320-545 kg. Large herds.
  Rut Jul-Sep. Calving Apr-May.
  Seasonal range shifts but not long-distance migration.
```

### 3.6.2 Cross-Attribute Correlation Rules (Must Enforce)

These rules prevent ecologically impossible combinations:

```
RULE 1: Season ↔ Activity
  - "denning" activities ONLY in April-May (wolves), never in summer/fall/winter
  - "at rendezvous site with pups" ONLY in June-September
  - "hunting (active chase)" rare in deep winter for bears (they're hibernating)
  - "migrating" for elk ONLY in April-May (spring) and Oct-Nov (fall)

RULE 2: Season ↔ Habitat ↔ Elevation
  - Winter: low elevation, valley floor, sagebrush/grassland
  - Summer: high elevation, alpine/subalpine meadow, forest edges
  - Animals can't be at 3000m in January (deep snow, inaccessible)
  - Exception: mountain goats/bighorn sheep CAN be high in winter

RULE 3: Weight ↔ Body Condition ↔ Season
  - Weight and BCS must move TOGETHER (can't have weight 55kg with BCS 2)
  - Both must follow seasonal arc (declining in winter, recovering spring, peak summer/fall)
  - Post-whelp females: weight dip in May, recovery through summer
  - Rut-affected males (elk, bison): weight loss Sep-Nov despite abundant food

RULE 4: Reproductive Status ↔ Activity ↔ Social Context ↔ Season
  - "pregnant" → activity must shift toward den-site preparation
  - "nursing (pups in den)" → movement < 5 km/day, location fixed near den
  - "post-weaning" → movement increases, pack ranges expand
  - Can't be "nursing" in December or "courting" in July

RULE 5: Health ↔ Weight ↔ Body Condition ↔ Survival
  - Mange progression: healthy → early mange → moderate → advanced → emaciated → mortality
  - Each step: weight drops 5-15%, BCS drops 1-2 points
  - Timeline: progression over 2-6 months without treatment
  - Can't have "sarcoptic mange (advanced)" with BCS 7 and weight 55kg

RULE 6: Pack Membership ↔ Social Context ↔ Location
  - Pack member → should be near other pack members (same zone or adjacent)
  - Disperser → should be ALONE, and location should progressively diverge from pack territory
  - New pair → two animals, same zone, previously from different packs

RULE 7: Location ↔ Previous Location ↔ Time Elapsed
  - Distance between observations ≤ (max_daily_movement × days_elapsed)
  - Zones must be geographically connected (use adjacency graph)
  - Wolves: max 30 km/day → 5 days between observations → max 150 km displacement
  - But TYPICAL displacement is less than maximum (meander, not straight-line)

RULE 8: Prey Association ↔ Predator Activity ↔ Location ↔ Season
  - Wolves near "elk herd (large)" in winter = hunting in low-elevation valleys
  - Wolves near "elk cow-calf group" in May-June = calving season predation
  - Bears near "elk carcass" in spring = scavenging winter-killed elk
  - Mountain lions near "mule deer" = year-round
  - Can't have prey association "bison group" in Isle Royale (no bison there)

RULE 9: Species ↔ Study Area Compatibility
  - Bison ONLY in Yellowstone/Greater Yellowstone
  - Moose: Yellowstone, Great Lakes, Northern Rockies
  - Pronghorn: Yellowstone, sagebrush steppe areas (NOT dense forest study areas)
  - Wolverine: alpine/subalpine only (NOT Great Lakes lowlands)
  - Lynx: boreal forest (Northern Rockies, Great Lakes, NOT sagebrush steppe)

RULE 10: Social Context ↔ Pack Membership Transitions
  Valid social_context values for each pack_membership state:

  pack_membership = "member of [Pack Name]":
    VALID social_context:
      - "with full pack (4-6 members detected)"
      - "with full pack (7-12 members detected)"
      - "with breeding partner only" (if alpha)
      - "with breeding partner and 2-3 yearlings" (if alpha)
      - "with 1-2 subordinate pack members"
      - "with pups at den" (if breeding female, spring)
      - "with pups at rendezvous site" (if breeding female, summer)
      - "near rival pack (<3 km, same drainage)"
      - "near rival pack (<1 km, boundary confrontation)"
      - "at shared kill site with another pack"
      - "trailing elk herd at 200-500m"
      - "near human activity (ranch, road, campground)"
      - "with mixed-species group (e.g., ravens, magpies at carcass)"
    INVALID: "alone" (unless temporarily separated), "with non-pack disperser"
    TRANSITIONS:
      - pack_interaction → "near rival pack (<3 km, same drainage)" or "near rival pack (<1 km, boundary confrontation)"
      - kill_site_investigation → "at shared kill site with another pack" or "trailing elk herd at 200-500m"
      - relocation_observation → any valid pack-member context
      - dispersal_event → transitions to disperser state (see below)

  pack_membership = "disperser (solo)":
    VALID social_context:
      - "alone (no other collared animals within 2 km)"
      - "separated from pack by >10 km"
      - "with non-pack disperser (potential mate)"
      - "with unmarked/uncollared companion(s)"
      - "within 500m of collared animal from different pack"
      - "near human activity (ranch, road, campground)"
    INVALID: "with full pack", "with breeding partner", "with pups at den", "at shared kill site with another pack"
    TRANSITIONS:
      - relocation_observation → "alone" or "separated from pack"
      - visual_observation → can detect "with non-pack disperser (potential mate)"
      - pack_interaction → "within 500m of collared animal from different pack"
      - If animal joins a pack → pack_membership changes, social_context updates to pack-member values

  pack_membership = "disperser (with unnamed companion)":
    VALID social_context:
      - "with unmarked/uncollared companion(s)"
      - "with non-pack disperser (potential mate)"
      - "alone (no other collared animals within 2 km)" (companion temporarily separated)
      - "near human activity (ranch, road, campground)"
    INVALID: "with full pack", "with pups at den"
    TRANSITIONS:
      - visual_observation → "with unmarked/uncollared companion(s)"
      - If pair forms pack → pack_membership = "new pair (forming pack)", social_context = "with breeding partner only"

  pack_membership = "new pair (forming pack)":
    VALID social_context:
      - "with breeding partner only"
      - "alone (no other collared animals within 2 km)" (temporary separation)
      - "near human activity (ranch, road, campground)"
    INVALID: "with full pack (7-12)", "with pups at den" (until first litter)
    TRANSITIONS:
      - denning_event → "with pups at den" (now a breeding pack)
      - Over time → transitions to full pack membership as pups survive

  Solitary species (mountain_lion, wolverine, grizzly_bear, black_bear, moose):
    VALID social_context:
      - "alone (no other collared animals within 2 km)" (default)
      - "with non-pack disperser (potential mate)" (breeding season only)
      - "near human activity (ranch, road, campground)"
      - "with mixed-species group (e.g., ravens, magpies at carcass)"
      - "within 500m of collared animal from different pack" (proximity event)
    INVALID: "with full pack", "with breeding partner and 2-3 yearlings"
    Exception: female with cubs/kittens → "with pups at den" is valid
```

### 3.6.3 Individual Trajectory Archetypes

Each animal is assigned a trajectory at initialization. The trajectory determines how the animal's story progresses across observations:

```python
INDIVIDUAL_TRAJECTORIES = {
    "stable_pack_member": {
        # Most common. Animal stays in pack, seasonal weight arc, no drama.
        "weight_trend": "seasonal_normal",
        "pack_change": False,
        "health_events": [],
        "mortality_risk": 0.05,  # per study period
    },
    "breeding_female": {
        # Denning in spring, restricted movement, weight dip post-whelp, recovery.
        "weight_trend": "breeding_dip_recovery",
        "pack_change": False,
        "special_events": ["denning", "pup_birth", "rendezvous_move"],
        "mortality_risk": 0.03,
    },
    "dispersing_yearling": {
        # Leaves pack mid-study, long-distance movement, eventually finds mate or dies.
        "weight_trend": "slight_decline_then_variable",
        "pack_change": True,  # from pack → disperser
        "dispersal_distance_km": (60, 300),
        "mortality_risk": 0.30,  # dispersers face high mortality
    },
    "declining_health": {
        # Mange, injury, or age-related decline. Weight drops, BCS drops, may die.
        "weight_trend": "declining",
        "health_progression": ["healthy", "early_mange_or_injury", "moderate", "advanced"],
        "mortality_risk": 0.50,
    },
    "territorial_challenger": {
        # Inter-pack conflict. May get injured, may claim new territory.
        "weight_trend": "seasonal_normal",
        "pack_change": "possible",
        "special_events": ["territorial_encounter", "possible_injury"],
        "mortality_risk": 0.15,
    },
    "new_immigrant": {
        # Disperser who joins existing pack. Initial unfamiliarity, integration.
        "weight_trend": "recovery_from_dispersal",
        "pack_change": True,  # disperser → new pack member
        "mortality_risk": 0.10,
    },
}
```

## 4. Narrative Generation

### 4.1 Story Structure

A trial narrative has this structure:

```
[STUDY HEADER]
  - Study area, species, tagged animals, study period, project name

[OBSERVATION BLOCK 1: Early study period]
  - 2-5 observation reports
  - Each mentions 1-3 animals' tracked attribute values

[OBSERVATION BLOCK 2: Mid study period]
  - 3-8 observation reports

[OBSERVATION BLOCK 3: Late study period]
  - 2-5 observation reports

[Optional: STUDY SUMMARY / SEASON END NOTES]
```

The number of blocks and observations per block scales with num_updates.

### 4.2 Narrative Voice — 50/50

**Field report voice** (formal, scientific, third person, past tense):
> "On March 14, wolf WF-2847 was relocated via aerial telemetry 12 km northeast of her last known position in the Copper Creek drainage. Weight at last capture (February 28) was 38.2 kg. She appeared to have separated from the Ridgeline Pack — no other collared pack members were detected within a 3-km radius."

**Research narrative voice** (accessible science writing, still past tense but warmer):
> "The Ridgeline Pack's alpha female had moved again. On March 14, her collar signal placed her deep in the Copper Creek drainage, a good 12 km from where the rest of the pack had bedded down three days earlier. She'd weighed just 38.2 kg at her last capture — lean for a breeding female heading into denning season. Something had pushed her to strike out alone."

Both voices convey the same information (location, weight, pack status) but with very different style.

### 4.3 Narrative Templates

```python
# ── STUDY HEADER TEMPLATES ──

HEADER_TEMPLATES = [
    "The {project_name} monitored {num_animals} {species_str} in the "
    "{study_area} from {start_date} through {end_date}. {collar_description}. "
    "{study_context}.",

    "{study_area} — {season} {year}. Researchers with the {project_name} "
    "tracked {num_animals} GPS-collared {species_str} across {area_size_km2} "
    "square kilometers of {terrain_description}. {study_objective}.",
]

# ── RELOCATION OBSERVATION TEMPLATES ──

RELOCATION_FIELD = [
    "{date}: {animal_id} relocated to {zone}, {distance_km} km {direction} of "
    "{previous_zone}. Elevation {elevation_m} m. {filler_detail}",

    "{date}: Aerial survey detected {animal_id} in {zone} at {elevation_m} m "
    "elevation. {movement_description}. {filler_detail}",

    "On {date}, the collar signal for {animal_id} placed {pronoun} in the "
    "{zone}, approximately {distance_km} km from {pronoun_pos} last known position. "
    "{context_detail}.",

    "GPS data from {date} showed {animal_id} had moved to {zone}. "
    "{pronoun_cap} had covered roughly {distance_km} km since the {prev_date} fix. "
    "{filler_detail}",
]

RELOCATION_NARRATIVE = [
    "{animal_id} was on the move. By {date}, {pronoun_pos} collar placed "
    "{pronoun} in the {zone} — {distance_km} km {direction} of where {pronoun}'d "
    "been {days_elapsed} days earlier. {context_detail}",

    "The {date} telemetry data told a clear story: {animal_id} had pushed "
    "{direction} into the {zone}, covering {distance_km} km. {context_detail}",
]

# ── CAPTURE WORKUP TEMPLATES ──

CAPTURE_FIELD = [
    "{date}: {animal_id} captured at {zone} via {capture_method}. "
    "Weight: {weight_kg} kg ({weight_change_str} from last capture). "
    "Body condition: {bcs}/9. Collar battery: {battery_pct}%. "
    "{health_notes}. Released at {release_time}.",

    "{date}: Capture #{capture_number} for {animal_id} at {zone}. "
    "Morphometrics: {weight_kg} kg, chest girth {girth_cm} cm, "
    "body condition {bcs}/9. {collar_action}. {additional_notes}.",
]

CAPTURE_NARRATIVE = [
    "The research team caught up with {animal_id} on {date}, darting "
    "{pronoun} near {zone}. On the scale, {pronoun} came in at {weight_kg} kg — "
    "{weight_interpretation}. Body condition scored {bcs} out of 9. "
    "{collar_action}. {release_notes}.",

    "It took three days of tracking, but on {date} the team immobilized "
    "{animal_id} at {zone}. The workup painted a {health_adjective} picture: "
    "{weight_kg} kg, body condition {bcs}/9, {health_notes}. {collar_action}.",
]

# ── KILL SITE TEMPLATES ──

KILL_SITE_FIELD = [
    "{date}: Kill site investigated at {zone}. {prey_species} carcass, estimated "
    "{prey_age}. {animal_id} and {num_packmates} other collared animals detected "
    "within {radius_km} km. {consumption_notes}.",

    "{date}: Cluster analysis flagged GPS points for {animal_id} at {zone}. "
    "Field investigation confirmed {prey_species} kill. {carcass_condition}. "
    "{other_animals_present}.",
]

# ── PACK INTERACTION TEMPLATES ──

INTERACTION_FIELD = [
    "{date}: {animal_id_1} ({pack_1}) and {animal_id_2} ({pack_2}) detected "
    "within {distance_m} m of each other near {zone}. {outcome_description}. "
    "{aftermath}.",
]

# ── DISPERSAL TEMPLATES ──

DISPERSAL_FIELD = [
    "{date}: {animal_id} appears to have left the {pack_name}. Collar data "
    "shows a {distance_km}-km movement {direction} over the past {days} days, "
    "with no proximity to other pack members. {animal_id} reclassified as "
    "disperser. Current location: {zone}.",
]

DISPERSAL_NARRATIVE = [
    "By {date}, it was clear that {animal_id} had left the {pack_name}. "
    "{pronoun_cap}'d been drifting {direction} for {days} days, putting "
    "{distance_km} km between {pronoun}self and the rest of the pack. "
    "{dispersal_context}.",
]

# ── DENNING TEMPLATES ──

DENNING_FIELD = [
    "{date}: {animal_id} GPS locations concentrated within 100-m radius at "
    "{zone} for {days} consecutive days. Presumed denning. {pack_activity}.",

    "{date}: Remote camera confirmed {animal_id} at den site in {zone}. "
    "{pup_count} pups observed. {pup_description}. {animal_id} weight "
    "estimated at {weight_estimate_kg} kg based on imagery.",
]

# ── MORTALITY TEMPLATES ──

MORTALITY_FIELD = [
    "{date}: Mortality signal received from {animal_id}'s collar at {zone}. "
    "Field investigation found {cause_description}. {animal_id} ({sex}, "
    "age {age_years}) confirmed dead. {necropsy_notes}.",
]

MORTALITY_NARRATIVE = [
    "The signal everyone dreads came on {date}. {animal_id}'s collar "
    "switched to mortality mode, its accelerometer detecting no movement for "
    "8+ hours. The team found {pronoun} at {zone}. {cause_narrative}. "
    "{animal_id} had been {age_years} years old — {age_context}.",
]

# ── HEALTH CHANGE TEMPLATES ──

HEALTH_FIELD = [
    "{date}: {animal_id} observed at {zone} showing signs of {condition}. "
    "{symptoms}. Body condition estimated {bcs}/9, {change_from_last}. "
    "{prognosis}.",
]

# ── SEASONAL SHIFT TEMPLATES ──

SEASONAL_FIELD = [
    "{date}: Seasonal summary — {num_animals} of {total_animals} collared "
    "{species} in the {study_area} have shifted to {seasonal_behavior}. "
    "{summary_stats}.",
]
```

### 4.4 Location / Zone Pools

Zones are organized by study area. Each study area has 15-30 named zones.

```python
STUDY_AREAS = {
    "northern_yellowstone": {
        "name": "Northern Yellowstone",
        "description": "Lamar Valley and surrounding drainages",
        "elevation_range": (1700, 3200),  # meters
        "area_km2": 2500,
        "zones": [
            {"name": "Lamar Valley", "type": "valley", "elevation": 2000, "adjacent": ["Specimen Ridge", "Druid Peak", "Soda Butte Creek"]},
            {"name": "Specimen Ridge", "type": "ridge", "elevation": 2600, "adjacent": ["Lamar Valley", "Amethyst Creek"]},
            {"name": "Druid Peak", "type": "mountain", "elevation": 2900, "adjacent": ["Lamar Valley", "Cache Creek"]},
            {"name": "Soda Butte Creek", "type": "riparian", "elevation": 2100, "adjacent": ["Lamar Valley", "Cooke City Corridor"]},
            {"name": "Slough Creek", "type": "riparian", "elevation": 1900, "adjacent": ["Lamar Valley", "Buffalo Plateau"]},
            {"name": "Buffalo Plateau", "type": "plateau", "elevation": 2400, "adjacent": ["Slough Creek", "Hellroaring Creek"]},
            {"name": "Hellroaring Creek", "type": "riparian", "elevation": 1800, "adjacent": ["Buffalo Plateau", "Yellowstone River North"]},
            {"name": "Tower Junction", "type": "valley", "elevation": 1900, "adjacent": ["Hellroaring Creek", "Elk Creek"]},
            {"name": "Elk Creek", "type": "riparian", "elevation": 2000, "adjacent": ["Tower Junction", "Washburn Range"]},
            {"name": "Washburn Range", "type": "mountain", "elevation": 3100, "adjacent": ["Elk Creek", "Dunraven Pass"]},
            {"name": "Cache Creek", "type": "drainage", "elevation": 2200, "adjacent": ["Druid Peak", "Miller Creek"]},
            {"name": "Miller Creek", "type": "drainage", "elevation": 2400, "adjacent": ["Cache Creek", "Absaroka Divide"]},
            {"name": "Absaroka Divide", "type": "ridge", "elevation": 3000, "adjacent": ["Miller Creek", "Sunlight Basin"]},
            {"name": "Amethyst Creek", "type": "drainage", "elevation": 2300, "adjacent": ["Specimen Ridge", "Mirror Plateau"]},
            {"name": "Mirror Plateau", "type": "plateau", "elevation": 2700, "adjacent": ["Amethyst Creek", "Pelican Valley"]},
            {"name": "Pelican Valley", "type": "valley", "elevation": 2400, "adjacent": ["Mirror Plateau", "Yellowstone Lake North"]},
            {"name": "Cooke City Corridor", "type": "valley", "elevation": 2300, "adjacent": ["Soda Butte Creek"]},
            {"name": "Yellowstone River North", "type": "riparian", "elevation": 1700, "adjacent": ["Hellroaring Creek", "Gardiner Basin"]},
            {"name": "Gardiner Basin", "type": "basin", "elevation": 1600, "adjacent": ["Yellowstone River North"]},
        ]
    },
    "northern_rockies_generic": {
        "name": "Bitterroot-Selway Wilderness",
        "description": "Central Idaho wilderness complex",
        "elevation_range": (900, 3000),
        "area_km2": 5000,
        "zones": [
            # Adjacency: Selway River is the main artery; drainages feed into it.
            # North-south axis: Lolo Pass (north) → Selway River → Salmon River breaks (south).
            # East-west axis: Gospel Peak / Big Fog (high ridges) above the drainages.
            {"name": "Selway River corridor", "type": "riparian", "elevation": 1000,
             "adjacent": ["Moose Creek drainage", "Running Creek", "White Cap Creek", "Indian Creek meadows"]},
            {"name": "Moose Creek drainage", "type": "drainage", "elevation": 1400,
             "adjacent": ["Selway River corridor", "Bear Creek basin", "Elk Summit"]},
            {"name": "Running Creek", "type": "drainage", "elevation": 1200,
             "adjacent": ["Selway River corridor", "Paradise flats", "Warm Springs Creek"]},
            {"name": "Bear Creek basin", "type": "basin", "elevation": 1800,
             "adjacent": ["Moose Creek drainage", "Gospel Peak ridgeline", "Elk Summit"]},
            {"name": "Magruder Corridor", "type": "valley", "elevation": 1600,
             "adjacent": ["Salmon River breaks", "Tin Cup Creek", "Big Fog Mountain"]},
            {"name": "Gospel Peak ridgeline", "type": "ridge", "elevation": 2700,
             "adjacent": ["Bear Creek basin", "Big Fog Mountain", "Elk Summit"]},
            {"name": "Elk Summit", "type": "pass", "elevation": 2200,
             "adjacent": ["Moose Creek drainage", "Bear Creek basin", "Gospel Peak ridgeline", "Tin Cup Creek"]},
            {"name": "White Cap Creek", "type": "drainage", "elevation": 1300,
             "adjacent": ["Selway River corridor", "Indian Creek meadows", "Big Fog Mountain"]},
            {"name": "Indian Creek meadows", "type": "meadow", "elevation": 1500,
             "adjacent": ["Selway River corridor", "White Cap Creek", "Tin Cup Creek"]},
            {"name": "Tin Cup Creek", "type": "drainage", "elevation": 1700,
             "adjacent": ["Indian Creek meadows", "Magruder Corridor", "Elk Summit"]},
            {"name": "Salmon River breaks", "type": "canyon", "elevation": 1100,
             "adjacent": ["Magruder Corridor", "Paradise flats"]},
            {"name": "Big Fog Mountain", "type": "mountain", "elevation": 2600,
             "adjacent": ["Gospel Peak ridgeline", "Magruder Corridor", "White Cap Creek"]},
            {"name": "Paradise flats", "type": "meadow", "elevation": 1400,
             "adjacent": ["Running Creek", "Salmon River breaks", "Warm Springs Creek"]},
            {"name": "Lolo Pass vicinity", "type": "pass", "elevation": 1600,
             "adjacent": ["Warm Springs Creek", "Running Creek"]},
            {"name": "Warm Springs Creek", "type": "riparian", "elevation": 1300,
             "adjacent": ["Lolo Pass vicinity", "Running Creek", "Paradise flats"]},
        ]
    },
    "great_lakes_generic": {
        "name": "Isle Royale / Upper Peninsula",
        "description": "Great Lakes wolf-moose study area",
        "elevation_range": (180, 425),
        "area_km2": 800,
        "zones": [
            # Isle Royale is a long narrow island (~72 km x 14 km), oriented SW-NE.
            # Greenstone Ridge runs the spine. Bays on north and south shores.
            # Adjacency follows the island's geography: ridges connect along the spine,
            # bays are adjacent to the nearest ridge section and neighboring bays.
            {"name": "Windigo", "type": "bay", "elevation": 185,
             "adjacent": ["Feldtmann Ridge", "Greenstone Ridge west"]},
            {"name": "Feldtmann Ridge", "type": "ridge", "elevation": 380,
             "adjacent": ["Windigo", "Siskiwit Bay", "Greenstone Ridge west"]},
            {"name": "Siskiwit Bay", "type": "bay", "elevation": 185,
             "adjacent": ["Feldtmann Ridge", "Houghton Ridge", "Malone Bay"]},
            {"name": "Greenstone Ridge east", "type": "ridge", "elevation": 400,
             "adjacent": ["Greenstone Ridge west", "Chippewa Harbor", "Tobin Harbor", "Lake Richie area", "Minong Ridge"]},
            {"name": "Greenstone Ridge west", "type": "ridge", "elevation": 390,
             "adjacent": ["Greenstone Ridge east", "Windigo", "Feldtmann Ridge", "Houghton Ridge"]},
            {"name": "Chippewa Harbor", "type": "bay", "elevation": 185,
             "adjacent": ["Greenstone Ridge east", "Lake Richie area", "Malone Bay"]},
            {"name": "Tobin Harbor", "type": "bay", "elevation": 185,
             "adjacent": ["Greenstone Ridge east", "Rock Harbor"]},
            {"name": "Rock Harbor", "type": "bay", "elevation": 185,
             "adjacent": ["Tobin Harbor", "Minong Ridge", "Ishpeming Point"]},
            {"name": "Malone Bay", "type": "bay", "elevation": 185,
             "adjacent": ["Siskiwit Bay", "Chippewa Harbor", "Houghton Ridge"]},
            {"name": "Houghton Ridge", "type": "ridge", "elevation": 350,
             "adjacent": ["Siskiwit Bay", "Malone Bay", "Greenstone Ridge west"]},
            {"name": "Lake Richie area", "type": "inland_lake", "elevation": 220,
             "adjacent": ["Greenstone Ridge east", "Chippewa Harbor"]},
            {"name": "Ishpeming Point", "type": "point", "elevation": 200,
             "adjacent": ["Rock Harbor", "Minong Ridge"]},
            {"name": "Minong Ridge", "type": "ridge", "elevation": 380,
             "adjacent": ["Greenstone Ridge east", "Rock Harbor", "Ishpeming Point"]},
        ]
    }
}
```

### 4.5 Filler / Context Sentences

Between tracked-attribute updates, we add context that mentions non-tracked attributes:

```python
FILLER_TEMPLATES = [
    "Snow depth in {zone} measured {snow_depth_cm} cm, {snow_assessment} for this time of year.",
    "Elk numbers in the {zone} had {elk_trend} since the previous survey — roughly {elk_count} head.",
    "Trail camera at {zone} recorded {camera_observation} over the past week.",
    "The team noted {weather_condition} during the overflight, limiting visibility to {visibility_km} km.",
    "{animal_id}'s collar was transmitting {fix_rate} GPS fixes per day, {battery_note}.",
    "Scat analysis from {zone} showed {diet_detail}.",
    "Howl surveys on {date} detected {howl_response} from the direction of {zone}.",
    "Human activity in the {zone} was {human_activity_level} — {human_detail}.",
    "Prey density in {zone} remained {prey_density_adj}: {prey_detail}.",
    "{adjacent_pack} had been detected {distance_km} km to the {direction}, {interaction_risk}.",
    "Recent {weather_event} had {effect_on_terrain}.",
    "Road surveys along {road_name} found {roadkill_or_tracks}.",
]
```

## 5. Controlling num_keys and num_updates

### num_keys (number of tagged animals)

- num_keys=2: pair study (breeding pair, or predator + prey individual)
- num_keys=3-4: one pack's collared members (not all pack members are collared)
- num_keys=5-6: one pack fully collared, or two packs with partial collars
- num_keys=8-10: multi-pack study in one area
- num_keys=12-15: regional monitoring program, multiple species

Each animal appears in the narrative roughly proportional to how "interesting" it is — a dispersing wolf generates more events than a stable pack member. But the tracked attribute must appear at least `num_updates` times per animal (target, not hard constraint).

### num_updates (state changes per animal)

| num_updates | Observations per animal | ~Word count | Study period covered |
|-------------|------------------------|-------------|---------------------|
| 3 | 3 | 300-500 | ~2-4 weeks |
| 5 | 5 | 500-900 | ~1-2 months |
| 7 | 7 | 800-1300 | ~2-3 months |
| 10 | 10 | 1200-2000 | ~3-6 months |
| 15 | 15 | 2000-3500 | ~6-12 months |
| 20+ | 20+ | 3000-5000 | ~full year+ |

### Interleaving Strategy

Observations naturally interleave because different events concern different animals:

```
Obs 1: Wolf WF-2847 relocated to Copper Creek (WF-2847 weight: 38.2 kg at last capture)
Obs 2: Wolf WM-1203 kill site at Lamar Valley (WM-1203 weight: 52.1 kg)
Obs 3: Elk EM-0456 observed in Pelican Valley (EM-0456 weight: 290 kg estimated)
Obs 4: Wolf WF-2847 captured, weighed (WF-2847 weight: 36.8 kg — lost 1.4 kg)
Obs 5: Wolf WM-3019 dispersed from Ridgeline Pack (WM-3019 weight: 44.6 kg at last capture)
...
```

Constraint: same animal should not be the primary subject of two consecutive observations (interleaving).

## 6. Distinguishing Setups

### Variation dimensions:
1. **Species composition** — wolves only, elk only, wolf-elk mixed, wolf-bear-elk triad
2. **Study area** — Yellowstone, Bitterroot, Great Lakes, generic Northern Rockies
3. **Season start** — winter study, spring denning season, summer field season, fall migration
4. **Study archetype** — determines dominant event patterns (see below)
5. **Queried animal** — which animal the RI/PI question asks about
6. **Queried attribute** — weight vs location vs body condition vs movement distance
7. **Project/team names** — cosmetic variation
8. **Pack dynamics** — stable vs splitting vs merging

### Study archetypes:

```python
STUDY_ARCHETYPES = [
    "stable_monitoring",      # routine season, no major events, steady state
    "dispersal_drama",        # 1-2 animals leave pack, long-distance movement
    "denning_season",         # spring focus, births, weight changes, restricted movement
    "harsh_winter",           # weight loss across all animals, mortality risk, prey scarcity
    "territorial_conflict",   # inter-pack encounters, boundary shifts, possible mortality
    "prey_migration",         # elk move, wolves follow, locations shift across study area
    "disease_outbreak",       # mange or distemper, multiple animals affected
    "reintroduction_study",   # newly released animals, finding territory, high uncertainty
    "mortality_investigation",# one or more animals die, cause-of-death analysis
    "pack_formation",         # two dispersers meet, form new pack, establish territory
]
```

Each archetype produces different trajectory patterns:
- `harsh_winter`: all weights decline, body conditions drop, mortality events likely
- `dispersal_drama`: one animal's location values jump dramatically while others stay stable
- `denning_season`: breeding females become stationary, weights fluctuate (post-birth loss then recovery)

## 7. Name Pools

### Animal ID Convention
```python
# Format: [species][sex]-[4-digit number]
# Species codes: W=wolf, E=elk, G=grizzly, L=mountain lion, P=pronghorn, M=moose
# Sex codes: F=female, M=male

COLLAR_ID_EXAMPLES = [
    "WF-2847",  # Wolf, female, #2847
    "WM-1203",  # Wolf, male, #1203
    "WF-3041",  # Wolf, female, #3041
    "EM-0456",  # Elk, male, #0456
    "GF-0098",  # Grizzly, female, #0098
]
```

### Pack / Herd Names
```python
PACK_NAMES = [
    # Geographic feature names (standard convention)
    "Ridgeline Pack", "Copper Creek Pack", "Lamar Canyon Pack",
    "Druid Peak Pack", "Slough Creek Pack", "Hellroaring Pack",
    "Junction Butte Pack", "Wapiti Lake Pack", "Mollie's Pack",
    "8-Mile Pack", "Leopold Pack", "Prospect Peak Pack",
    "Canyon Pack", "Bechler Pack", "Cougar Creek Pack",
    "Snake River Pack", "Teton Pack", "Gros Ventre Pack",
    "Buffalo Fork Pack", "Pacific Creek Pack", "Phantom Springs Pack",
    "Hoodoo Pack", "Beartooth Pack", "Shoshone Pack",
    "Boulder Creek Pack", "Lava Creek Pack",
]

ELK_HERD_NAMES = [
    "Northern Yellowstone herd", "Madison-Firehole herd",
    "Gallatin herd", "Sand Creek herd", "Gardiner Basin herd",
    "Pelican Valley herd", "Blacktail herd",
]
```

### Common Names (optional, for known individuals)
```python
# Some study animals get common names (especially wolves in Yellowstone)
WOLF_COMMON_NAMES = [
    "Scarface", "Old Blue", "Cinderella", "Big Blaze",
    "White Lady", "Puff", "Limpy", "Ghost", "Shadow",
    "Storm", "Canyon", "Silver", "Jet", "Copper",
]
```

### Project / Study Names
```python
PROJECT_NAMES = [
    "Northern Range Wolf Project", "Greater Yellowstone Carnivore Study",
    "Yellowstone Wolf Recovery Program", "Northern Rockies Wolf Monitoring",
    "Upper Missouri Elk-Predator Study", "Selway-Bitterroot Wolf Assessment",
    "Isle Royale Wolf-Moose Project", "Central Idaho Wolf Recovery",
    "Greater Yellowstone Grizzly Study", "Northern Yellowstone Cooperative Study",
]
```

## 8. Question Templates

```python
RI_TEMPLATES = [
    "What was {animal_id}'s {attribute} at the first observation recorded?",
    "At the earliest mention of {animal_id}'s {attribute}, what was the value?",
    "When {animal_id} first appears in these field notes, what was {pronoun} {attribute}?",
    "What was the initial recorded {attribute} for {animal_id}?",
]

PI_TEMPLATES = [
    "What was {animal_id}'s {attribute} at the most recent observation?",
    "In the last recorded entry for {animal_id}, what was {pronoun} {attribute}?",
    "What was {animal_id}'s most recently noted {attribute}?",
    "At the final mention of {animal_id}, what was {pronoun} {attribute}?",
]
```

## 9. Output Format

Same schema as Dota 2 (universal):

```json
{
    "id": "wildlife_001",
    "domain": "wildlife_tracking",
    "num_keys": 5,
    "num_updates": 7,
    "narrative": "The Northern Range Wolf Project monitored 5 GPS-collared wolves in the Lamar Valley and surrounding drainages from January through April 2024...",
    "questions": {
        "RI": {
            "question": "What was WF-2847's weight at the first observation recorded?",
            "expected_answer": "38.2",
            "target_entity": "WF-2847",
            "target_attribute": "weight_kg"
        },
        "PI": {
            "question": "What was WF-2847's weight at the most recent observation?",
            "expected_answer": "42.7",
            "target_entity": "WF-2847",
            "target_attribute": "weight_kg"
        }
    },
    "entity_tracking": {
        "WF-2847 / weight_kg": ["38.2", "36.8", "37.5", "39.1", "40.3", "41.8", "42.7"],
        "WM-1203 / weight_kg": ["52.1", "50.8", "49.3", "50.6", "51.2", "52.8", "53.4"],
        "WF-3041 / weight_kg": ["41.5", "40.2", "39.6", "40.8", "42.1", "43.0", "43.5"],
        "WM-3019 / weight_kg": ["44.6", "43.1", "42.4", "41.8", "43.5", "44.2", "45.1"],
        "WF-4108 / weight_kg": ["36.0", "35.2", "34.8", "36.1", "37.5", "38.2", "39.0"]
    },
    "config": {
        "species_mode": "same",
        "species": "gray_wolf",
        "study_area": "northern_yellowstone",
        "season_start": "winter",
        "archetype": "harsh_winter",
        "tracked_attribute": "weight_kg",
        "attribute_mode": "same",
        "filler_budget": "medium",
        "voice": "field_report",
        "seed": 42
    }
}
```

## 10. Implementation Order

1. **Species database** — species profiles, weight/movement ranges, seasonal patterns
2. **Study area database** — zone maps with adjacency, elevation, habitat type
3. **Name pools** — collar IDs, pack names, project names, common names
4. **State class** — AnimalState with all mutable attributes (see dataclass below)
5. **Event generators** — one function per event type, each enforcing ecological constraints
6. **Event scheduler** — decides which event happens next based on season and study archetype
7. **Narrative renderer** — converts event sequence into prose using templates
8. **Trial generator** — orchestrates everything into a single trial output
9. **Validation** — verify all tracked values appear in narrative, RI/PI answers correct

### 10.1 AnimalState Dataclass

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class AnimalState:
    """Mutable state for a single tracked animal. Updated by event handlers."""

    # --- Identity (immutable after init) ---
    collar_id: str                          # e.g. "WF-2847"
    common_name: Optional[str] = None       # e.g. "Scarface" (optional)
    species: str = "gray_wolf"              # species key from SPECIES_POOL
    sex: str = "female"                     # "male" | "female"
    age_years: float = 3.0                  # age at study start
    trajectory_archetype: str = "stable_pack_member"  # from INDIVIDUAL_TRAJECTORIES

    # --- Mutable state (updated by events) ---
    alive: bool = True                      # False after mortality_event
    study_day_last_seen: int = 0            # last observation day

    # Numeric continuous
    weight_kg: float = 40.0                 # current weight
    daily_movement_km: float = 15.0         # last observed daily movement
    elevation_m: float = 2000.0             # current elevation
    distance_from_den_km: float = 0.0       # 0 if not denning season
    territory_size_km2: float = 400.0       # current territory estimate
    collar_battery_pct: float = 100.0       # collar battery remaining
    distance_traveled_today_km: float = 10.0  # last day's movement

    # Numeric integer
    body_condition_score: int = 5           # 1-9 Heinze scale
    pup_count: int = 0                      # current live pup count
    kill_count_observed: int = 0            # cumulative observed kills
    days_since_last_sighting: int = 0       # days since last observation
    scent_post_count: int = 0               # last survey count

    # Categorical
    location_zone: str = "Lamar Valley"     # current zone name
    activity_state: str = "resting in open meadow"
    health_status: str = "healthy (average condition)"
    pack_membership: str = "Ridgeline Pack"  # pack name or "disperser (solo)" etc.
    reproductive_status: str = "non-breeding subordinate adult"
    habitat_type: str = "sagebrush-grassland mosaic"
    overnight_location: str = "bedding area (south-facing slope in timber)"
    prey_association: str = "no prey observed (area appears empty)"
    social_context: str = "with full pack (4-6 members detected)"

    # Tracking metadata
    value_history: dict = field(default_factory=dict)
    # {attribute_name: [list of (study_day, value) tuples]}
    # Populated as events update attributes. Used to answer RI/PI questions.

    def update_attribute(self, attr: str, value, study_day: int):
        """Update an attribute and record in history."""
        setattr(self, attr, value)
        if attr not in self.value_history:
            self.value_history[attr] = []
        self.value_history[attr].append((study_day, value))
        self.study_day_last_seen = study_day

    def get_first_value(self, attr: str):
        """RI answer: first recorded value."""
        if attr in self.value_history and self.value_history[attr]:
            return self.value_history[attr][0][1]
        return getattr(self, attr)

    def get_last_value(self, attr: str):
        """PI answer: last recorded value."""
        if attr in self.value_history and self.value_history[attr]:
            return self.value_history[attr][-1][1]
        return getattr(self, attr)
```

## 11. Design Decisions (Resolved)

### Q1: Track weight_kg or location_zone as primary?
**DECISION: Randomly sample from full attribute pool.** weight_kg is the "gold equivalent" (numeric, similar ranges, gradual changes) but the pool has 20 options. Random sampling per trial = maximum variety.

### Q2: Same species or mixed species?
**DECISION: Sample 60/40.** 60% same-species (max interference), 40% mixed (more ecological realism, still creates interference for attributes with overlapping ranges). When mixed, pick 2-3 species that naturally co-occur in the study area.

### Q3: How many study areas?
**DECISION: Start with 3 (Yellowstone, Bitterroot-Selway, Isle Royale).** Each has distinct geography, elevation ranges, and species assemblages. Add more later if needed.

### Q4: Realistic observation frequency?
**DECISION: Keep realistic.** Collar GPS pings: every 4-12 hours. Field observations: weekly to monthly. Captures: 1-2 per year. The narrative reports a subset of observations (the interesting ones). This means some animals may go days/weeks between mentions — that's fine and ecologically authentic.

### Q5: How to handle mortality?
**DECISION: Dead animals stop generating events.** If an animal dies mid-narrative, it gets a mortality event and then no further observations. This naturally reduces num_keys partway through, which is realistic but means some animals have fewer updates than target. Track this in ground truth.

## 12. Trial Config Schema (Final)

```python
from typing import List

@dataclass
class WildlifeTrialConfig:
    num_keys: int              # 2-15 tagged animals
    num_updates: int           # 3-30 observations per animal (target)
    condition: str             # "RI" or "PI"
    seed: int                  # reproducibility
    tracked_attribute: str     # any attribute name from TRACKABLE_ATTRIBUTES, or "mixed"
    attribute_mode: str        # "same" (all animals same attr) | "mixed" (per-animal)
    species_mode: str          # "same" (all one species) | "mixed" (2-3 species)
    species: List[str]         # List of species keys. Single element for same-species mode,
                               # 2-3 elements for mixed-species mode.
                               # e.g. ["gray_wolf"] or ["gray_wolf", "elk", "grizzly_bear"]
    study_area: str            # "northern_yellowstone" | "bitterroot_selway" | "isle_royale"
    season_start: str          # "winter" | "spring" | "summer" | "fall"
    filler_budget: str         # "minimal" | "light" | "medium" | "heavy"
    voice: str                 # "field_report" | "research_narrative"
    study_archetype: str       # see archetypes above
    queried_animal_idx: int    # which animal to ask about (0 to num_keys-1)
```

### 12.1 Species, Study Area, and Archetype Compatibility

```python
# ALL 15 species in the pool
SPECIES_POOL = [
    "gray_wolf", "mountain_lion", "grizzly_bear", "black_bear",
    "coyote", "red_fox", "wolverine", "canada_lynx",
    "elk", "mule_deer", "whitetail_deer", "moose",
    "pronghorn", "bighorn_sheep", "bison",
]

# Which species can appear in which study area.
# Prevents impossible combinations like grizzly + Isle Royale or bison + Bitterroot.
SPECIES_STUDY_AREA_COMPATIBILITY = {
    "gray_wolf":     ["northern_yellowstone", "northern_rockies_generic", "great_lakes_generic"],
    "mountain_lion": ["northern_yellowstone", "northern_rockies_generic"],
    "grizzly_bear":  ["northern_yellowstone", "northern_rockies_generic"],
    "black_bear":    ["northern_yellowstone", "northern_rockies_generic", "great_lakes_generic"],
    "coyote":        ["northern_yellowstone", "northern_rockies_generic", "great_lakes_generic"],
    "red_fox":       ["northern_yellowstone", "northern_rockies_generic", "great_lakes_generic"],
    "wolverine":     ["northern_yellowstone", "northern_rockies_generic"],  # NOT Great Lakes (too low elevation)
    "canada_lynx":   ["northern_rockies_generic", "great_lakes_generic"],   # boreal specialist, NOT Yellowstone sagebrush
    "elk":           ["northern_yellowstone", "northern_rockies_generic"],
    "mule_deer":     ["northern_yellowstone", "northern_rockies_generic"],
    "whitetail_deer":["northern_rockies_generic", "great_lakes_generic"],
    "moose":         ["northern_yellowstone", "northern_rockies_generic", "great_lakes_generic"],
    "pronghorn":     ["northern_yellowstone"],                              # sagebrush steppe only
    "bighorn_sheep": ["northern_yellowstone", "northern_rockies_generic"],  # alpine specialist
    "bison":         ["northern_yellowstone"],                              # Yellowstone only
}

# Which archetypes are valid for which species.
# Prevents impossible combos like "denning_season" for elk or "pack_formation" for mountain lion.
ARCHETYPE_SPECIES_COMPATIBILITY = {
    "stable_monitoring":      SPECIES_POOL,  # all species
    "dispersal_drama":        ["gray_wolf", "mountain_lion", "wolverine", "coyote", "canada_lynx",
                               "mule_deer", "elk", "whitetail_deer"],
    "denning_season":         ["gray_wolf", "coyote", "red_fox", "canada_lynx", "grizzly_bear",
                               "black_bear", "wolverine", "mountain_lion"],  # species with dens
    "harsh_winter":           SPECIES_POOL,  # affects all
    "territorial_conflict":   ["gray_wolf", "mountain_lion", "wolverine", "coyote"],  # territorial species
    "prey_migration":         ["gray_wolf", "mountain_lion", "coyote",  # predators that follow prey
                               "elk", "mule_deer", "pronghorn", "bighorn_sheep", "bison"],  # migratory prey
    "disease_outbreak":       ["gray_wolf", "bighorn_sheep", "elk", "mule_deer", "moose",
                               "whitetail_deer", "bison"],  # mange, CWD, pneumonia, brucellosis
    "reintroduction_study":   ["gray_wolf", "wolverine", "canada_lynx", "bighorn_sheep", "bison"],
    "mortality_investigation":SPECIES_POOL,  # any species
    "pack_formation":         ["gray_wolf", "coyote"],  # pack-forming species only
}
```

### 12.2 Auto-Config Function

When `seed` is provided, all other fields are auto-sampled deterministically:

```python
def auto_config(num_keys, num_updates, condition, seed):
    rng = Random(seed)

    # 1. Pick study area first (constrains species)
    study_area = rng.choice(["northern_yellowstone", "northern_rockies_generic", "great_lakes_generic"])

    # 2. Pick species mode
    species_mode = rng.choices(["same", "mixed"], weights=[60, 40])[0]

    # 3. Pick species, filtered by study area compatibility
    valid_species = [sp for sp in SPECIES_POOL
                     if study_area in SPECIES_STUDY_AREA_COMPATIBILITY[sp]]

    if species_mode == "same":
        species = [rng.choice(valid_species)]
    else:
        # Mixed: pick 2-3 species that all co-occur in this study area
        n_species = rng.choice([2, 3])
        species = rng.sample(valid_species, min(n_species, len(valid_species)))

    # 4. Pick archetype, filtered by species compatibility
    valid_archetypes = [a for a in ARCHETYPE_SPECIES_COMPATIBILITY
                        if any(sp in ARCHETYPE_SPECIES_COMPATIBILITY[a] for sp in species)]
    study_archetype = rng.choice(valid_archetypes)

    # 5. Pick tracked attribute, weighted by interference_quality
    # Sample from the FULL pool (all 21 attributes), weighted by quality
    attr_names = [a["name"] for a in TRACKABLE_ATTRIBUTES]
    attr_weights = [INTERFERENCE_QUALITY_WEIGHTS[a["interference_quality"]]
                    for a in TRACKABLE_ATTRIBUTES]

    # Filter out species-incompatible attributes
    for i, attr_def in enumerate(TRACKABLE_ATTRIBUTES):
        applicable = attr_def["species_applicable"]
        if applicable != "all":
            if not any(sp in applicable for sp in species):
                attr_weights[i] = 0  # species can't use this attribute

    tracked_attribute = rng.choices(attr_names, weights=attr_weights, k=1)[0]

    # 6. Validate: ensure the combination is coherent
    assert all(study_area in SPECIES_STUDY_AREA_COMPATIBILITY[sp] for sp in species), \
        f"Species {species} not compatible with study area {study_area}"
    assert any(sp in ARCHETYPE_SPECIES_COMPATIBILITY[study_archetype] for sp in species), \
        f"Archetype {study_archetype} not compatible with species {species}"

    return WildlifeTrialConfig(
        num_keys=num_keys,
        num_updates=num_updates,
        condition=condition,
        seed=seed,
        tracked_attribute=tracked_attribute,
        attribute_mode=rng.choice(["same", "mixed"]),
        species_mode=species_mode,
        species=species,
        study_area=study_area,
        season_start=rng.choice(["winter", "spring", "summer", "fall"]),
        filler_budget=rng.choice(["minimal", "light", "medium", "heavy"]),
        voice=rng.choice(["field_report", "research_narrative"]),
        study_archetype=study_archetype,
        queried_animal_idx=rng.randint(0, num_keys - 1),
    )
```
