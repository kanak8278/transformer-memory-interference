"""Wildlife tracking narrative interference trial generator.

Implements the full generation flow:
- GPS-collared animal selection (species, sex, age, pack/group)
- Event sequence generation with season-aware weights
- State tracking for all animals
- Narrative rendering with field-report and narrative-style templates
- Question generation for RI/PI conditions
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from ..base import NarrativeTrialGenerator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEASON_MONTHS = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "fall": [9, 10, 11],
}

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

MONTH_ABBR = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

# Event weights per season
EVENT_WEIGHTS = {
    "winter": {
        "relocation_observation": 30,
        "visual_observation": 8,
        "capture_workup": 2,
        "kill_site_investigation": 20,
        "pack_interaction": 5,
        "dispersal_event": 2,
        "denning_event": 0,
        "mortality_event": 8,
        "seasonal_shift": 5,
        "collar_event": 5,
        "territorial_event": 8,
        "health_change": 7,
    },
    "spring": {
        "relocation_observation": 20,
        "visual_observation": 15,
        "capture_workup": 8,
        "kill_site_investigation": 12,
        "pack_interaction": 5,
        "dispersal_event": 5,
        "denning_event": 15,
        "mortality_event": 3,
        "seasonal_shift": 5,
        "collar_event": 4,
        "territorial_event": 4,
        "health_change": 4,
    },
    "summer": {
        "relocation_observation": 22,
        "visual_observation": 18,
        "capture_workup": 10,
        "kill_site_investigation": 10,
        "pack_interaction": 5,
        "dispersal_event": 10,
        "denning_event": 5,
        "mortality_event": 2,
        "seasonal_shift": 3,
        "collar_event": 5,
        "territorial_event": 5,
        "health_change": 5,
    },
    "fall": {
        "relocation_observation": 22,
        "visual_observation": 15,
        "capture_workup": 5,
        "kill_site_investigation": 15,
        "pack_interaction": 8,
        "dispersal_event": 8,
        "denning_event": 0,
        "mortality_event": 5,
        "seasonal_shift": 8,
        "collar_event": 4,
        "territorial_event": 10,
        "health_change": 0,
    },
}

# Safe tracked attributes (ones that appear explicitly in narrative text and change)
SAFE_TRACKED_ATTRIBUTES = [
    "weight_kg",       # changes every observation, wide range, high interference quality
    "location_zone",   # changes every relocation, large categorical pool
    "daily_movement_km",  # changes every observation, numeric
    "activity_state",  # large categorical pool (22 values)
    "health_status",   # large categorical pool (20 values)
    "habitat_type",    # large categorical pool (22 values)
]

# Extended attributes for larger num_updates (7+) where slow-changing attributes
# have time to produce multiple distinct values
EXTENDED_TRACKED_ATTRIBUTES = SAFE_TRACKED_ATTRIBUTES + [
    "body_condition_score",   # slow-changing (1-9 scale), needs 5+ events to show variance
    "elevation_m",            # moderate change rate, fine for 5+ updates
    "collar_battery_pct",     # slowly decreasing only — good for demonstrating PI but
                              # not RI since first=highest always. Use only with 7+ updates.
]

# Monthly weight trajectories: {month: (lo, hi)} by species/sex key
# Values represent typical weight ranges in kg
SPECIES_TRAJECTORY_CURVES = {
    "gray_wolf_M": {
        1: (42, 50), 2: (40, 48), 3: (42, 50), 4: (44, 52),
        5: (46, 54), 6: (48, 56), 7: (50, 58), 8: (50, 58),
        9: (48, 56), 10: (50, 58), 11: (48, 55), 12: (45, 52),
    },
    "gray_wolf_F": {
        1: (34, 42), 2: (32, 40), 3: (34, 42), 4: (36, 44),
        5: (34, 42), 6: (36, 44), 7: (40, 48), 8: (40, 48),
        9: (38, 46), 10: (40, 48), 11: (38, 45), 12: (36, 43),
    },
    "elk_F": {  # cow elk
        1: (200, 230), 2: (190, 220), 3: (195, 228), 4: (200, 235),
        5: (195, 230), 6: (195, 225), 7: (210, 248), 8: (222, 258),
        9: (225, 264), 10: (220, 260), 11: (215, 252), 12: (206, 242),
    },
    "elk_M": {  # bull elk
        1: (260, 310), 2: (255, 300), 3: (260, 308), 4: (270, 318),
        5: (280, 328), 6: (290, 340), 7: (295, 345), 8: (300, 350),
        9: (245, 292), 10: (230, 278), 11: (240, 290), 12: (258, 308),
    },
    "grizzly_bear_M": {
        # Hibernation Nov–Mar; only Apr–Oct active
        4: (136, 180), 5: (150, 200), 6: (164, 220), 7: (178, 238),
        8: (192, 258), 9: (200, 270), 10: (208, 280),
    },
    "grizzly_bear_F": {
        4: (100, 140), 5: (110, 155), 6: (118, 168), 7: (128, 180),
        8: (138, 192), 9: (145, 200), 10: (150, 205),
    },
    "mountain_lion_M": {
        1: (62, 90), 2: (60, 88), 3: (61, 89), 4: (63, 91),
        5: (64, 92), 6: (65, 93), 7: (65, 95), 8: (65, 95),
        9: (64, 93), 10: (63, 92), 11: (62, 90), 12: (62, 90),
    },
    "mountain_lion_F": {
        1: (34, 58), 2: (33, 57), 3: (34, 58), 4: (35, 60),
        5: (35, 62), 6: (36, 63), 7: (36, 64), 8: (36, 64),
        9: (35, 62), 10: (35, 60), 11: (34, 59), 12: (34, 58),
    },
    "pronghorn_M": {
        1: (42, 62), 2: (41, 60), 3: (42, 62), 4: (44, 64),
        5: (45, 65), 6: (46, 65), 7: (46, 65), 8: (45, 64),
        9: (38, 58), 10: (39, 60), 11: (41, 62), 12: (42, 62),
    },
    "pronghorn_F": {
        1: (36, 48), 2: (35, 46), 3: (36, 48), 4: (37, 50),
        5: (37, 50), 6: (38, 50), 7: (39, 50), 8: (39, 50),
        9: (37, 49), 10: (38, 50), 11: (37, 49), 12: (36, 48),
    },
}

# BCS ranges by season (1-9 scale)
BCS_BY_SEASON = {
    "winter": (3, 6),
    "spring": (3, 7),
    "summer": (5, 8),
    "fall": (5, 9),
}

# Typical daily movement ranges (km) by season and species
DAILY_MOVEMENT_BY_SEASON = {
    "gray_wolf": {
        "winter": (10, 35), "spring": (8, 25), "summer": (6, 20), "fall": (10, 30),
    },
    "elk": {
        "winter": (2, 10), "spring": (4, 15), "summer": (3, 12), "fall": (5, 18),
    },
    "grizzly_bear": {
        "winter": (0, 0), "spring": (5, 20), "summer": (8, 25), "fall": (10, 30),
    },
    "mountain_lion": {
        "winter": (5, 20), "spring": (5, 18), "summer": (4, 15), "fall": (6, 22),
    },
    "pronghorn": {
        "winter": (3, 12), "spring": (5, 18), "summer": (4, 15), "fall": (6, 20),
    },
}

# Elevation ranges (m) by species and season
ELEVATION_BY_SEASON = {
    "gray_wolf": {
        "winter": (1400, 2100), "spring": (1600, 2400), "summer": (1800, 2600), "fall": (1500, 2400),
    },
    "elk": {
        "winter": (1400, 1900), "spring": (1600, 2400), "summer": (2000, 2800), "fall": (1700, 2500),
    },
    "grizzly_bear": {
        "winter": (1800, 2400), "spring": (1500, 2200), "summer": (2000, 2800), "fall": (1900, 2700),
    },
    "mountain_lion": {
        "winter": (1300, 2000), "spring": (1500, 2300), "summer": (1700, 2600), "fall": (1500, 2400),
    },
    "pronghorn": {
        "winter": (1400, 1800), "spring": (1500, 2000), "summer": (1600, 2200), "fall": (1500, 2000),
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AnimalState:
    collar_id: str
    species: str
    sex: str               # "M" or "F"
    age_years: float
    pack_name: Optional[str]     # None for solitary species
    individual_name: Optional[str]
    trajectory_type: str   # "stable", "improving", "declining", "dispersing"
    # Tracked attributes
    weight_kg: float
    body_condition_score: int
    location_zone: str
    activity_state: str
    health_status: str
    pack_membership: str
    reproductive_status: str
    habitat_type: str
    collar_battery_pct: float
    daily_movement_km: float
    elevation_m: int
    # Study metadata
    study_day: int         # days since study start
    study_month: int       # 1-12
    last_capture_day: Optional[int]
    alive: bool = True
    dispersed: bool = False

    def get_attribute(self, attr_name: str):
        return getattr(self, attr_name, None)

    def format_attribute(self, attr_name: str) -> str:
        """Return a canonical string suitable for both narrative text and entity_tracking."""
        val = self.get_attribute(attr_name)
        if val is None:
            return "unknown"
        if attr_name == "weight_kg":
            return f"{val:.1f} kg"
        if attr_name == "body_condition_score":
            return f"{val}/9"
        if attr_name == "collar_battery_pct":
            return f"{int(val)}%"
        if attr_name == "daily_movement_km":
            return f"{val:.1f} km"
        if attr_name == "elevation_m":
            return f"{int(val)} m"
        return str(val)


@dataclass
class WildlifeTrialConfig:
    num_keys: int
    num_updates: int
    condition: str
    seed: int
    species: str                  # "gray_wolf", "elk", etc.
    tracked_attribute: str
    filler_budget: str            # "minimal","light","medium","heavy"
    voice: str                    # "field" or "narrative"
    study_area_key: str
    queried_animal_idx: int
    start_month: int              # 1–12
    study_duration_days: int      # total study window in days


# ---------------------------------------------------------------------------
# Filler config
# ---------------------------------------------------------------------------

FILLER_CONFIG = {
    "minimal": {"insert_probability": 0.05, "sentences_per_insertion": 1},
    "light":   {"insert_probability": 0.25, "sentences_per_insertion": 1},
    "medium":  {"insert_probability": 0.45, "sentences_per_insertion": (1, 2)},
    "heavy":   {"insert_probability": 0.65, "sentences_per_insertion": (1, 3)},
}


# ---------------------------------------------------------------------------
# Main Generator
# ---------------------------------------------------------------------------

class WildlifeTrialGenerator(NarrativeTrialGenerator):

    def __init__(self):
        data_dir = Path(__file__).parent / "data"
        with open(data_dir / "wildlife_templates.json") as f:
            self.templates = json.load(f)
        with open(data_dir / "wildlife_data.json") as f:
            self.data = json.load(f)

        self.study_areas = self.data["study_areas"]
        self.pack_names = self.data["pack_names"]
        self.elk_herd_names = self.data["elk_herd_names"]
        self.project_names = self.data["project_names"]
        self.wolf_names = self.data["wolf_names"]
        self.species_info = self.data["species_info"]
        self.directions = self.data["directions"]
        self.cause_of_death = self.data["cause_of_death"]

    # -------------------------------------------------------------------
    # Auto-config
    # -------------------------------------------------------------------

    # Species compatible with each study area (ecological realism)
    SPECIES_STUDY_AREA_COMPAT = {
        "northern_yellowstone": ["gray_wolf", "elk", "grizzly_bear", "mountain_lion", "pronghorn"],
        "bitterroot_selway":    ["gray_wolf", "elk", "grizzly_bear", "mountain_lion"],
        "isle_royale":          ["gray_wolf"],  # Isle Royale only has wolves (predator) + moose; no elk/grizzly
    }

    def _auto_config(self, num_keys: int, num_updates: int,
                     condition: str, seed: int,
                     rng: random.Random) -> WildlifeTrialConfig:
        # Pick study area first, then compatible species
        study_area_key = rng.choice(list(self.study_areas.keys()))
        valid_species = self.SPECIES_STUDY_AREA_COMPAT.get(study_area_key,
                                                            list(self.species_info.keys()))
        species = rng.choice(valid_species)
        # Use extended pool only for larger num_updates where slow attrs can diverge
        attr_pool = EXTENDED_TRACKED_ATTRIBUTES if num_updates >= 7 else SAFE_TRACKED_ATTRIBUTES
        tracked_attribute = rng.choice(attr_pool)
        filler_budget = rng.choice(["minimal", "light", "medium", "heavy"])
        voice = rng.choice(["field", "narrative"])
        queried_animal_idx = rng.randint(0, num_keys - 1)
        # Grizzly bears are hibernating Nov-Mar; exclude winter start months to
        # ensure enough diverse events can be generated
        if species == "grizzly_bear":
            valid_months = [4, 5, 6, 7, 8, 9, 10]  # active season only
        else:
            valid_months = list(range(1, 13))
        start_month = rng.choice(valid_months)
        study_duration_days = rng.choice([60, 90, 120, 180, 240, 365])
        # For grizzly, ensure study doesn't run into hibernation at the start
        if species == "grizzly_bear" and num_updates < 10:
            study_duration_days = min(study_duration_days, 120)  # short study in active season

        return WildlifeTrialConfig(
            num_keys=num_keys,
            num_updates=num_updates,
            condition=condition,
            seed=seed,
            species=species,
            tracked_attribute=tracked_attribute,
            filler_budget=filler_budget,
            voice=voice,
            study_area_key=study_area_key,
            queried_animal_idx=queried_animal_idx,
            start_month=start_month,
            study_duration_days=study_duration_days,
        )

    # -------------------------------------------------------------------
    # Season helpers
    # -------------------------------------------------------------------

    def _get_season(self, month: int) -> str:
        for season, months in SEASON_MONTHS.items():
            if month in months:
                return season
        return "winter"

    def _month_from_day(self, start_month: int, study_day: int) -> int:
        """Approximate month given study start month and day offset."""
        total_month = start_month + study_day // 30
        return ((total_month - 1) % 12) + 1

    def _format_date(self, start_month: int, study_day: int, rng: random.Random) -> str:
        month = self._month_from_day(start_month, study_day)
        day = min(28, max(1, (study_day % 30) + 1 + rng.randint(0, 2)))
        year_offset = (start_month + study_day // 30 - 1) // 12
        base_year = 2022 + year_offset
        return f"{MONTH_ABBR[month]} {day}, {base_year}"

    # -------------------------------------------------------------------
    # Animal state initialization
    # -------------------------------------------------------------------

    def _get_weight_range(self, species: str, sex: str, month: int) -> Tuple[float, float]:
        key = f"{species}_{sex}"
        curve = SPECIES_TRAJECTORY_CURVES.get(key)
        if curve is None:
            # Fallback: use any available key for the species
            for k in SPECIES_TRAJECTORY_CURVES:
                if k.startswith(species):
                    curve = SPECIES_TRAJECTORY_CURVES[k]
                    break
        if curve is None:
            return (40.0, 60.0)

        months = sorted(curve.keys())
        if month in curve:
            return curve[month]
        # Interpolate
        if month < months[0]:
            return curve[months[0]]
        if month > months[-1]:
            return curve[months[-1]]
        for i in range(len(months) - 1):
            m0, m1 = months[i], months[i + 1]
            if m0 <= month <= m1:
                lo0, hi0 = curve[m0]
                lo1, hi1 = curve[m1]
                frac = (month - m0) / max(1, m1 - m0)
                lo = lo0 + (lo1 - lo0) * frac
                hi = hi0 + (hi1 - hi0) * frac
                return (lo, hi)
        return curve[months[-1]]

    def _init_animal_state(self, collar_id: str, species: str, sex: str,
                           start_month: int, pack_name: Optional[str],
                           individual_name: Optional[str],
                           zone: str, rng: random.Random) -> AnimalState:
        season = self._get_season(start_month)
        lo, hi = self._get_weight_range(species, sex, start_month)
        weight = round(rng.uniform(lo, hi), 1)

        bcs_lo, bcs_hi = BCS_BY_SEASON[season]
        bcs = rng.randint(bcs_lo, bcs_hi)

        move_lo, move_hi = DAILY_MOVEMENT_BY_SEASON.get(species, {}).get(season, (5, 20))
        # Grizzly in winter = hibernating
        if species == "grizzly_bear" and season == "winter":
            daily_km = 0.0
        else:
            daily_km = round(rng.uniform(move_lo, move_hi), 1)

        elev_lo, elev_hi = ELEVATION_BY_SEASON.get(species, {}).get(season, (1500, 2200))
        elevation = rng.randint(elev_lo, elev_hi)

        collar_battery = round(rng.uniform(55, 100), 0)

        trajectory_type = rng.choice(["stable", "improving", "declining", "dispersing"])

        sp_info = self.species_info.get(species, {})
        health_conditions = sp_info.get("health_conditions", ["normal condition"])
        activity_options = sp_info.get("seasonal_behaviors", {}).get(season,
                                      ["moving through territory"])
        activity = rng.choice(activity_options) if activity_options else "moving through territory"
        health_status = rng.choice(health_conditions)

        social_values = sp_info.get("social_values", {})
        social_opt = rng.choice(list(social_values.values())) if social_values else "unknown"
        pack_membership = social_opt

        # Reproductive status
        repro_options = {
            "gray_wolf": {
                "M": ["non-breeding adult", "breeding male", "subordinate male"],
                "F": ["non-breeding female", "breeding female", "nursing female", "pre-estrus female"],
            },
            "elk": {
                "M": ["spike bull", "raghorn bull", "mature bull"],
                "F": ["cow, no calf", "cow with calf", "yearling heifer"],
            },
            "grizzly_bear": {
                "M": ["adult solitary male", "subadult male"],
                "F": ["female with cubs", "female with yearlings", "solitary female"],
            },
            "mountain_lion": {
                "M": ["adult territorial male", "subadult male"],
                "F": ["female with kittens", "solitary female"],
            },
            "pronghorn": {
                "M": ["adult buck", "yearling buck"],
                "F": ["doe with fawn", "dry doe", "yearling doe"],
            },
        }
        opts = repro_options.get(species, {"M": ["adult"], "F": ["adult"]}).get(sex, ["adult"])
        reproductive_status = rng.choice(opts)

        habitat_options = {
            "gray_wolf": ["conifer forest", "open meadow", "riparian corridor", "alpine tundra", "shrubland"],
            "elk": ["open grassland", "aspen grove", "conifer forest edge", "riparian meadow", "alpine meadow"],
            "grizzly_bear": ["whitebark pine forest", "open slope", "riparian corridor", "subalpine meadow", "talus slope"],
            "mountain_lion": ["rocky canyon", "dense conifer", "mixed forest", "cliff band", "brushy draw"],
            "pronghorn": ["sagebrush flat", "open grassland", "shortgrass prairie", "shrubsteppe", "valley floor"],
        }
        habitat_type = rng.choice(habitat_options.get(species, ["mixed habitat"]))

        age = round(rng.uniform(1.5, 9.0), 1)

        return AnimalState(
            collar_id=collar_id,
            species=species,
            sex=sex,
            age_years=age,
            pack_name=pack_name,
            individual_name=individual_name,
            trajectory_type=trajectory_type,
            weight_kg=weight,
            body_condition_score=bcs,
            location_zone=zone,
            activity_state=activity,
            health_status=health_status,
            pack_membership=pack_membership,
            reproductive_status=reproductive_status,
            habitat_type=habitat_type,
            collar_battery_pct=collar_battery,
            daily_movement_km=daily_km,
            elevation_m=elevation,
            study_day=0,
            study_month=start_month,
            last_capture_day=None,
            alive=True,
            dispersed=False,
        )

    # -------------------------------------------------------------------
    # Event weights
    # -------------------------------------------------------------------

    def _get_event_weights(self, season: str, animal: AnimalState) -> dict:
        base = dict(EVENT_WEIGHTS[season])

        # Filter impossible events
        if animal.species == "grizzly_bear" and season == "winter":
            # Grizzly in winter = denning; most events impossible
            return {"denning_event": 100}

        if animal.species in ("grizzly_bear", "mountain_lion", "pronghorn"):
            base["pack_interaction"] = 0

        if animal.species in ("elk", "pronghorn"):
            base["kill_site_investigation"] = 0

        if animal.dispersed:
            base["pack_interaction"] = 0
            base["denning_event"] = 0

        if not animal.alive:
            return {}

        # Remove zero-weight events
        return {k: v for k, v in base.items() if v > 0}

    # -------------------------------------------------------------------
    # Day / time helpers
    # -------------------------------------------------------------------

    def _advance_day(self, animal: AnimalState, days: int, start_month: int):
        animal.study_day += days
        animal.study_month = self._month_from_day(start_month, animal.study_day)

    # -------------------------------------------------------------------
    # Event handlers — each returns (event_data dict, state_changes list)
    # state_changes: list of (collar_id, attr_name, old_val_str, new_val_str, mentioned: bool)
    # -------------------------------------------------------------------

    def _get_adjacent_zone(self, zone: str, study_area_key: str, rng: random.Random) -> str:
        adj = self.study_areas[study_area_key]["zone_adjacency"].get(zone, [])
        if adj:
            return rng.choice(adj)
        # Fallback: any zone in the study area
        zones = self.study_areas[study_area_key]["zones"]
        return rng.choice([z for z in zones if z != zone] or zones)

    def _get_pronoun(self, sex: str) -> Tuple[str, str, str]:
        """Returns (subject, object, possessive)."""
        if sex == "M":
            return ("he", "him", "his")
        else:
            return ("she", "her", "her")

    def _handle_relocation(self, collar_id: str,
                           animals: Dict[str, AnimalState],
                           study_area_key: str,
                           start_month: int,
                           rng: random.Random,
                           tracked: Dict[str, str]) -> Tuple[dict, list]:
        an = animals[collar_id]
        attr_name = tracked[collar_id]
        old_val_str = an.format_attribute(attr_name)

        prev_zone = an.location_zone
        new_zone = self._get_adjacent_zone(prev_zone, study_area_key, rng)
        an.location_zone = new_zone

        # Update movement
        season = self._get_season(an.study_month)
        move_lo, move_hi = DAILY_MOVEMENT_BY_SEASON.get(an.species, {}).get(season, (5, 15))
        an.daily_movement_km = round(rng.uniform(move_lo, move_hi), 1)

        # Slight weight drift
        drift = rng.uniform(-0.5, 0.5)
        lo, hi = self._get_weight_range(an.species, an.sex, an.study_month)
        an.weight_kg = round(max(lo * 0.85, min(hi * 1.05, an.weight_kg + drift)), 1)

        # Update elevation
        elev_lo, elev_hi = ELEVATION_BY_SEASON.get(an.species, {}).get(season, (1500, 2200))
        an.elevation_m = rng.randint(elev_lo, elev_hi)

        # Update activity_state — moving animals change activity
        sp_info = self.species_info.get(an.species, {})
        activity_options = sp_info.get("seasonal_behaviors", {}).get(season, [])
        if activity_options and rng.random() < 0.7:  # 70% chance of activity change on movement
            an.activity_state = rng.choice(activity_options)

        # Update habitat_type based on new zone's typical habitat
        zone_habitat_map = {
            "alpine meadow": "alpine meadow (above treeline)",
            "ridge": "subalpine meadow (near treeline)",
            "mountain": "spruce-fir forest (dense canopy)",
            "valley": "open grassland (valley floor)",
            "riparian": "riparian cottonwood gallery",
            "drainage": "lodgepole pine forest (dense)",
            "plateau": "subalpine meadow (near treeline)",
            "basin": "sagebrush-grassland mosaic",
            "meadow": "subalpine meadow (near treeline)",
            "corridor": "lodgepole pine forest (open/burned)",
            "pass": "alpine meadow (above treeline)",
            "bay": "wetland/marsh (sedge meadow)",
        }
        # Infer habitat from zone name keywords
        zone_lower = new_zone.lower()
        for keyword, habitat in zone_habitat_map.items():
            if keyword in zone_lower:
                an.habitat_type = habitat
                break

        distance_km = round(rng.uniform(2.0, 18.0), 1)
        direction = rng.choice(self.directions)

        date_str = self._format_date(start_month, an.study_day, rng)
        pron_sub, pron_obj, pron_pos = self._get_pronoun(an.sex)
        filler = self._render_filler_sentence(collar_id, animals, study_area_key, rng)
        attr_label = attr_name.replace("_", " ")

        new_val_str = an.format_attribute(attr_name)

        # Ensure value changed for rapidly-changing attributes
        if str(new_val_str) == str(old_val_str) and attr_name in ("weight_kg", "daily_movement_km"):
            an.weight_kg = round(an.weight_kg + rng.choice([-0.8, 0.8, 1.0, -1.2]), 1)
            new_val_str = an.format_attribute(attr_name)

        # For categorical attributes, force a change if still same
        if str(new_val_str) == str(old_val_str) and attr_name == "activity_state" and activity_options:
            different = [a for a in activity_options if a != old_val_str]
            if different:
                an.activity_state = rng.choice(different)
                new_val_str = an.format_attribute(attr_name)

        if str(new_val_str) == str(old_val_str) and attr_name == "habitat_type":
            habitat_pool = [
                "alpine meadow (above treeline)", "subalpine meadow (near treeline)",
                "lodgepole pine forest (dense)", "spruce-fir forest (dense canopy)",
                "sagebrush-grassland mosaic", "open grassland (valley floor)",
                "riparian cottonwood gallery", "dense willow thicket (riparian)"
            ]
            different = [h for h in habitat_pool if h != old_val_str]
            if different:
                an.habitat_type = rng.choice(different)
                new_val_str = an.format_attribute(attr_name)

        if str(new_val_str) == str(old_val_str) and attr_name == "health_status":
            sp_info_local = self.species_info.get(an.species, {})
            conditions = sp_info_local.get("health_conditions", [])
            different = [h for h in conditions if h != old_val_str]
            if different:
                an.health_status = rng.choice(different)
                new_val_str = an.format_attribute(attr_name)

        event_data = {
            "date": date_str,
            "animal_id": collar_id,
            "zone": new_zone,
            "prev_zone": prev_zone,
            "distance_km": distance_km,
            "direction": direction,
            "attr_label": attr_label,
            "attr_value": an.format_attribute(attr_name),
            "pron_sub": pron_sub,
            "pron_obj": pron_obj,
            "pron_pos": pron_pos,
            "filler": filler,
        }

        state_changes = [
            (collar_id, attr_name, old_val_str, an.format_attribute(attr_name), True),
        ]
        return event_data, state_changes

    def _handle_visual_observation(self, collar_id: str,
                                   animals: Dict[str, AnimalState],
                                   study_area_key: str,
                                   start_month: int,
                                   rng: random.Random,
                                   tracked: Dict[str, str]) -> Tuple[dict, list]:
        # Reuse relocation logic but use narrative templates
        return self._handle_relocation(collar_id, animals, study_area_key,
                                       start_month, rng, tracked)

    def _handle_capture_workup(self, collar_id: str,
                                animals: Dict[str, AnimalState],
                                study_area_key: str,
                                start_month: int,
                                rng: random.Random,
                                tracked: Dict[str, str]) -> Tuple[dict, list]:
        an = animals[collar_id]
        attr_name = tracked[collar_id]
        old_val_str = an.format_attribute(attr_name)

        season = self._get_season(an.study_month)
        lo, hi = self._get_weight_range(an.species, an.sex, an.study_month)
        # Capture gives precise weight
        an.weight_kg = round(rng.uniform(lo, hi), 1)
        bcs_lo, bcs_hi = BCS_BY_SEASON[season]
        an.body_condition_score = rng.randint(bcs_lo, bcs_hi)
        # Service collar battery
        an.collar_battery_pct = round(rng.uniform(60, 100), 0)
        an.last_capture_day = an.study_day

        sp_info = self.species_info.get(an.species, {})
        methods = self.templates["filler_variable_pools"]["capture_methods"]
        method = rng.choice(methods)

        filler = self._render_filler_sentence(collar_id, animals, study_area_key, rng)
        date_str = self._format_date(start_month, an.study_day, rng)
        pron_sub, pron_obj, pron_pos = self._get_pronoun(an.sex)
        attr_label = attr_name.replace("_", " ")
        new_val_str = an.format_attribute(attr_name)

        if new_val_str == old_val_str and attr_name in ("weight_kg",):
            an.weight_kg = round(an.weight_kg + rng.choice([0.5, 1.0, 1.5, -0.5, -1.0]), 1)
            new_val_str = an.format_attribute(attr_name)

        event_data = {
            "date": date_str,
            "animal_id": collar_id,
            "zone": an.location_zone,
            "method": method,
            "attr_label": attr_label,
            "attr_value": an.format_attribute(attr_name),
            "bcs": an.body_condition_score,
            "battery": int(an.collar_battery_pct),
            "pron_sub": pron_sub,
            "pron_obj": pron_obj,
            "pron_pos": pron_pos,
            "filler": filler,
        }

        state_changes = [
            (collar_id, attr_name, old_val_str, an.format_attribute(attr_name), True),
        ]
        return event_data, state_changes

    def _handle_kill_site(self, collar_id: str,
                          animals: Dict[str, AnimalState],
                          study_area_key: str,
                          start_month: int,
                          rng: random.Random,
                          tracked: Dict[str, str]) -> Tuple[dict, list]:
        an = animals[collar_id]
        attr_name = tracked[collar_id]
        old_val_str = an.format_attribute(attr_name)

        # Move to adjacent zone (kill site)
        kill_zone = self._get_adjacent_zone(an.location_zone, study_area_key, rng)
        an.location_zone = kill_zone
        an.activity_state = "feeding at kill"

        # Weight gain from feeding
        an.weight_kg = round(min(
            self._get_weight_range(an.species, an.sex, an.study_month)[1] * 1.05,
            an.weight_kg + rng.uniform(0.5, 3.0)
        ), 1)

        sp_info = self.species_info.get(an.species, {})
        prey_species_list = sp_info.get("prey_species", ["ungulate"])
        prey_ages_list = sp_info.get("prey_ages", ["1-2 days"])
        # Isle Royale wolves only hunt moose — override prey pool if needed
        if study_area_key == "isle_royale" and an.species == "gray_wolf":
            prey_species_list = ["moose calf", "adult moose", "moose cow-calf pair", "injured moose"]
        prey_species = rng.choice(prey_species_list) if prey_species_list else "ungulate"
        prey_age = rng.choice(prey_ages_list) if prey_ages_list else "unknown age"
        radius_km = round(rng.uniform(0.1, 0.5), 2)

        date_str = self._format_date(start_month, an.study_day, rng)
        attr_label = attr_name.replace("_", " ")
        new_val_str = an.format_attribute(attr_name)

        if new_val_str == old_val_str and attr_name == "weight_kg":
            an.weight_kg = round(an.weight_kg + rng.uniform(0.5, 1.5), 1)
            new_val_str = an.format_attribute(attr_name)

        event_data = {
            "date": date_str,
            "animal_id": collar_id,
            "zone": kill_zone,
            "prey_species": prey_species,
            "prey_age": prey_age,
            "radius_km": radius_km,
            "attr_label": attr_label,
            "attr_value": an.format_attribute(attr_name),
        }

        state_changes = [
            (collar_id, attr_name, old_val_str, an.format_attribute(attr_name), True),
        ]
        return event_data, state_changes

    def _handle_pack_interaction(self, collar_id: str,
                                  animals: Dict[str, AnimalState],
                                  study_area_key: str,
                                  start_month: int,
                                  rng: random.Random,
                                  tracked: Dict[str, str],
                                  all_animals: List[str]) -> Tuple[dict, list]:
        an = animals[collar_id]
        attr_name = tracked[collar_id]
        old_val_str = an.format_attribute(attr_name)

        # Pick another animal for interaction
        others = [c for c in all_animals if c != collar_id and animals[c].alive]
        if not others:
            others = [c for c in all_animals if c != collar_id]

        other_id = rng.choice(others) if others else collar_id
        other_an = animals[other_id]
        other_attr = tracked.get(other_id, attr_name)

        dist_m = rng.randint(20, 500)
        date_str = self._format_date(start_month, an.study_day, rng)
        attr_label = attr_name.replace("_", " ")

        # Minor weight change
        drift = rng.uniform(-0.3, 0.3)
        an.weight_kg = round(an.weight_kg + drift, 1)
        new_val_str = an.format_attribute(attr_name)

        if new_val_str == old_val_str:
            an.weight_kg = round(an.weight_kg + rng.choice([0.4, -0.4, 0.6, -0.6]), 1)
            new_val_str = an.format_attribute(attr_name)

        event_data = {
            "date": date_str,
            "animal_id_1": collar_id,
            "animal_id_2": other_id,
            "zone": an.location_zone,
            "dist_m": dist_m,
            "attr_label": attr_label,
            "attr_value_1": an.format_attribute(attr_name),
            "attr_value_2": other_an.format_attribute(other_attr),
        }

        state_changes = [
            (collar_id, attr_name, old_val_str, an.format_attribute(attr_name), True),
        ]
        # Note: other_id's value is in event_data as attr_value_2 but templates only render
        # attr_value_1 (the primary animal). Mark other_id as mentioned=False so it doesn't
        # appear in entity_tracking without a corresponding narrative text.
        if other_id in tracked:
            other_old = other_an.format_attribute(other_attr)
            state_changes.append(
                (other_id, other_attr, other_old, other_an.format_attribute(other_attr), False)
            )
        return event_data, state_changes

    def _handle_dispersal(self, collar_id: str,
                           animals: Dict[str, AnimalState],
                           study_area_key: str,
                           start_month: int,
                           rng: random.Random,
                           tracked: Dict[str, str]) -> Tuple[dict, list]:
        an = animals[collar_id]
        attr_name = tracked[collar_id]
        old_val_str = an.format_attribute(attr_name)

        an.dispersed = True
        an.pack_membership = "disperser — no pack affiliation"
        an.pack_name = None

        # Move far — pick a random zone
        zones = self.study_areas[study_area_key]["zones"]
        new_zone = rng.choice([z for z in zones if z != an.location_zone] or zones)
        an.location_zone = new_zone

        distance_km = round(rng.uniform(25.0, 120.0), 1)
        season = self._get_season(an.study_month)
        move_lo, move_hi = DAILY_MOVEMENT_BY_SEASON.get(an.species, {}).get(season, (10, 30))
        an.daily_movement_km = round(rng.uniform(move_lo * 1.3, move_hi * 1.5), 1)

        # Weight may drop during dispersal
        an.weight_kg = round(an.weight_kg - rng.uniform(0.5, 2.5), 1)
        an.weight_kg = max(an.weight_kg, 25.0)

        date_str = self._format_date(start_month, an.study_day, rng)
        attr_label = attr_name.replace("_", " ")
        new_val_str = an.format_attribute(attr_name)

        if new_val_str == old_val_str and attr_name in ("weight_kg", "daily_movement_km"):
            an.daily_movement_km = round(an.daily_movement_km + rng.uniform(1.0, 3.0), 1)
            new_val_str = an.format_attribute(attr_name)

        event_data = {
            "date": date_str,
            "animal_id": collar_id,
            "zone": new_zone,
            "distance_km": distance_km,
            "attr_label": attr_label,
            "attr_value": an.format_attribute(attr_name),
        }

        state_changes = [
            (collar_id, attr_name, old_val_str, an.format_attribute(attr_name), True),
        ]
        return event_data, state_changes

    def _handle_denning(self, collar_id: str,
                         animals: Dict[str, AnimalState],
                         study_area_key: str,
                         start_month: int,
                         rng: random.Random,
                         tracked: Dict[str, str]) -> Tuple[dict, list]:
        an = animals[collar_id]
        attr_name = tracked[collar_id]
        old_val_str = an.format_attribute(attr_name)

        # Minimal movement
        an.daily_movement_km = round(rng.uniform(0.1, 0.8), 1)
        an.activity_state = "denning — restricted movement"

        # Weight drops during nursing/denning
        an.weight_kg = round(an.weight_kg - rng.uniform(0.5, 2.0), 1)
        lo, _ = self._get_weight_range(an.species, an.sex, an.study_month)
        an.weight_kg = max(an.weight_kg, lo * 0.75)

        days = rng.randint(5, 21)
        date_str = self._format_date(start_month, an.study_day, rng)
        attr_label = attr_name.replace("_", " ")
        new_val_str = an.format_attribute(attr_name)

        if new_val_str == old_val_str:
            an.weight_kg = round(an.weight_kg - rng.uniform(0.3, 1.0), 1)
            new_val_str = an.format_attribute(attr_name)

        event_data = {
            "date": date_str,
            "animal_id": collar_id,
            "zone": an.location_zone,
            "days": days,
            "attr_label": attr_label,
            "attr_value": an.format_attribute(attr_name),
        }

        state_changes = [
            (collar_id, attr_name, old_val_str, an.format_attribute(attr_name), True),
        ]
        return event_data, state_changes

    def _handle_mortality(self, collar_id: str,
                           animals: Dict[str, AnimalState],
                           study_area_key: str,
                           start_month: int,
                           rng: random.Random,
                           tracked: Dict[str, str]) -> Tuple[dict, list]:
        an = animals[collar_id]
        attr_name = tracked[collar_id]
        old_val_str = an.format_attribute(attr_name)

        an.alive = False
        cause = rng.choice(self.cause_of_death)
        date_str = self._format_date(start_month, an.study_day, rng)
        attr_label = attr_name.replace("_", " ")

        event_data = {
            "date": date_str,
            "animal_id": collar_id,
            "zone": an.location_zone,
            "cause": cause,
            "attr_label": attr_label,
            "attr_value": an.format_attribute(attr_name),
        }

        # Mortality template includes {attr_value} — record the last known value
        state_changes = [
            (collar_id, attr_name, old_val_str, an.format_attribute(attr_name), True),
        ]
        return event_data, state_changes

    def _handle_health_change(self, collar_id: str,
                               animals: Dict[str, AnimalState],
                               study_area_key: str,
                               start_month: int,
                               rng: random.Random,
                               tracked: Dict[str, str]) -> Tuple[dict, list]:
        an = animals[collar_id]
        attr_name = tracked[collar_id]
        old_val_str = an.format_attribute(attr_name)

        sp_info = self.species_info.get(an.species, {})
        conditions = sp_info.get("health_conditions", ["altered gait observed"])
        condition = rng.choice(conditions)
        an.health_status = condition

        # Update BCS
        direction_sign = rng.choice([-1, -1, 1])
        season = self._get_season(an.study_month)
        bcs_lo, bcs_hi = BCS_BY_SEASON[season]
        an.body_condition_score = max(bcs_lo, min(bcs_hi, an.body_condition_score + direction_sign))

        # Adjust weight
        an.weight_kg = round(an.weight_kg + direction_sign * rng.uniform(0.5, 2.0), 1)
        lo, _ = self._get_weight_range(an.species, an.sex, an.study_month)
        an.weight_kg = max(an.weight_kg, lo * 0.70)

        date_str = self._format_date(start_month, an.study_day, rng)
        attr_label = attr_name.replace("_", " ")
        new_val_str = an.format_attribute(attr_name)

        if new_val_str == old_val_str and attr_name in ("weight_kg", "body_condition_score"):
            an.weight_kg = round(an.weight_kg + rng.choice([-0.8, 0.8, 1.2, -1.2]), 1)
            new_val_str = an.format_attribute(attr_name)

        event_data = {
            "date": date_str,
            "animal_id": collar_id,
            "zone": an.location_zone,
            "condition": condition,
            "bcs": an.body_condition_score,
            "attr_label": attr_label,
            "attr_value": an.format_attribute(attr_name),
        }

        state_changes = [
            (collar_id, attr_name, old_val_str, an.format_attribute(attr_name), True),
        ]
        return event_data, state_changes

    def _handle_seasonal_shift(self, collar_id: str,
                                animals: Dict[str, AnimalState],
                                study_area_key: str,
                                start_month: int,
                                rng: random.Random,
                                tracked: Dict[str, str]) -> Tuple[dict, list]:
        an = animals[collar_id]
        attr_name = tracked[collar_id]
        old_val_str = an.format_attribute(attr_name)

        season = self._get_season(an.study_month)
        sp_info = self.species_info.get(an.species, {})
        behaviors = sp_info.get("seasonal_behaviors", {}).get(season, ["adjusting to seasonal conditions"])
        seasonal_behavior = rng.choice(behaviors)
        an.activity_state = seasonal_behavior

        # Move to new zone (seasonal range shift)
        new_zone = self._get_adjacent_zone(an.location_zone, study_area_key, rng)
        an.location_zone = new_zone

        elev_lo, elev_hi = ELEVATION_BY_SEASON.get(an.species, {}).get(season, (1500, 2200))
        an.elevation_m = rng.randint(elev_lo, elev_hi)

        date_str = self._format_date(start_month, an.study_day, rng)
        attr_label = attr_name.replace("_", " ")
        new_val_str = an.format_attribute(attr_name)

        if new_val_str == old_val_str and attr_name in ("elevation_m",):
            an.elevation_m = rng.randint(elev_lo, elev_hi)
            new_val_str = an.format_attribute(attr_name)

        if new_val_str == old_val_str and attr_name in ("weight_kg",):
            an.weight_kg = round(an.weight_kg + rng.choice([-1.0, 1.0, 1.5, -0.5]), 1)
            new_val_str = an.format_attribute(attr_name)

        event_data = {
            "date": date_str,
            "animal_id": collar_id,
            "zone": new_zone,
            "seasonal_behavior": seasonal_behavior,
            "attr_label": attr_label,
            "attr_value": an.format_attribute(attr_name),
        }

        state_changes = [
            (collar_id, attr_name, old_val_str, an.format_attribute(attr_name), True),
        ]
        return event_data, state_changes

    def _handle_collar_event(self, collar_id: str,
                              animals: Dict[str, AnimalState],
                              study_area_key: str,
                              start_month: int,
                              rng: random.Random,
                              tracked: Dict[str, str]) -> Tuple[dict, list]:
        an = animals[collar_id]
        attr_name = tracked[collar_id]
        old_val_str = an.format_attribute(attr_name)

        # Battery drain or service
        if rng.random() < 0.3:
            # Service — battery up
            an.collar_battery_pct = round(rng.uniform(80, 100), 0)
        else:
            # Drain
            drain = rng.uniform(5, 20)
            an.collar_battery_pct = max(5.0, an.collar_battery_pct - drain)
            an.collar_battery_pct = round(an.collar_battery_pct, 0)

        new_val_str = an.format_attribute(attr_name)

        # If battery is the tracked attr, value changed; otherwise record as not mentioned
        mentioned = (attr_name == "collar_battery_pct")

        # For non-battery tracked attrs, force a small weight drift for tracking purposes
        if attr_name != "collar_battery_pct":
            an.weight_kg = round(an.weight_kg + rng.choice([-0.5, 0.5, -0.8, 0.8]), 1)
            new_val_str = an.format_attribute(attr_name)
            mentioned = True  # relay event does update the state

        date_str = self._format_date(start_month, an.study_day, rng)
        attr_label = attr_name.replace("_", " ")

        event_data = {
            "date": date_str,
            "animal_id": collar_id,
            "zone": an.location_zone,
            "attr_label": attr_label,
            "attr_value": an.format_attribute(attr_name),
            "battery": int(an.collar_battery_pct),
        }

        # Use relocation_field template as fallback rendering
        event_data["prev_zone"] = an.location_zone
        event_data["distance_km"] = 0.0
        event_data["direction"] = "N"
        event_data["filler"] = f"Collar battery at {int(an.collar_battery_pct)}%."

        state_changes = [
            (collar_id, attr_name, old_val_str, an.format_attribute(attr_name), mentioned),
        ]
        return event_data, state_changes

    def _handle_territorial_event(self, collar_id: str,
                                    animals: Dict[str, AnimalState],
                                    study_area_key: str,
                                    start_month: int,
                                    rng: random.Random,
                                    tracked: Dict[str, str]) -> Tuple[dict, list]:
        # Treat as relocation with increased movement
        an = animals[collar_id]
        attr_name = tracked[collar_id]
        old_val_str = an.format_attribute(attr_name)

        season = self._get_season(an.study_month)
        move_lo, move_hi = DAILY_MOVEMENT_BY_SEASON.get(an.species, {}).get(season, (5, 20))
        an.daily_movement_km = round(rng.uniform(move_lo * 1.2, move_hi * 1.4), 1)

        prev_zone = an.location_zone
        new_zone = self._get_adjacent_zone(prev_zone, study_area_key, rng)
        an.location_zone = new_zone

        distance_km = round(rng.uniform(5.0, 25.0), 1)
        direction = rng.choice(self.directions)

        # Weight can drop slightly from increased exertion
        an.weight_kg = round(an.weight_kg - rng.uniform(0.2, 1.0), 1)
        lo, _ = self._get_weight_range(an.species, an.sex, an.study_month)
        an.weight_kg = max(an.weight_kg, lo * 0.75)

        new_val_str = an.format_attribute(attr_name)
        if new_val_str == old_val_str and attr_name in ("weight_kg", "daily_movement_km"):
            an.daily_movement_km = round(an.daily_movement_km + rng.uniform(1.0, 3.0), 1)
            new_val_str = an.format_attribute(attr_name)

        filler = self._render_filler_sentence(collar_id, animals, study_area_key, rng)
        date_str = self._format_date(start_month, an.study_day, rng)
        pron_sub, pron_obj, pron_pos = self._get_pronoun(an.sex)
        attr_label = attr_name.replace("_", " ")

        event_data = {
            "date": date_str,
            "animal_id": collar_id,
            "zone": new_zone,
            "prev_zone": prev_zone,
            "distance_km": distance_km,
            "direction": direction,
            "attr_label": attr_label,
            "attr_value": an.format_attribute(attr_name),
            "pron_sub": pron_sub,
            "pron_obj": pron_obj,
            "pron_pos": pron_pos,
            "filler": filler,
        }

        state_changes = [
            (collar_id, attr_name, old_val_str, an.format_attribute(attr_name), True),
        ]
        return event_data, state_changes

    # -------------------------------------------------------------------
    # Filler sentence rendering
    # -------------------------------------------------------------------

    def _render_filler_sentence(self, collar_id: str,
                                 animals: Dict[str, AnimalState],
                                 study_area_key: str,
                                 rng: random.Random) -> str:
        an = animals[collar_id]
        pools = self.templates["filler_variable_pools"]
        zone = an.location_zone
        zones = self.study_areas[study_area_key]["zones"]
        other_zone = rng.choice([z for z in zones if z != zone] or [zone])

        data = {
            "zone": zone,
            "other_zone": other_zone,
            "animal_id": collar_id,
            "date": "this period",
            "snow_cm": rng.randint(5, 80),
            "snow_desc": rng.choice(pools["snow_descriptions"]),
            "camera_obs": rng.choice(pools["camera_observations"]),
            "weather": rng.choice(pools["weather_descriptions"]),
            "prey_density": rng.choice(pools["prey_density_descriptions"]),
            "elk_count": rng.randint(10, 200),
        }
        template = rng.choice(self.templates["filler_templates"])
        return self.render_template(template, data)

    # -------------------------------------------------------------------
    # Narrative rendering
    # -------------------------------------------------------------------

    def _get_templates_for_event(self, event_type: str, voice: str) -> list:
        voice_suffix = "_field" if voice == "field" else "_narrative"
        mapping = {
            "relocation_observation": ["relocation" + voice_suffix, "relocation_field"],
            "visual_observation": ["relocation_narrative", "relocation_field"],
            "capture_workup": ["capture_workup" + voice_suffix, "capture_workup_field"],
            "kill_site_investigation": ["kill_site_field"],
            "pack_interaction": ["pack_interaction_field"],
            "dispersal_event": ["dispersal_field"],
            "denning_event": ["denning_field"],
            "mortality_event": ["mortality_field"],
            "health_change": ["health_change_field"],
            "seasonal_shift": ["seasonal_shift_field"],
            "collar_event": ["relocation_field"],
            "territorial_event": ["relocation" + voice_suffix, "relocation_field"],
        }
        for key in mapping.get(event_type, ["relocation_field"]):
            if key in self.templates:
                return self.templates[key]
        return self.templates["relocation_field"]

    # -------------------------------------------------------------------
    # Collar ID generation
    # -------------------------------------------------------------------

    def _generate_collar_ids(self, species: str, num_animals: int,
                              rng: random.Random) -> List[Tuple[str, str]]:
        """Generate list of (collar_id, sex) pairs."""
        sp_info = self.species_info.get(species, {})
        code = sp_info.get("code", "A")
        sex_codes = sp_info.get("sex_codes", {"M": "AM", "F": "AF"})

        result = []
        used_numbers = set()
        for i in range(num_animals):
            sex = rng.choice(["M", "F"])
            sc = sex_codes.get(sex, code)
            while True:
                num = rng.randint(100, 999)
                if num not in used_numbers:
                    used_numbers.add(num)
                    break
            collar_id = f"{sc}{num}"
            result.append((collar_id, sex))
        return result

    def _assign_pack_names(self, collar_ids: List[Tuple[str, str]],
                           species: str, rng: random.Random) -> Dict[str, Optional[str]]:
        sp_info = self.species_info.get(species, {})
        social_unit = sp_info.get("social_unit", "solitary")
        pack_assignments = {}

        if social_unit == "pack":
            available_packs = list(self.pack_names)
            rng.shuffle(available_packs)
            n_packs = max(1, len(collar_ids) // 3)
            packs_used = available_packs[:n_packs]
            for cid, sex in collar_ids:
                pack_assignments[cid] = rng.choice(packs_used)
        else:
            for cid, sex in collar_ids:
                pack_assignments[cid] = None

        return pack_assignments

    # -------------------------------------------------------------------
    # Main generation flow
    # -------------------------------------------------------------------

    def generate_trial(self, num_keys: int, num_updates: int,
                       condition: str, seed: int, **kwargs) -> dict:
        """Generate a single wildlife tracking narrative interference trial.

        Args:
            num_keys: number of animals to track (2–10)
            num_updates: target tracked-value mentions per animal (3–50)
            condition: "RI" or "PI"
            seed: random seed for full reproducibility
            **kwargs: optional config overrides (species, tracked_attribute,
                      filler_budget, voice, study_area_key, queried_animal_idx,
                      start_month, study_duration_days)

        Returns:
            dict matching the output JSON schema
        """
        rng = random.Random(seed)

        # Step 1: Auto-sample config, then apply overrides
        config = self._auto_config(num_keys, num_updates, condition, seed, rng)
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        species = config.species
        study_area_key = config.study_area_key
        study_area_name = self.study_areas[study_area_key]["name"]
        zones = self.study_areas[study_area_key]["zones"]

        # Step 2: Generate collar IDs
        collar_pairs = self._generate_collar_ids(species, num_keys, rng)
        pack_assignments = self._assign_pack_names(collar_pairs, species, rng)

        # Assign individual names to some animals (wolves get famous names)
        individual_names: Dict[str, Optional[str]] = {}
        available_names = list(self.wolf_names)
        rng.shuffle(available_names)
        for i, (cid, sex) in enumerate(collar_pairs):
            if species == "gray_wolf" and i < len(available_names) and rng.random() < 0.4:
                individual_names[cid] = available_names[i]
            else:
                individual_names[cid] = None

        # Step 3: Initialize animal states
        animal_states: Dict[str, AnimalState] = {}
        initial_zones = rng.sample(zones, min(num_keys, len(zones)))
        if len(initial_zones) < num_keys:
            initial_zones = [rng.choice(zones) for _ in range(num_keys)]

        for i, (cid, sex) in enumerate(collar_pairs):
            zone = initial_zones[i]
            animal_states[cid] = self._init_animal_state(
                cid, species, sex, config.start_month,
                pack_assignments[cid], individual_names[cid], zone, rng
            )

        all_animal_ids = [cid for cid, _ in collar_pairs]

        # Step 4: Assign tracked attributes
        tracked: Dict[str, str] = {}
        for cid, _ in collar_pairs:
            tracked[cid] = config.tracked_attribute

        # Step 5: Setup tracking structures
        entity_tracking: Dict[str, List[str]] = {
            f"{cid} / {tracked[cid]}": [] for cid, _ in collar_pairs
        }
        mention_counts: Dict[str, int] = {cid: 0 for cid, _ in collar_pairs}
        total_target_mentions = num_keys * num_updates

        event_log = []
        full_state_log = []
        last_updated_animal = None

        # Step 6: Project metadata
        project_name = rng.choice(self.project_names)
        sp_info = self.species_info.get(species, {})
        species_plural = sp_info.get("plural", species.replace("_", " ") + "s")

        start_month_name = MONTH_NAMES[config.start_month]
        end_month_idx = ((config.start_month - 1 + config.study_duration_days // 30) % 12) + 1
        end_month_name = MONTH_NAMES[end_month_idx]
        end_year_offset = (config.start_month - 1 + config.study_duration_days // 30) // 12
        start_year = 2022
        end_year = start_year + end_year_offset
        start_date_str = f"{start_month_name} {start_year}"
        end_date_str = f"{end_month_name} {end_year}"

        # Step 7: Event generation loop
        study_day_counter = 0
        days_per_event = max(1, config.study_duration_days // max(total_target_mentions, 1))
        max_iterations = total_target_mentions * 8
        iteration = 0
        current_season = self._get_season(config.start_month)

        while sum(mention_counts.values()) < total_target_mentions:
            iteration += 1
            if iteration > max_iterations:
                break

            # Pick animal (interleave — not the same as last)
            alive_animals = [c for c in all_animal_ids if animal_states[c].alive]
            if not alive_animals:
                # Revive all if needed (study continues with other animals)
                alive_animals = all_animal_ids[:]

            candidates = [c for c in alive_animals if c != last_updated_animal]
            if not candidates:
                candidates = alive_animals

            # Prioritize under-mentioned animals
            min_mc = min(mention_counts[c] for c in candidates)
            urgent = [c for c in candidates if mention_counts[c] < 2]
            if urgent:
                candidates = urgent
            else:
                priority = [c for c in candidates if mention_counts[c] <= min_mc + 1]
                if priority:
                    candidates = priority

            collar_id = rng.choice(candidates)
            an = animal_states[collar_id]

            # Advance study day for this animal
            day_advance = rng.randint(max(1, days_per_event - 2),
                                       max(1, days_per_event + 2))
            self._advance_day(an, day_advance, config.start_month)
            study_day_counter = max(study_day_counter, an.study_day)

            # Determine season and event weights
            season = self._get_season(an.study_month)
            weights = self._get_event_weights(season, an)
            if not weights:
                # Animal dead or no valid events, skip
                last_updated_animal = collar_id
                continue

            # Choose event type
            event_type = self.weighted_random_choice(weights, rng)

            # Execute event
            event_data = {}
            state_changes = []

            if event_type in ("relocation_observation", "visual_observation"):
                event_data, state_changes = self._handle_relocation(
                    collar_id, animal_states, study_area_key, config.start_month, rng, tracked)
                # Choose template voice
                if event_type == "visual_observation":
                    event_type = "visual_observation"

            elif event_type == "capture_workup":
                event_data, state_changes = self._handle_capture_workup(
                    collar_id, animal_states, study_area_key, config.start_month, rng, tracked)

            elif event_type == "kill_site_investigation":
                event_data, state_changes = self._handle_kill_site(
                    collar_id, animal_states, study_area_key, config.start_month, rng, tracked)

            elif event_type == "pack_interaction":
                event_data, state_changes = self._handle_pack_interaction(
                    collar_id, animal_states, study_area_key, config.start_month,
                    rng, tracked, all_animal_ids)

            elif event_type == "dispersal_event":
                event_data, state_changes = self._handle_dispersal(
                    collar_id, animal_states, study_area_key, config.start_month, rng, tracked)

            elif event_type == "denning_event":
                event_data, state_changes = self._handle_denning(
                    collar_id, animal_states, study_area_key, config.start_month, rng, tracked)

            elif event_type == "mortality_event":
                # Only kill if enough animals remain
                if len(alive_animals) > 1:
                    event_data, state_changes = self._handle_mortality(
                        collar_id, animal_states, study_area_key, config.start_month, rng, tracked)
                else:
                    event_data, state_changes = self._handle_relocation(
                        collar_id, animal_states, study_area_key, config.start_month, rng, tracked)
                    event_type = "relocation_observation"

            elif event_type == "health_change":
                event_data, state_changes = self._handle_health_change(
                    collar_id, animal_states, study_area_key, config.start_month, rng, tracked)

            elif event_type == "seasonal_shift":
                event_data, state_changes = self._handle_seasonal_shift(
                    collar_id, animal_states, study_area_key, config.start_month, rng, tracked)

            elif event_type in ("collar_event",):
                event_data, state_changes = self._handle_collar_event(
                    collar_id, animal_states, study_area_key, config.start_month, rng, tracked)

            elif event_type == "territorial_event":
                event_data, state_changes = self._handle_territorial_event(
                    collar_id, animal_states, study_area_key, config.start_month, rng, tracked)

            # Record state changes
            for cid_sc, attr, old_val, new_val, mentioned in state_changes:
                full_state_log.append({
                    "day": an.study_day,
                    "event": event_type,
                    "entity": cid_sc,
                    "attribute": attr,
                    "old_value": old_val,
                    "new_value": new_val,
                    "mentioned_in_narrative": mentioned,
                })
                if mentioned and cid_sc in tracked:
                    key = f"{cid_sc} / {tracked[cid_sc]}"
                    if key in entity_tracking:
                        if (not entity_tracking[key] or
                                entity_tracking[key][-1] != new_val):
                            entity_tracking[key].append(new_val)
                            mention_counts[cid_sc] += 1

            event_log.append({
                "type": event_type,
                "day": an.study_day,
                "data": event_data,
            })

            last_updated_animal = collar_id

        # Step 8: Enforce minimum 2 mentions per entity
        for force_pass in range(5):
            under = [c for c in all_animal_ids if mention_counts[c] < 2 and animal_states[c].alive]
            if not under:
                break
            for cid in under:
                an = animal_states[cid]
                self._advance_day(an, days_per_event, config.start_month)
                event_data, state_changes = self._handle_relocation(
                    cid, animal_states, study_area_key, config.start_month, rng, tracked)
                for cid_sc, attr, old_val, new_val, mentioned in state_changes:
                    full_state_log.append({
                        "day": an.study_day,
                        "event": "relocation_observation",
                        "entity": cid_sc,
                        "attribute": attr,
                        "old_value": old_val,
                        "new_value": new_val,
                        "mentioned_in_narrative": mentioned,
                    })
                    if mentioned and cid_sc in tracked:
                        key = f"{cid_sc} / {tracked[cid_sc]}"
                        if key in entity_tracking:
                            if (not entity_tracking[key] or
                                    entity_tracking[key][-1] != new_val):
                                entity_tracking[key].append(new_val)
                                mention_counts[cid_sc] += 1
                event_log.append({
                    "type": "relocation_observation",
                    "day": an.study_day,
                    "data": event_data,
                })

        # Step 8b: Ensure ALL entities have at least 1 value in entity_tracking
        # (entities with 0 values are invisible in the narrative — unacceptable)
        for cid, _ in collar_pairs:
            key = f"{cid} / {tracked[cid]}"
            if key in entity_tracking and len(entity_tracking[key]) == 0:
                an = animal_states[cid]
                if not an.alive:
                    continue
                # Force a relocation event that will update the entity
                self._advance_day(an, 3, config.start_month)
                event_data, state_changes = self._handle_relocation(
                    cid, animal_states, study_area_key, config.start_month, rng, tracked)
                # Force: always record the value even if same as previous
                for cid_sc, attr, old_val, new_val, mentioned in state_changes:
                    if cid_sc == cid:
                        entity_tracking[key] = [new_val]  # forced first mention
                        mention_counts[cid] = max(mention_counts[cid], 1)
                event_log.append({
                    "type": "relocation_observation",
                    "day": an.study_day,
                    "data": event_data,
                })

        # Step 9: Render narrative
        narrative_parts = []

        # 9a: Study header
        header_template = rng.choice(self.templates["header_templates"])
        header_data = {
            "project_name": project_name,
            "num_animals": num_keys,
            "species_plural": species_plural,
            "study_area": study_area_name,
            "start_date": start_date_str,
            "end_date": end_date_str,
        }
        header = self.render_template(header_template, header_data)
        narrative_parts.append(header)
        narrative_parts.append("")

        # 9b: Render events with optional filler
        filler_cfg = FILLER_CONFIG[config.filler_budget]
        current_season_rendered = self._get_season(config.start_month)

        for i, event in enumerate(event_log):
            # Insert season transition if season changed
            event_animal = event["data"].get("animal_id", all_animal_ids[0])
            if event_animal in animal_states:
                event_month = animal_states[event_animal].study_month
            else:
                event_month = config.start_month
            event_season = self._get_season(event_month)

            if event_season != current_season_rendered and i > 0:
                trans_template = rng.choice(self.templates["season_transition_templates"])
                trans = self.render_template(trans_template, {
                    "season": event_season,
                    "species_plural": species_plural,
                    "study_area": study_area_name,
                })
                narrative_parts.append(trans)
                narrative_parts.append("")
                current_season_rendered = event_season

            # Render the event
            templates_list = self._get_templates_for_event(event["type"], config.voice)
            template = rng.choice(templates_list)
            rendered = self.render_template(template, event["data"])

            # Replace wolf-specific social terminology with species-appropriate terms
            sp_info = self.species_info.get(config.species, {})
            group_label = sp_info.get("group_label", "group")
            offspring_label = sp_info.get("offspring_label", "young")
            if group_label != "pack":
                rendered = rendered.replace("pack territory", f"{group_label} territory")
                rendered = rendered.replace("pack range", f"{group_label} range")
                rendered = rendered.replace("pack affiliation", f"{group_label} affiliation")
                rendered = rendered.replace("pack activity", f"{group_label} activity")
                rendered = rendered.replace("from pack", f"from {group_label}")
                rendered = rendered.replace(" pack ", f" {group_label} ")

            narrative_parts.append(rendered)

            # Maybe insert filler
            if rng.random() < filler_cfg["insert_probability"]:
                n_sentences = filler_cfg["sentences_per_insertion"]
                if isinstance(n_sentences, tuple):
                    n_sentences = rng.randint(n_sentences[0], n_sentences[1])
                filler_templates_list = rng.sample(
                    self.templates["filler_templates"],
                    min(n_sentences, len(self.templates["filler_templates"]))
                )
                for ft in filler_templates_list:
                    filler_animal = rng.choice(all_animal_ids)
                    filler_text = self._render_filler_sentence(
                        filler_animal, animal_states, study_area_key, rng)
                    narrative_parts.append(filler_text)

        # Step 10: Generate questions
        queried_idx = min(config.queried_animal_idx, len(all_animal_ids) - 1)
        queried_id = all_animal_ids[queried_idx]
        queried_attr = tracked[queried_id]
        tracking_key = f"{queried_id} / {queried_attr}"
        values = entity_tracking[tracking_key]

        # Ensure RI != PI: find an animal with distinct first/last values
        if len(values) < 2 or values[0] == values[-1]:
            for alt_id in all_animal_ids:
                alt_key = f"{alt_id} / {tracked[alt_id]}"
                alt_values = entity_tracking[alt_key]
                if len(alt_values) >= 2 and alt_values[0] != alt_values[-1]:
                    queried_id = alt_id
                    queried_attr = tracked[alt_id]
                    tracking_key = alt_key
                    values = alt_values
                    break

        # Last resort: force divergence with a bonus event
        if len(values) < 2 or values[0] == values[-1]:
            best_id = max(
                all_animal_ids,
                key=lambda c: len(entity_tracking[f"{c} / {tracked[c]}"])
            )
            best_attr = tracked[best_id]
            best_key = f"{best_id} / {best_attr}"
            best_an = animal_states[best_id]

            old_fmt = best_an.format_attribute(best_attr)
            if best_attr == "weight_kg":
                best_an.weight_kg = round(best_an.weight_kg + rng.choice([1.5, 2.0, -1.5, -2.0]), 1)
            elif best_attr == "body_condition_score":
                best_an.body_condition_score = max(1, min(9, best_an.body_condition_score + rng.choice([-1, 1])))
            elif best_attr == "collar_battery_pct":
                best_an.collar_battery_pct = max(5.0, best_an.collar_battery_pct - rng.uniform(10, 25))
            elif best_attr == "daily_movement_km":
                best_an.daily_movement_km = round(best_an.daily_movement_km + rng.choice([3.0, -3.0, 4.0, -4.0]), 1)
            elif best_attr == "elevation_m":
                best_an.elevation_m += rng.choice([-150, 150, 200, -200])
            elif best_attr == "location_zone":
                # Force move to an adjacent zone
                current_zone = best_an.location_zone
                sa_data = self.data["study_areas"].get(study_area_key, {})
                adjacency = sa_data.get("zone_adjacency", {})
                neighbors = adjacency.get(current_zone, [])
                all_zones = sa_data.get("zones", [current_zone])
                candidates = [z for z in (neighbors if neighbors else all_zones) if z != current_zone]
                if candidates:
                    best_an.location_zone = rng.choice(candidates)
            elif best_attr == "activity_state":
                activity_options = ["resting at den site", "traveling through drainage",
                                    "at rendezvous site", "hunting (stalking)", "feeding at kill site",
                                    "patrolling territory", "bedded in timber", "on exposed ridge"]
                candidates = [a for a in activity_options if a != best_an.activity_state]
                if candidates:
                    best_an.activity_state = rng.choice(candidates)
            elif best_attr == "health_status":
                health_options = ["healthy (excellent condition)", "healthy (average condition)",
                                  "minor limp (left hind)", "bite wound (shoulder, healing)",
                                  "sarcoptic mange (early)", "emaciated (ribs prominent)"]
                candidates = [h for h in health_options if h != best_an.health_status]
                if candidates:
                    best_an.health_status = rng.choice(candidates)
            elif best_attr == "habitat_type":
                habitat_options = ["alpine meadow (above treeline)", "subalpine meadow (near treeline)",
                                   "lodgepole pine forest (dense)", "spruce-fir forest (dense canopy)",
                                   "sagebrush steppe (open)", "open grassland (valley floor)",
                                   "riparian cottonwood gallery", "dense willow thicket (riparian)"]
                candidates = [h for h in habitat_options if h != best_an.habitat_type]
                if candidates:
                    best_an.habitat_type = rng.choice(candidates)
            new_fmt = best_an.format_attribute(best_attr)

            bonus_text = (
                f"By the end of the monitoring period, "
                f"{best_id}'s {best_attr.replace('_', ' ')} had shifted to {new_fmt}."
            )
            narrative_parts.append(bonus_text)
            if not entity_tracking[best_key] or entity_tracking[best_key][-1] != new_fmt:
                entity_tracking[best_key].append(new_fmt)
                mention_counts[best_id] += 1

            queried_id = best_id
            queried_attr = best_attr
            tracking_key = best_key
            values = entity_tracking[best_key]

        ri_answer = values[0] if values else "unknown"
        pi_answer = values[-1] if values else "unknown"

        attr_display = queried_attr.replace("_", " ")
        RI_QUESTION_TEMPLATES = [
            f"What was {queried_id}'s {attr_display} when first recorded in this study?",
            f"What was the first reported {attr_display} for {queried_id}?",
            f"At {queried_id}'s first mention in the field log, what was the {attr_display}?",
        ]
        PI_QUESTION_TEMPLATES = [
            f"What was {queried_id}'s {attr_display} at the most recent observation?",
            f"In the last mention of {queried_id}'s {attr_display}, what was the value?",
            f"What was the final recorded {attr_display} for {queried_id}?",
        ]

        questions = {
            "RI": {
                "question": rng.choice(RI_QUESTION_TEMPLATES),
                "expected_answer": ri_answer,
                "target_entity": queried_id,
                "target_attribute": queried_attr,
            },
            "PI": {
                "question": rng.choice(PI_QUESTION_TEMPLATES),
                "expected_answer": pi_answer,
                "target_entity": queried_id,
                "target_attribute": queried_attr,
            },
        }

        # Step 11: Finalize narrative text
        final_parts = []
        for p in narrative_parts:
            if p == "":
                if final_parts and not final_parts[-1].endswith("\n"):
                    final_parts.append("\n")
            else:
                final_parts.append(p)
        narrative = " ".join(final_parts).replace(" \n ", "\n\n")

        # Step 12: Assemble output
        return {
            "id": f"wildlife_{seed:06d}",
            "domain": "wildlife",
            "num_keys": num_keys,
            "num_updates": num_updates,
            "narrative": narrative,
            "questions": questions,
            "entity_tracking": entity_tracking,
            "full_state_log": full_state_log,
            "mention_counts": mention_counts,
            "config": {
                "seed": seed,
                "num_keys": num_keys,
                "num_updates": num_updates,
                "condition": condition,
                "species": config.species,
                "tracked_attribute": config.tracked_attribute,
                "filler_budget": config.filler_budget,
                "voice": config.voice,
                "study_area": study_area_key,
                "study_area_name": study_area_name,
                "queried_animal_idx": config.queried_animal_idx,
                "start_month": config.start_month,
                "study_duration_days": config.study_duration_days,
                "project_name": project_name,
                "animals": [[cid, sex] for cid, sex in collar_pairs],
            },
        }

    # -------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------

    def validate_trial(self, trial: dict) -> bool:
        narrative = trial["narrative"]
        entity_tracking = trial["entity_tracking"]
        questions = trial["questions"]

        # Check tracked values appear in narrative
        for key, vals in entity_tracking.items():
            for val in vals:
                assert val in narrative, (
                    f"Value '{val}' for '{key}' not found in narrative text"
                )

        # Check RI != PI
        ri_answer = questions["RI"]["expected_answer"]
        pi_answer = questions["PI"]["expected_answer"]
        assert ri_answer != pi_answer, (
            f"RI and PI answers must differ, got '{ri_answer}' for both"
        )

        # Check queried entity has at least 2 mentions
        target = questions["RI"]["target_entity"]
        target_attr = questions["RI"]["target_attribute"]
        tracking_key = f"{target} / {target_attr}"
        values = entity_tracking.get(tracking_key, [])
        assert len(values) >= 2, (
            f"Need at least 2 mentions for '{tracking_key}', got {len(values)}"
        )

        # Check RI is first value, PI is last
        assert values[0] == ri_answer, (
            f"RI answer should be first value '{values[0]}', got '{ri_answer}'"
        )
        assert values[-1] == pi_answer, (
            f"PI answer should be last value '{values[-1]}', got '{pi_answer}'"
        )

        return True

    # -------------------------------------------------------------------
    # Batch generation
    # -------------------------------------------------------------------

    DOMAIN = "wildlife"

    @staticmethod
    def _get_data_dir() -> Path:
        """Return data/narrative_interference/wildlife/ (project root relative)."""
        current = Path(__file__).resolve().parent
        for _ in range(10):
            if (current / ".git").exists() or (current / "CLAUDE.md").exists():
                break
            current = current.parent
        return current / "data" / "narrative_interference" / "wildlife"

    def generate_batch(
        self,
        key_levels: List[int] = None,
        update_levels: List[int] = None,
        trials_per_cell: int = 30,
        seed_start: int = 200000,
        **kwargs,
    ) -> dict:
        """Generate a full batch of trials across a grid.

        Args:
            key_levels: list of num_keys values (default: [2,3,5,7,10])
            update_levels: list of num_updates values (default: [1,3,5,10,20,30,40,50])
            trials_per_cell: trials per (keys, updates, condition) cell
            seed_start: starting seed value
            **kwargs: config overrides passed to generate_trial()

        Returns:
            dict with 'metadata' and 'trials' keys
        """
        import time
        from datetime import datetime, timezone

        if key_levels is None:
            key_levels = [2, 3, 5, 7, 10]
        if update_levels is None:
            update_levels = [1, 3, 5, 10, 20, 30, 40, 50]

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        trials = []
        failures = []
        seed = seed_start
        total_cells = len(key_levels) * len(update_levels) * 2
        cell_idx = 0
        t0 = time.time()

        for nk in key_levels:
            for nu in update_levels:
                for cond in ["RI", "PI"]:
                    cell_idx += 1
                    cell_ok = 0
                    for _ in range(trials_per_cell):
                        try:
                            trial = self.generate_trial(nk, nu, cond, seed, **kwargs)
                            self.validate_trial(trial)
                            trials.append(trial)
                            cell_ok += 1
                        except Exception as e:
                            failures.append({
                                "seed": seed, "nk": nk, "nu": nu,
                                "cond": cond, "error": str(e)[:100],
                            })
                        seed += 1

                    elapsed = time.time() - t0
                    rate = len(trials) / max(elapsed, 0.1)
                    print(
                        f"Cell {cell_idx}/{total_cells}: "
                        f"nk={nk} nu={nu} cond={cond} "
                        f"ok={cell_ok}/{trials_per_cell} "
                        f"rate={rate:.1f}/s"
                    )

        return {
            "metadata": {
                "domain": "wildlife",
                "generated_at": ts,
                "total_trials": len(trials),
                "total_failures": len(failures),
                "key_levels": key_levels,
                "update_levels": update_levels,
                "trials_per_cell": trials_per_cell,
            },
            "trials": trials,
            "failures": failures,
        }
