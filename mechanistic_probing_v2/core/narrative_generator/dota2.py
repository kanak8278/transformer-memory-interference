"""Dota 2 narrative interference trial generator.

Implements the full generation flow from the code spec:
- Hero/team/tournament selection
- Event sequence generation with phase-aware weights
- State tracking for all heroes
- Narrative rendering with templates and filler
- Question generation for RI/PI conditions
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from .base import NarrativeTrialGenerator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAME_PHASES = {
    "laning": {"start": 0, "end": 12},
    "mid_game": {"start": 12, "end": 26},
    "late_game": {"start": 26, "end": 60},
}

EVENT_TIME_ADVANCE = {
    "farming_update":    {"laning": (1.0, 2.5), "mid_game": (1.0, 3.0), "late_game": (1.0, 2.0)},
    "solo_kill":         {"laning": (0.5, 1.5), "mid_game": (0.5, 2.0), "late_game": (0.3, 1.5)},
    "teamfight":         {"laning": (1.0, 2.0), "mid_game": (0.5, 2.0), "late_game": (0.5, 1.5)},
    "item_purchase":     {"laning": (0.5, 1.5), "mid_game": (0.5, 2.0), "late_game": (0.3, 1.0)},
    "tower_kill":        {"laning": (1.0, 2.0), "mid_game": (0.5, 2.0), "late_game": (0.5, 1.5)},
    "roshan_kill":       {"laning": (2.0, 3.0), "mid_game": (1.0, 2.5), "late_game": (0.5, 2.0)},
    "level_up":          {"laning": (0.5, 1.5), "mid_game": (0.5, 2.0), "late_game": (0.5, 1.5)},
    "death_and_respawn": {"laning": (0.5, 1.5), "mid_game": (0.5, 2.0), "late_game": (0.5, 2.0)},
    "barracks_kill":     {"laning": (2.0, 3.0), "mid_game": (1.0, 2.0), "late_game": (0.5, 1.5)},
}

DEFAULT_EVENT_WEIGHTS = {
    "laning": {
        "farming_update": 40, "solo_kill": 15, "item_purchase": 25,
        "level_up": 15, "tower_kill": 5, "teamfight": 0,
        "roshan_kill": 0, "death_and_respawn": 0, "barracks_kill": 0,
    },
    "mid_game": {
        "farming_update": 15, "solo_kill": 20, "teamfight": 15,
        "item_purchase": 20, "tower_kill": 10, "roshan_kill": 5,
        "level_up": 10, "death_and_respawn": 5, "barracks_kill": 0,
    },
    "late_game": {
        "farming_update": 5, "solo_kill": 10, "teamfight": 25,
        "item_purchase": 15, "tower_kill": 15, "roshan_kill": 10,
        "barracks_kill": 10, "level_up": 5, "death_and_respawn": 5,
    },
}

ARCHETYPE_MODIFIERS = {
    "stomp": {
        "weight_mods": {
            "laning":    {"solo_kill": 1.5, "farming_update": 0.8, "tower_kill": 1.5},
            "mid_game":  {"solo_kill": 1.5, "teamfight": 1.3, "tower_kill": 1.5, "farming_update": 0.5},
            "late_game": {"teamfight": 1.5, "tower_kill": 1.5, "barracks_kill": 1.5, "farming_update": 0.3},
        },
    },
    "comeback": {
        "weight_mods": {
            "laning":    {"solo_kill": 1.3, "farming_update": 1.0},
            "mid_game":  {"teamfight": 1.5, "roshan_kill": 1.5, "solo_kill": 0.8},
            "late_game": {"teamfight": 1.5, "roshan_kill": 1.3, "tower_kill": 1.3},
        },
    },
    "close": {
        "weight_mods": {
            "laning":    {"farming_update": 1.2, "solo_kill": 1.0},
            "mid_game":  {"teamfight": 1.3, "solo_kill": 1.0, "roshan_kill": 1.3},
            "late_game": {"teamfight": 1.5, "roshan_kill": 1.5},
        },
    },
    "farmfest": {
        "weight_mods": {
            "laning":    {"farming_update": 1.8, "solo_kill": 0.3, "item_purchase": 1.3},
            "mid_game":  {"farming_update": 1.5, "solo_kill": 0.4, "teamfight": 0.4, "item_purchase": 1.5},
            "late_game": {"farming_update": 1.0, "teamfight": 1.2, "item_purchase": 1.3},
        },
    },
    "bloodbath": {
        "weight_mods": {
            "laning":    {"solo_kill": 2.0, "farming_update": 0.5, "death_and_respawn": 1.5},
            "mid_game":  {"solo_kill": 1.8, "teamfight": 1.8, "farming_update": 0.3, "death_and_respawn": 1.3},
            "late_game": {"teamfight": 2.0, "solo_kill": 1.5, "death_and_respawn": 1.5, "farming_update": 0.2},
        },
    },
    "split_push": {
        "weight_mods": {
            "laning":    {"farming_update": 1.3, "tower_kill": 1.5},
            "mid_game":  {"tower_kill": 2.0, "farming_update": 1.3, "teamfight": 0.5, "solo_kill": 0.7},
            "late_game": {"tower_kill": 2.0, "barracks_kill": 2.0, "teamfight": 0.5, "farming_update": 1.0},
        },
    },
    "roshan_centric": {
        "weight_mods": {
            "laning":    {"farming_update": 1.0},
            "mid_game":  {"roshan_kill": 3.0, "teamfight": 1.5},
            "late_game": {"roshan_kill": 3.0, "teamfight": 1.5, "barracks_kill": 1.3},
        },
    },
    "base_race": {
        "weight_mods": {
            "laning":    {"farming_update": 1.2},
            "mid_game":  {"tower_kill": 1.5, "teamfight": 1.0},
            "late_game": {"tower_kill": 2.5, "barracks_kill": 2.5, "teamfight": 0.5, "farming_update": 0.2},
        },
    },
}

FILLER_CONFIG = {
    "minimal": {"insert_probability": 0.05, "sentences_per_insertion": 1},
    "light":   {"insert_probability": 0.25, "sentences_per_insertion": 1},
    "medium":  {"insert_probability": 0.45, "sentences_per_insertion": (1, 2)},
    "heavy":   {"insert_probability": 0.65, "sentences_per_insertion": (1, 3)},
}

# Trackable attributes that are safe to use as tracked attributes
# (ones that change frequently and have numeric or string values)
SAFE_TRACKED_ATTRIBUTES = [
    "gold", "net_worth", "kills", "deaths", "assists", "kda_string",
    "last_hits", "level", "gpm", "xpm", "total_damage_dealt",
    "total_damage_taken",
]

GOLD_TRAJECTORY = {
    "pos1": {5: (1800, 2500), 10: (3500, 5000), 15: (5500, 8000), 20: (8000, 12000), 25: (11000, 16000), 30: (14000, 22000), 35: (17000, 28000), 40: (20000, 35000)},
    "pos2": {5: (1600, 2300), 10: (3200, 4500), 15: (5000, 7200), 20: (7000, 11000), 25: (9500, 14000), 30: (12000, 19000), 35: (14500, 24000), 40: (17000, 30000)},
    "pos3": {5: (1200, 1800), 10: (2500, 3800), 15: (4000, 6000), 20: (5500, 8500), 25: (7500, 11000), 30: (9500, 14000), 35: (11500, 17000), 40: (13500, 20000)},
    "pos4": {5: (900, 1400), 10: (1800, 2800), 15: (2800, 4200), 20: (3800, 5800), 25: (5000, 7500), 30: (6200, 9500), 35: (7500, 11500), 40: (8500, 13500)},
    "pos5": {5: (800, 1200), 10: (1500, 2500), 15: (2500, 4000), 20: (3500, 5500), 25: (4500, 7000), 30: (6000, 9000), 35: (7000, 11000), 40: (8000, 13000)},
}

LEVEL_TRAJECTORY = {
    "pos1": {5: (4, 5), 10: (8, 10), 15: (12, 14), 20: (16, 18), 25: (19, 22), 30: (22, 25), 35: (24, 27), 40: (26, 30)},
    "pos2": {5: (5, 6), 10: (9, 11), 15: (13, 15), 20: (17, 19), 25: (20, 23), 30: (23, 26), 35: (25, 28), 40: (27, 30)},
    "pos3": {5: (3, 5), 10: (7, 9), 15: (10, 13), 20: (14, 17), 25: (17, 20), 30: (20, 23), 35: (22, 26), 40: (24, 28)},
    "pos4": {5: (3, 4), 10: (5, 7), 15: (8, 11), 20: (11, 14), 25: (14, 17), 30: (16, 20), 35: (18, 22), 40: (20, 25)},
    "pos5": {5: (2, 4), 10: (4, 6), 15: (7, 10), 20: (10, 13), 25: (12, 16), 30: (15, 19), 35: (17, 21), 40: (19, 24)},
}

LAST_HIT_TRAJECTORY = {
    "pos1": {10: (55, 70), 20: (130, 170), 30: (220, 280), 40: (300, 380)},
    "pos2": {10: (50, 65), 20: (120, 150), 30: (200, 250), 40: (270, 340)},
    "pos3": {10: (30, 45), 20: (70, 100), 30: (120, 160), 40: (160, 220)},
    "pos4": {10: (5, 15), 20: (20, 40), 30: (30, 60), 40: (40, 80)},
    "pos5": {10: (0, 5), 20: (5, 15), 30: (10, 25), 40: (15, 35)},
}

# KDA rates per minute by role
KPM = {"pos1": 0.25, "pos2": 0.28, "pos3": 0.18, "pos4": 0.12, "pos5": 0.08}
DPM = {"pos1": 0.12, "pos2": 0.15, "pos3": 0.18, "pos4": 0.20, "pos5": 0.25}
APM = {"pos1": 0.25, "pos2": 0.30, "pos3": 0.35, "pos4": 0.40, "pos5": 0.45}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HeroState:
    name: str
    role: str
    team: str  # "radiant" or "dire"
    gold: int
    net_worth: int
    level: int
    kills: int
    deaths: int
    assists: int
    last_hits: int
    denies: int
    items: list
    alive: bool
    kill_streak: int
    total_damage_dealt: int
    total_damage_taken: int
    tower_damage: int
    healing_done: int
    wards_placed: int
    camps_stacked: int
    runes_collected: int
    respawn_at: float = 0.0
    _gpm: int = 0
    _xpm: int = 0
    _nw_rank: int = 1
    _gold_advantage: int = 0

    def get_attribute(self, attr_name: str):
        if attr_name == "kda_string":
            return f"{self.kills}/{self.deaths}/{self.assists}"
        elif attr_name == "gpm":
            return self._gpm
        elif attr_name == "xpm":
            return self._xpm
        elif attr_name == "latest_item":
            return self.items[-1] if self.items else "none"
        elif attr_name == "inventory_count":
            return len(self.items)
        elif attr_name == "buyback_status":
            return "available"
        elif attr_name == "respawn_timer":
            return int(5 + 3.8 * self.level)
        elif attr_name == "net_worth_rank":
            return self._nw_rank
        elif attr_name == "gold_advantage":
            return self._gold_advantage
        else:
            return getattr(self, attr_name, 0)

    def format_attribute(self, attr_name: str) -> str:
        """Return a string representation of the attribute value suitable for narrative.
        This is the canonical string form used in BOTH the narrative text and entity_tracking."""
        val = self.get_attribute(attr_name)
        return str(val)


@dataclass
class DotaTrialConfig:
    num_keys: int
    num_updates: int
    condition: str
    seed: int
    tracked_attribute: str
    attribute_mode: str
    filler_budget: str
    voice: str
    match_scope: str
    game_archetype: str
    queried_hero_idx: int
    start_time: float = 0.0
    stomp_winner: Optional[str] = None
    early_leader: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_phase(game_time_minutes: float) -> str:
    if game_time_minutes < 12:
        return "laning"
    elif game_time_minutes < 26:
        return "mid_game"
    else:
        return "late_game"


def get_event_weights(phase: str, archetype: str) -> dict:
    base = dict(DEFAULT_EVENT_WEIGHTS[phase])
    mods = ARCHETYPE_MODIFIERS[archetype]["weight_mods"].get(phase, {})
    for event_type, multiplier in mods.items():
        if event_type in base:
            base[event_type] = base[event_type] * multiplier
    total = sum(base.values())
    if total > 0:
        base = {k: (v / total) * 100 for k, v in base.items()}
    return base


def get_gold_multiplier(archetype: str, phase: str, team: str,
                        config: DotaTrialConfig) -> float:
    if archetype == "stomp":
        if team == config.stomp_winner:
            return 1.5 if phase == "laning" else 2.0
        else:
            return 0.7 if phase == "laning" else 0.5
    elif archetype == "comeback":
        if phase == "laning":
            return 1.4 if team == config.early_leader else 0.7
        else:
            return 0.7 if team == config.early_leader else 1.3
    return 1.0


def determine_start_time(num_updates: int, rng: random.Random) -> float:
    if num_updates <= 5:
        return round(rng.uniform(5.0, 20.0), 1)
    elif num_updates <= 10:
        return round(rng.uniform(0.0, 15.0), 1)
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Main Generator
# ---------------------------------------------------------------------------

class DotaTrialGenerator(NarrativeTrialGenerator):

    def __init__(self):
        data_dir = Path(__file__).parent / "data"
        with open(data_dir / "dota2_heroes.json") as f:
            heroes_data = json.load(f)
        with open(data_dir / "dota2_items.json") as f:
            items_data = json.load(f)
        with open(data_dir / "dota2_templates.json") as f:
            self.templates = json.load(f)
        with open(data_dir / "dota2_names.json") as f:
            self.names = json.load(f)

        self.hero_roles = heroes_data["hero_roles"]
        self.role_attribute_affinity = heroes_data["role_attribute_affinity"]
        self.items_by_tier = items_data["items"]
        self.role_item_builds = items_data["role_item_builds"]

        # Build item cost lookup
        self.item_costs: Dict[str, int] = {}
        for tier_items in self.items_by_tier.values():
            for item in tier_items:
                self.item_costs[item["name"]] = item["cost"]

        # Build role pools
        self.role_pool = self._build_role_pool()

    def _build_role_pool(self) -> Dict[str, List[str]]:
        pool = {"pos1": [], "pos2": [], "pos3": [], "pos4": [], "pos5": []}
        for hero, roles in self.hero_roles.items():
            for role in roles:
                pool[role].append(hero)
        return pool

    def _get_item_cost(self, item_name: str) -> int:
        return self.item_costs.get(item_name, 0)

    # -------------------------------------------------------------------
    # Auto-config
    # -------------------------------------------------------------------

    def _auto_config(self, num_keys: int, num_updates: int,
                     condition: str, seed: int,
                     rng: random.Random) -> DotaTrialConfig:
        tracked_attribute = rng.choice(SAFE_TRACKED_ATTRIBUTES)
        attribute_mode = rng.choice(["same", "mixed"])
        filler_budget = rng.choice(["minimal", "light", "medium", "heavy"])
        voice = rng.choice(["analyst", "caster"])
        match_scope = rng.choice(["full", "window"])
        game_archetype = rng.choice(list(ARCHETYPE_MODIFIERS.keys()))
        queried_hero_idx = rng.randint(0, num_keys - 1)

        return DotaTrialConfig(
            num_keys=num_keys,
            num_updates=num_updates,
            condition=condition,
            seed=seed,
            tracked_attribute=tracked_attribute,
            attribute_mode=attribute_mode,
            filler_budget=filler_budget,
            voice=voice,
            match_scope=match_scope,
            game_archetype=game_archetype,
            queried_hero_idx=queried_hero_idx,
        )

    # -------------------------------------------------------------------
    # Hero state initialization
    # -------------------------------------------------------------------

    def _init_kda(self, role: str, start_time: float,
                  rng: random.Random) -> Tuple[int, int, int]:
        kills = max(0, int(rng.gauss(KPM[role] * start_time, 1.0)))
        deaths = max(0, int(rng.gauss(DPM[role] * start_time, 1.0)))
        assists = max(0, int(rng.gauss(APM[role] * start_time, 1.5)))
        return kills, deaths, assists

    def _init_items(self, role: str, gold: int, rng: random.Random) -> list:
        build = self.role_item_builds[role]
        items = []
        spent = 0

        # Always have boots by mid game
        boot_item = build["early"][0]
        items.append(boot_item)
        spent += self._get_item_cost(boot_item)

        for item_name in build["early"][1:]:
            cost = self._get_item_cost(item_name)
            if spent + cost <= gold * 0.9:
                items.append(item_name)
                spent += cost

        for item_name in build["core"]:
            cost = self._get_item_cost(item_name)
            if spent + cost <= gold * 0.9 and len(items) < 6:
                items.append(item_name)
                spent += cost

        for item_name in build["luxury"]:
            cost = self._get_item_cost(item_name)
            if spent + cost <= gold * 0.9 and len(items) < 6:
                items.append(item_name)
                spent += cost

        return items[:6]

    def _initialize_hero_state(self, hero: str, role: str, start_time: float,
                               archetype: str, team: str,
                               config: DotaTrialConfig,
                               rng: random.Random) -> HeroState:
        if start_time == 0:
            starting_items = list(self.role_item_builds[role]["starting"])
            return HeroState(
                name=hero, role=role, team=team,
                gold=600, net_worth=600, level=1,
                kills=0, deaths=0, assists=0,
                last_hits=0, denies=0,
                items=starting_items,
                alive=True, kill_streak=0,
                total_damage_dealt=0, total_damage_taken=0,
                tower_damage=0, healing_done=0,
                wards_placed=0, camps_stacked=0, runes_collected=0,
            )

        phase = get_phase(start_time)
        gold_mult = get_gold_multiplier(archetype, phase, team, config)
        gold_range = self.interpolate_range(GOLD_TRAJECTORY[role], start_time)
        raw_gold = rng.randint(int(gold_range[0] * gold_mult),
                               max(int(gold_range[0] * gold_mult),
                                   int(gold_range[1] * gold_mult)))

        kills, deaths, assists = self._init_kda(role, start_time, rng)
        level_range = self.interpolate_range(LEVEL_TRAJECTORY[role], start_time)
        level = rng.randint(level_range[0], max(level_range[0], level_range[1]))

        lh_range = self.interpolate_range(LAST_HIT_TRAJECTORY[role], start_time)
        last_hits = rng.randint(lh_range[0], max(lh_range[0], lh_range[1]))

        items = self._init_items(role, raw_gold, rng)
        net_worth = sum(self._get_item_cost(i) for i in items) + rng.randint(200, 1500)

        td_mean = max(1, int(start_time * 400))
        td_std = max(1, int(start_time * 80))
        tt_mean = max(1, int(start_time * 300))
        tt_std = max(1, int(start_time * 60))

        return HeroState(
            name=hero, role=role, team=team,
            gold=rng.randint(200, 2000),
            net_worth=net_worth,
            level=level,
            kills=kills, deaths=deaths, assists=assists,
            last_hits=last_hits,
            denies=rng.randint(0, max(1, int(start_time * 0.5))),
            items=items,
            alive=True,
            kill_streak=max(0, kills - deaths) if kills > deaths else 0,
            total_damage_dealt=max(0, int(rng.gauss(td_mean, td_std))),
            total_damage_taken=max(0, int(rng.gauss(tt_mean, tt_std))),
            tower_damage=rng.randint(0, max(1, int(start_time * 100))),
            healing_done=rng.randint(0, max(1, int(start_time * 50))) if role in ("pos4", "pos5") else 0,
            wards_placed=rng.randint(0, max(1, int(start_time * 0.3))) if role in ("pos4", "pos5") else 0,
            camps_stacked=rng.randint(0, max(1, int(start_time * 0.15))) if role in ("pos4", "pos5") else 0,
            runes_collected=rng.randint(0, max(1, int(start_time * 0.1))),
        )

    # -------------------------------------------------------------------
    # Computed attributes update
    # -------------------------------------------------------------------

    def _update_computed_attrs(self, hero_states: Dict[str, HeroState],
                               game_time: float,
                               all_match_heroes: list,
                               rng: random.Random):
        """Update GPM, XPM, NW rank, gold advantage for all heroes."""
        if game_time <= 0:
            return

        for hero_name, _ in all_match_heroes:
            hs = hero_states[hero_name]
            hs._gpm = max(100, min(900, int(hs.net_worth / max(1, game_time)))) if game_time > 0 else 0
            hs._xpm = max(100, int(hs.level * 40 + rng.randint(-30, 30)))

        # NW rank
        nw_list = [(hs.net_worth, hero_name) for hero_name, _ in all_match_heroes]
        nw_list.sort(reverse=True)
        for rank, (_, hero_name) in enumerate(nw_list, 1):
            hero_states[hero_name]._nw_rank = rank

        # Gold advantage per team
        radiant_nw = sum(hero_states[h].net_worth for h, _ in all_match_heroes
                         if hero_states[h].team == "radiant")
        dire_nw = sum(hero_states[h].net_worth for h, _ in all_match_heroes
                      if hero_states[h].team == "dire")
        for hero_name, _ in all_match_heroes:
            hs = hero_states[hero_name]
            if hs.team == "radiant":
                hs._gold_advantage = radiant_nw - dire_nw
            else:
                hs._gold_advantage = dire_nw - radiant_nw

    # -------------------------------------------------------------------
    # Event handlers
    # -------------------------------------------------------------------

    def _handle_farming_update(self, hero: str, hero_states: Dict[str, HeroState],
                               game_time: float, rng: random.Random,
                               config: DotaTrialConfig,
                               tracked: Dict[str, str]) -> Tuple[dict, list]:
        hs = hero_states[hero]
        attr_name = tracked[hero]
        old_val = hs.get_attribute(attr_name)

        # Farming-updatable attributes (change naturally from farming)
        FARMING_ATTRS = {"gold", "net_worth", "last_hits", "gpm", "xpm", "level",
                         "denies", "total_damage_dealt"}
        # Combat-only attributes (don't change from farming)
        COMBAT_ONLY_ATTRS = {"kills", "deaths", "assists", "kda_string", "kill_streak",
                             "total_damage_taken"}

        # Update farming stats (always happens regardless of tracked attr)
        lh_gain = rng.randint(5, 25)
        hs.last_hits += lh_gain
        hs.denies += rng.randint(0, 3)
        gold_gain = lh_gain * rng.randint(35, 55)
        phase = get_phase(game_time)
        gold_mult = get_gold_multiplier(config.game_archetype, phase, hs.team, config)
        gold_gain = int(gold_gain * gold_mult)
        hs.gold += gold_gain
        hs.net_worth += gold_gain

        # Maybe bump level
        if rng.random() < 0.3 and hs.level < 30:
            hs.level += 1

        # Determine what to mention in narrative
        if attr_name in COMBAT_ONLY_ATTRS:
            # Don't mention combat stats in farming context — mention gold/CS instead
            display_attr = rng.choice(["gold", "net_worth", "last_hits"])
            display_val = hs.format_attribute(display_attr)
            mentioned = False  # NOT a tracked mention — just filler
            new_val = old_val  # tracked attr didn't change
        else:
            display_attr = attr_name
            new_val = hs.get_attribute(attr_name)
            # Ensure value changed for tracked attribute
            if str(new_val) == str(old_val) and attr_name in FARMING_ATTRS:
                hs.gold += rng.randint(50, 200)
                hs.net_worth += rng.randint(50, 200)
                hs.last_hits += rng.randint(1, 5)
                new_val = hs.get_attribute(attr_name)
            display_val = hs.format_attribute(attr_name)
            mentioned = True

        location = rng.choice(self.names["locations"])
        cs = hs.last_hits
        comparison = rng.choice(["above average", "solid", "impressive", "on pace", "expected"])

        event_data = {
            "hero": hero,
            "time": int(game_time),
            "location": location,
            "attr_name": display_attr,
            "attr_value": display_val,
            "attr_label": display_attr.replace("_", " "),
            "cs": cs,
            "comparison": comparison,
            "role": hs.role,
            "pron_pos": "their",
            "pron_sub": "Their",
        }

        state_changes = [
            (hero, attr_name, str(old_val), str(new_val), mentioned),
        ]
        return event_data, state_changes

    def _handle_solo_kill(self, killer: str, hero_states: Dict[str, HeroState],
                          game_time: float, rng: random.Random,
                          config: DotaTrialConfig, tracked: Dict[str, str],
                          all_match_heroes: list) -> Tuple[dict, list]:
        ks = hero_states[killer]

        # Pick a victim from opposite team who is alive
        opposite_team = "dire" if ks.team == "radiant" else "radiant"
        victims = [h for h, _ in all_match_heroes
                    if hero_states[h].team == opposite_team and hero_states[h].alive
                    and h != killer]
        if not victims:
            victims = [h for h, _ in all_match_heroes if h != killer and hero_states[h].alive]
        if not victims:
            # Fallback: just use any other hero
            victims = [h for h, _ in all_match_heroes if h != killer]

        victim = rng.choice(victims)
        vs = hero_states[victim]

        killer_attr = tracked[killer]
        victim_attr = tracked.get(victim, tracked[killer])

        old_killer_val = ks.get_attribute(killer_attr)
        old_victim_val = vs.get_attribute(victim_attr)

        bounty = rng.randint(150, 600)

        # Update killer
        ks.kills += 1
        ks.kill_streak += 1
        ks.gold += bounty
        ks.net_worth += bounty
        ks.total_damage_dealt += rng.randint(300, 1500)

        # Update victim
        vs.deaths += 1
        gold_lost = rng.randint(50, 200)
        vs.gold = max(0, vs.gold - gold_lost)
        vs.total_damage_taken += rng.randint(300, 1500)
        vs.kill_streak = 0
        vs.alive = False
        respawn = int(5 + 3.8 * vs.level)
        vs.respawn_at = game_time + respawn / 60.0

        new_killer_val = ks.get_attribute(killer_attr)
        new_victim_val = vs.get_attribute(victim_attr)

        # Ensure values changed
        if str(new_killer_val) == str(old_killer_val):
            ks.gold += rng.randint(50, 150)
            ks.net_worth += rng.randint(50, 150)
            new_killer_val = ks.get_attribute(killer_attr)

        kill_verb = rng.choice(self.templates["kill_verbs"])
        location = rng.choice(self.names["locations"])

        event_data = {
            "killer": killer,
            "victim": victim,
            "time": int(game_time),
            "seconds": f"{rng.randint(0, 59):02d}",
            "location": location,
            "kill_verb": kill_verb,
            "bounty": bounty,
            "respawn": respawn,
            "attr_name": killer_attr,
            "attr_label": killer_attr.replace("_", " "),
            "killer_attr_value": ks.format_attribute(killer_attr),
            "victim_attr_value": vs.format_attribute(victim_attr),
            "pron_obj": "them",
            "pron_pos": "their",
        }

        state_changes = [
            (killer, killer_attr, str(old_killer_val), str(new_killer_val), True),
        ]
        # Only track victim if they are a tracked entity
        if victim in tracked:
            state_changes.append(
                (victim, victim_attr, str(old_victim_val), str(new_victim_val), True)
            )

        return event_data, state_changes

    def _handle_teamfight(self, primary_hero: str,
                          hero_states: Dict[str, HeroState],
                          game_time: float, rng: random.Random,
                          config: DotaTrialConfig, tracked: Dict[str, str],
                          all_match_heroes: list,
                          team1_name: str, team2_name: str) -> Tuple[dict, list]:
        hs = hero_states[primary_hero]

        # Decide winning team
        winning_team_side = rng.choice(["radiant", "dire"])
        winning_team_name = team1_name if winning_team_side == "radiant" else team2_name

        win_kills = rng.randint(2, 5)
        loss_kills = rng.randint(0, 2)

        # MVP is from winning team, worst is from losing team
        winners = [h for h, _ in all_match_heroes
                   if hero_states[h].team == winning_team_side and hero_states[h].alive]
        losers = [h for h, _ in all_match_heroes
                  if hero_states[h].team != winning_team_side and hero_states[h].alive]

        if not winners:
            winners = [h for h, _ in all_match_heroes if hero_states[h].team == winning_team_side]
        if not losers:
            losers = [h for h, _ in all_match_heroes if hero_states[h].team != winning_team_side]

        # Ensure primary hero is either MVP or worst for tracking
        if hs.team == winning_team_side:
            mvp = primary_hero
            worst = rng.choice(losers) if losers else primary_hero
        else:
            worst = primary_hero
            mvp = rng.choice(winners) if winners else primary_hero

        mvp_hs = hero_states[mvp]
        worst_hs = hero_states[worst]

        mvp_attr = tracked.get(mvp, tracked[primary_hero])
        worst_attr = tracked.get(worst, tracked[primary_hero])

        old_mvp_val = mvp_hs.get_attribute(mvp_attr)
        old_worst_val = worst_hs.get_attribute(worst_attr)

        mvp_kills = rng.randint(1, win_kills)

        # Update MVP
        mvp_hs.kills += mvp_kills
        mvp_hs.kill_streak += mvp_kills
        mvp_hs.gold += rng.randint(300, 1000)
        mvp_hs.net_worth += rng.randint(300, 1000)
        mvp_hs.total_damage_dealt += rng.randint(500, 3000)

        # Update worst
        worst_hs.deaths += 1
        gold_lost = rng.randint(100, 400)
        worst_hs.gold = max(0, worst_hs.gold - gold_lost)
        worst_hs.total_damage_taken += rng.randint(500, 2000)
        worst_hs.kill_streak = 0
        worst_hs.alive = False
        respawn = int(5 + 3.8 * worst_hs.level)
        worst_hs.respawn_at = game_time + respawn / 60.0

        # Give assists to other winners
        for h, _ in all_match_heroes:
            if hero_states[h].team == winning_team_side and h != mvp:
                hero_states[h].assists += rng.randint(1, 3)

        new_mvp_val = mvp_hs.get_attribute(mvp_attr)
        new_worst_val = worst_hs.get_attribute(worst_attr)

        if str(new_mvp_val) == str(old_mvp_val):
            mvp_hs.gold += rng.randint(50, 200)
            mvp_hs.net_worth += rng.randint(50, 200)
            new_mvp_val = mvp_hs.get_attribute(mvp_attr)

        location = rng.choice(self.names["locations"])
        fight_descriptions = [
            f"{mvp} led the charge with {mvp_kills} kills",
            f"A chaotic brawl saw {mvp} emerge with {mvp_kills} kills",
            f"{winning_team_name} executed a clean initiation, with {mvp} getting {mvp_kills} kills",
        ]

        event_data = {
            "location": location,
            "time": int(game_time),
            "winning_team": winning_team_name,
            "win_kills": win_kills,
            "loss_kills": loss_kills,
            "mvp": mvp,
            "mvp_kills": mvp_kills,
            "mvp_attr_value": mvp_hs.format_attribute(mvp_attr),
            "worst": worst,
            "worst_attr_value": worst_hs.format_attribute(worst_attr),
            "attr_name": mvp_attr,
            "attr_label": mvp_attr.replace("_", " "),
            "size": rng.randint(6, 10),
            "fight_description": rng.choice(fight_descriptions),
        }

        state_changes = []
        if mvp in tracked:
            state_changes.append(
                (mvp, mvp_attr, str(old_mvp_val), str(new_mvp_val), True)
            )
        if worst in tracked and worst != mvp:
            state_changes.append(
                (worst, worst_attr, str(old_worst_val), str(new_worst_val), True)
            )

        return event_data, state_changes

    def _handle_item_purchase(self, hero: str, hero_states: Dict[str, HeroState],
                              game_time: float, rng: random.Random,
                              config: DotaTrialConfig,
                              tracked: Dict[str, str]) -> Tuple[dict, list]:
        hs = hero_states[hero]
        attr_name = tracked[hero]
        old_val = hs.get_attribute(attr_name)

        # Pick next item from build
        build = self.role_item_builds[hs.role]
        current_items = set(hs.items)
        item_name = None
        item_tier = "core"

        for tier in ["early", "core", "luxury"]:
            for candidate in build[tier]:
                if candidate not in current_items and len(hs.items) < 6:
                    item_name = candidate
                    item_tier = tier
                    break
            if item_name:
                break

        if not item_name:
            # Pick random from appropriate tier
            phase = get_phase(game_time)
            tier = "core" if phase == "mid_game" else "luxury"
            tier_items = self.items_by_tier.get(tier, self.items_by_tier["core"])
            candidates = [i["name"] for i in tier_items if i["name"] not in current_items]
            if candidates:
                item_name = rng.choice(candidates)
            else:
                item_name = "Magic Wand"
            item_tier = tier

        cost = self._get_item_cost(item_name)

        if len(hs.items) < 6:
            hs.items.append(item_name)
        else:
            # Replace cheapest item
            cheapest_idx = min(range(len(hs.items)),
                               key=lambda i: self._get_item_cost(hs.items[i]))
            hs.items[cheapest_idx] = item_name

        hs.gold = max(0, hs.gold - cost)
        hs.net_worth += max(0, cost - rng.randint(0, 200))  # Some items upgrade from components

        new_val = hs.get_attribute(attr_name)
        if str(new_val) == str(old_val) and attr_name not in ("latest_item", "inventory_count"):
            hs.gold += rng.randint(50, 200)
            hs.net_worth += rng.randint(50, 200)
            new_val = hs.get_attribute(attr_name)

        significance_options = [
            f"A key power spike for the {hs.role}",
            "This opened up new options in fights",
            f"The timing was good for {int(game_time)} minutes",
        ]

        event_data = {
            "hero": hero,
            "item": item_name,
            "cost": cost,
            "time": int(game_time),
            "remaining_gold": max(0, hs.gold),
            "attr_name": attr_name,
            "attr_value": hs.format_attribute(attr_name),
            "attr_label": attr_name.replace("_", " "),
            "role": hs.role,
            "item_tier": item_tier,
            "item_significance": rng.choice(significance_options),
            "pron_obj": "them",
            "pron_pos": "their",
        }

        state_changes = [
            (hero, attr_name, str(old_val), str(new_val), True),
        ]
        return event_data, state_changes

    def _handle_tower_kill(self, primary_hero: str,
                           hero_states: Dict[str, HeroState],
                           game_time: float, rng: random.Random,
                           config: DotaTrialConfig, tracked: Dict[str, str],
                           towers_destroyed: dict,
                           all_match_heroes: list,
                           team1_name: str, team2_name: str) -> Tuple[dict, list]:
        hs = hero_states[primary_hero]
        attacking_team = hs.team
        defending_team = "dire" if attacking_team == "radiant" else "radiant"
        team_name = team1_name if attacking_team == "radiant" else team2_name

        # Find a valid tower to destroy
        lane = rng.choice(["top", "mid", "bot"])
        tier = 1
        for t in [1, 2, 3]:
            if (lane, t) not in towers_destroyed[defending_team]:
                tier = t
                break
        else:
            # All towers in this lane destroyed, try another lane
            for l in ["top", "mid", "bot"]:
                for t in [1, 2, 3]:
                    if (l, t) not in towers_destroyed[defending_team]:
                        lane, tier = l, t
                        break
                else:
                    continue
                break

        tower_gold = rng.choice([120, 150, 180, 200])
        attr_name = tracked[primary_hero]
        old_val = hs.get_attribute(attr_name)

        # Update hero states
        for h, _ in all_match_heroes:
            if hero_states[h].team == attacking_team:
                hero_states[h].gold += tower_gold
                hero_states[h].net_worth += tower_gold
        hs.tower_damage += rng.randint(200, 800)

        new_val = hs.get_attribute(attr_name)
        if str(new_val) == str(old_val):
            hs.gold += rng.randint(50, 150)
            hs.net_worth += rng.randint(50, 150)
            new_val = hs.get_attribute(attr_name)

        event_data = {
            "team": team_name,
            "lane": lane,
            "tier": tier,
            "time": int(game_time),
            "tower_gold": tower_gold,
            "hero": primary_hero,
            "attr_name": attr_name,
            "attr_value": hs.format_attribute(attr_name),
            "attr_label": attr_name.replace("_", " "),
            "defending_team": defending_team,
        }

        state_changes = [
            (primary_hero, attr_name, str(old_val), str(new_val), True),
        ]
        return event_data, state_changes

    def _handle_roshan_kill(self, primary_hero: str,
                            hero_states: Dict[str, HeroState],
                            game_time: float, rng: random.Random,
                            config: DotaTrialConfig, tracked: Dict[str, str],
                            roshan_count: int,
                            all_match_heroes: list,
                            team1_name: str, team2_name: str) -> Tuple[dict, list]:
        hs = hero_states[primary_hero]
        team_name = team1_name if hs.team == "radiant" else team2_name
        attr_name = tracked[primary_hero]
        old_val = hs.get_attribute(attr_name)

        rosh_gold = rng.randint(200, 400)
        for h, _ in all_match_heroes:
            if hero_states[h].team == hs.team:
                hero_states[h].gold += rosh_gold
                hero_states[h].net_worth += rosh_gold

        new_val = hs.get_attribute(attr_name)
        if str(new_val) == str(old_val):
            hs.gold += rng.randint(50, 200)
            hs.net_worth += rng.randint(50, 200)
            new_val = hs.get_attribute(attr_name)

        # Pick aegis carrier (pos1 from same team preferred)
        team_heroes = [h for h, _ in all_match_heroes if hero_states[h].team == hs.team]
        carrier = primary_hero
        for h in team_heroes:
            if hero_states[h].role == "pos1":
                carrier = h
                break

        count = roshan_count + 1
        nth = {1: "1st", 2: "2nd", 3: "3rd"}.get(count, f"{count}th")

        drops_options = [
            f"{carrier} claimed the Aegis of the Immortal",
            f"The Aegis went to {carrier}",
            f"{carrier} picked up the Aegis",
        ]
        if count >= 2:
            drops_options = [
                f"{carrier} took the Aegis and Cheese dropped",
                f"Aegis to {carrier}, with Cheese for the team",
            ]

        event_data = {
            "team": team_name,
            "time": int(game_time),
            "carrier": carrier,
            "hero": primary_hero,
            "attr_name": attr_name,
            "attr_value": hs.format_attribute(attr_name),
            "attr_label": attr_name.replace("_", " "),
            "nth": nth,
            "rosh_count": count,
            "drops": rng.choice(drops_options),
        }

        state_changes = [
            (primary_hero, attr_name, str(old_val), str(new_val), True),
        ]
        return event_data, state_changes

    def _handle_level_up(self, hero: str, hero_states: Dict[str, HeroState],
                         game_time: float, rng: random.Random,
                         config: DotaTrialConfig,
                         tracked: Dict[str, str]) -> Tuple[dict, list]:
        hs = hero_states[hero]
        old_val = hs.get_attribute("level")

        if hs.level < 30:
            hs.level += 1

        new_val = hs.get_attribute("level")

        # The tracked attribute for this hero
        attr_name = tracked[hero]
        # For level_up, the template uses attr_value which should be the level
        # but we track whatever the tracked attribute is
        # The templates say "{attr_value}" which is the level formatted as "level N"
        # We need to ensure the tracked value is what gets recorded

        # If level is the tracked attr, great. Otherwise we still track the level
        # in the narrative but record the actual tracked attr change.
        tracked_old = hs.get_attribute(attr_name)
        # Level up might also bump gold/xp
        hs.gold += rng.randint(0, 100)
        hs.net_worth += rng.randint(0, 100)
        tracked_new = hs.get_attribute(attr_name)

        if str(tracked_new) == str(tracked_old) and attr_name != "level":
            hs.gold += rng.randint(50, 200)
            hs.net_worth += rng.randint(50, 200)
            tracked_new = hs.get_attribute(attr_name)

        ability_notes = [
            "a new rank in their ultimate",
            "access to a key talent",
            "additional spell power",
            "a crucial talent tier",
            "the next skill point in their core ability",
        ]
        timing_options = ["ahead of schedule", "on time", "a bit late", "right on pace", "solid"]
        xp_sources = [
            "a jungle camp", "a creep wave", "a hero kill",
            "the lane", "stacked camps", "a teamfight",
        ]

        # For level_up templates, attr_value is the level
        level_str = f"level {hs.level}"

        event_data = {
            "hero": hero,
            "time": int(game_time),
            "attr_value": level_str,
            "attr_name": attr_name,
            "attr_label": attr_name.replace("_", " "),
            "role": hs.role,
            "ability_note": rng.choice(ability_notes),
            "timing_assessment": rng.choice(timing_options),
            "xp_source": rng.choice(xp_sources),
        }

        # For level_up, we track the level value in the narrative
        # but the entity_tracking records the tracked attribute
        state_changes = []
        if attr_name == "level":
            state_changes.append(
                (hero, "level", str(old_val), str(new_val), True)
            )
        else:
            # The template mentions level, but we need to track the actual tracked attr
            # We'll record the level mention but mark the tracked attr as mentioned too
            state_changes.append(
                (hero, attr_name, str(tracked_old), str(tracked_new), False)
            )
            # For the level mention itself - we record it but it won't be tracked
            # unless level is the tracked attr

        return event_data, state_changes

    def _handle_death_and_respawn(self, hero: str,
                                  hero_states: Dict[str, HeroState],
                                  game_time: float, rng: random.Random,
                                  config: DotaTrialConfig,
                                  tracked: Dict[str, str]) -> Tuple[dict, list]:
        hs = hero_states[hero]
        attr_name = tracked[hero]
        old_val = hs.get_attribute(attr_name)

        hs.deaths += 1
        gold_lost = rng.randint(100, 400)
        hs.gold = max(0, hs.gold - gold_lost)
        hs.kill_streak = 0
        hs.total_damage_taken += rng.randint(200, 1000)
        hs.alive = False
        respawn = int(5 + 3.8 * hs.level)
        hs.respawn_at = game_time + respawn / 60.0

        new_val = hs.get_attribute(attr_name)
        if str(new_val) == str(old_val):
            hs.gold = max(0, hs.gold - rng.randint(50, 150))
            new_val = hs.get_attribute(attr_name)

        location = rng.choice(self.names["locations"])
        kill_verb = rng.choice(self.templates["kill_verbs"])
        death_contexts = ["costly", "unfortunate", "critical", "punishing", "avoidable"]

        event_data = {
            "hero": hero,
            "time": int(game_time),
            "location": location,
            "kill_verb": kill_verb,
            "respawn": respawn,
            "gold_lost": gold_lost,
            "attr_name": attr_name,
            "attr_value": hs.format_attribute(attr_name),
            "attr_label": attr_name.replace("_", " "),
            "pron_pos": "their",
            "death_context": rng.choice(death_contexts),
        }

        state_changes = [
            (hero, attr_name, str(old_val), str(new_val), True),
        ]
        return event_data, state_changes

    def _handle_barracks_kill(self, primary_hero: str,
                              hero_states: Dict[str, HeroState],
                              game_time: float, rng: random.Random,
                              config: DotaTrialConfig, tracked: Dict[str, str],
                              towers_destroyed: dict,
                              barracks_destroyed: dict,
                              all_match_heroes: list,
                              team1_name: str, team2_name: str) -> Tuple[dict, list]:
        hs = hero_states[primary_hero]
        attacking_team = hs.team
        defending_team = "dire" if attacking_team == "radiant" else "radiant"
        team_name = team1_name if attacking_team == "radiant" else team2_name

        # Find a valid barracks
        lane = None
        rax_type = rng.choice(["melee", "ranged"])
        for l in ["top", "mid", "bot"]:
            if (l, 3) in towers_destroyed[defending_team]:
                if (l, rax_type) not in barracks_destroyed[defending_team]:
                    lane = l
                    break

        if not lane:
            # Fallback: pick any lane with T3 down
            for l in ["top", "mid", "bot"]:
                if (l, 3) in towers_destroyed[defending_team]:
                    for rt in ["melee", "ranged"]:
                        if (l, rt) not in barracks_destroyed[defending_team]:
                            lane = l
                            rax_type = rt
                            break
                if lane:
                    break

        if not lane:
            lane = "mid"  # Fallback

        rax_gold = rng.choice([150, 175, 200])
        attr_name = tracked[primary_hero]
        old_val = hs.get_attribute(attr_name)

        for h, _ in all_match_heroes:
            if hero_states[h].team == attacking_team:
                hero_states[h].gold += rax_gold
                hero_states[h].net_worth += rax_gold

        new_val = hs.get_attribute(attr_name)
        if str(new_val) == str(old_val):
            hs.gold += rng.randint(50, 150)
            hs.net_worth += rng.randint(50, 150)
            new_val = hs.get_attribute(attr_name)

        event_data = {
            "team": team_name,
            "lane": lane,
            "rax_type": rax_type,
            "time": int(game_time),
            "rax_gold": rax_gold,
            "hero": primary_hero,
            "attr_name": attr_name,
            "attr_value": hs.format_attribute(attr_name),
            "attr_label": attr_name.replace("_", " "),
            "defending_team": defending_team,
        }

        state_changes = [
            (primary_hero, attr_name, str(old_val), str(new_val), True),
        ]
        return event_data, state_changes

    # -------------------------------------------------------------------
    # Filler rendering
    # -------------------------------------------------------------------

    def _render_filler(self, template: str, hero: str,
                       hero_states: Dict[str, HeroState],
                       rng: random.Random, tracked: Dict[str, str],
                       all_match_heroes: list,
                       team1_name: str, team2_name: str) -> str:
        hs = hero_states[hero]
        team_name = team1_name if hs.team == "radiant" else team2_name
        other_team = team2_name if hs.team == "radiant" else team1_name
        location = rng.choice(self.names["locations"])

        # Pick a support hero
        supports = [h for h, _ in all_match_heroes
                    if hero_states[h].role in ("pos4", "pos5")]
        support = rng.choice(supports) if supports else hero

        # Pick a teammate
        teammates = [h for h, _ in all_match_heroes
                     if hero_states[h].team == hs.team and h != hero]
        teammate = rng.choice(teammates) if teammates else hero

        abilities = ["Blink Strike", "Black Hole", "Ravage", "Echo Slam",
                     "Reverse Polarity", "Chronosphere", "Dream Coil",
                     "Sonic Wave", "Finger of Death", "Laguna Blade"]
        ability_types = ["teamfight", "pickoff", "pushing", "defensive", "offensive"]
        rune_types = ["Double Damage", "Haste", "Regeneration", "Illusion", "Arcane"]
        lanes = ["top", "mid", "bot"]

        data = {
            "hero": hero,
            "support": support,
            "teammate": teammate,
            "team": team_name,
            "other_team": other_team,
            "location": location,
            "ability": rng.choice(abilities),
            "ability_type": rng.choice(ability_types),
            "pron_obj": "them",
            "pron_pos": "their",
            "rune_spot": rng.choice(["top power rune spot", "bottom power rune spot"]),
            "rune_type": rng.choice(rune_types),
            "lane": rng.choice(lanes),
        }

        return self.render_template(template, data)

    # -------------------------------------------------------------------
    # Header rendering
    # -------------------------------------------------------------------

    def _render_header(self, template: str, team1_name: str, team2_name: str,
                       radiant_heroes: list, dire_heroes: list,
                       tournament_name: str, config: DotaTrialConfig,
                       rng: random.Random) -> str:
        team1_hero_names = ", ".join(h for h, _ in radiant_heroes)
        team2_hero_names = ", ".join(h for h, _ in dire_heroes)
        first_pick = radiant_heroes[0][0] if radiant_heroes else "unknown"
        second_pick = dire_heroes[0][0] if dire_heroes else "unknown"
        pos1_hero = next((h for h, r in radiant_heroes if r == "pos1"), first_pick)

        adjectives = ["thrilling", "anticipated", "heated", "pivotal", "crucial"]
        draft_styles = ["aggressive", "defensive", "balanced", "greedy", "tempo"]
        paces = ["fast-paced", "methodical", "explosive", "controlled", "chaotic"]

        data = {
            "game_id": rng.randint(1, 5),
            "tournament": tournament_name,
            "team1": team1_name,
            "team2": team2_name,
            "team1_heroes": team1_hero_names,
            "team2_heroes": team2_hero_names,
            "first_pick": first_pick,
            "second_pick": second_pick,
            "pos1_hero": pos1_hero,
            "adjective": rng.choice(adjectives),
            "favored_team": rng.choice([team1_name, team2_name]),
            "draft_style": rng.choice(draft_styles),
            "expected_pace": rng.choice(paces),
        }

        return self.render_template(template, data)

    # -------------------------------------------------------------------
    # Template selection
    # -------------------------------------------------------------------

    def _get_templates_for_event(self, event_type: str) -> list:
        mapping = {
            "farming_update": "farming_templates",
            "solo_kill": "solo_kill_templates",
            "teamfight": "teamfight_templates",
            "item_purchase": "item_purchase_templates",
            "tower_kill": "tower_kill_templates",
            "roshan_kill": "roshan_templates",
            "level_up": "level_up_templates",
            "death_and_respawn": "death_respawn_templates",
            "barracks_kill": "barracks_kill_templates",
        }
        key = mapping.get(event_type, "farming_templates")
        return self.templates[key]

    # -------------------------------------------------------------------
    # Main generation flow
    # -------------------------------------------------------------------

    def generate_trial(self, num_keys: int, num_updates: int,
                       condition: str, seed: int, **kwargs) -> dict:
        """Generate a single Dota 2 narrative interference trial.

        Args:
            num_keys: number of heroes to track (2-10)
            num_updates: target tracked-value mentions per hero (3-50)
            condition: "RI" or "PI"
            seed: random seed for full reproducibility
            **kwargs: optional config overrides. Any of:
                tracked_attribute, attribute_mode, filler_budget,
                voice, match_scope, game_archetype, queried_hero_idx
                If not provided, auto-sampled from seed.

        Returns:
            dict matching the output JSON schema
        """
        rng = random.Random(seed)

        # Step 2: Auto-sample config, then apply overrides
        config = self._auto_config(num_keys, num_updates, condition, seed, rng)
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Step 3: Pick heroes, assign teams and roles
        all_heroes = list(self.hero_roles.keys())
        rng.shuffle(all_heroes)

        radiant_count = (num_keys + 1) // 2
        dire_count = num_keys - radiant_count

        roles_needed_radiant = ["pos1", "pos2", "pos3", "pos4", "pos5"][:radiant_count]
        roles_needed_dire = ["pos1", "pos2", "pos3", "pos4", "pos5"][:dire_count]

        radiant_heroes = []
        dire_heroes = []
        used_heroes = set()

        for role in roles_needed_radiant:
            candidates = [h for h in self.role_pool[role] if h not in used_heroes]
            hero = rng.choice(candidates)
            radiant_heroes.append((hero, role))
            used_heroes.add(hero)

        for role in roles_needed_dire:
            candidates = [h for h in self.role_pool[role] if h not in used_heroes]
            hero = rng.choice(candidates)
            dire_heroes.append((hero, role))
            used_heroes.add(hero)

        all_match_heroes = radiant_heroes + dire_heroes

        # Step 4: Assign tracked attributes
        tracked: Dict[str, str] = {}
        if config.attribute_mode == "same":
            for hero, role in all_match_heroes:
                tracked[hero] = config.tracked_attribute
        else:
            for hero, role in all_match_heroes:
                tracked[hero] = rng.choice(self.role_attribute_affinity[role])

        # Step 5: Determine start_time
        if config.match_scope == "window":
            start_time = determine_start_time(num_updates, rng)
        else:
            start_time = 0.0
        config.start_time = start_time

        # Step 6: Initialize hero states
        if config.game_archetype == "stomp":
            config.stomp_winner = rng.choice(["radiant", "dire"])
        if config.game_archetype == "comeback":
            config.early_leader = rng.choice(["radiant", "dire"])

        hero_states: Dict[str, HeroState] = {}
        for hero, role in all_match_heroes:
            team = "radiant" if (hero, role) in radiant_heroes else "dire"
            hero_states[hero] = self._initialize_hero_state(
                hero, role, start_time, config.game_archetype, team, config, rng
            )

        # Step 7: Pick team and tournament names
        team1_name, team2_name = rng.sample(self.names["team_names"], 2)
        tournament_name = rng.choice(self.names["tournament_names"])

        # Step 8: Generate event sequence
        game_time = start_time
        event_log = []
        full_state_log = []
        entity_tracking = {f"{hero} / {tracked[hero]}": [] for hero, _ in all_match_heroes}
        mention_counts = {hero: 0 for hero, _ in all_match_heroes}
        total_target_mentions = num_keys * num_updates

        last_updated_hero = None
        towers_destroyed = {"radiant": set(), "dire": set()}
        barracks_destroyed = {"radiant": set(), "dire": set()}
        roshan_kill_times: List[float] = []
        roshan_count = 0

        max_iterations = total_target_mentions * 5  # Safety valve
        iteration = 0

        while sum(mention_counts.values()) < total_target_mentions and game_time < 60.0:
            iteration += 1
            if iteration > max_iterations:
                break

            phase = get_phase(game_time)
            weights = get_event_weights(phase, config.game_archetype)

            # Step 8a: Filter impossible events
            possible_weights = dict(weights)

            if phase == "laning":
                possible_weights.pop("barracks_kill", None)
                possible_weights.pop("teamfight", None)
                possible_weights.pop("roshan_kill", None)

            if roshan_kill_times and (game_time - roshan_kill_times[-1]) < 8:
                possible_weights.pop("roshan_kill", None)

            can_kill_barracks = False
            for team_key in ["radiant", "dire"]:
                for lane in ["top", "mid", "bot"]:
                    if (lane, 3) in towers_destroyed[team_key]:
                        can_kill_barracks = True
            if not can_kill_barracks:
                possible_weights.pop("barracks_kill", None)

            if all(hero_states[h].level >= 30 for h, _ in all_match_heroes):
                possible_weights.pop("level_up", None)

            # Remove zero-weight events
            possible_weights = {k: v for k, v in possible_weights.items() if v > 0}

            total_w = sum(possible_weights.values())
            if total_w == 0:
                break

            # Step 8b: Pick event type
            event_type = self.weighted_random_choice(possible_weights, rng)

            # Step 8c: Pick involved heroes
            alive_heroes = [h for h, _ in all_match_heroes if hero_states[h].alive]
            if not alive_heroes:
                # Force respawn all
                for h, _ in all_match_heroes:
                    hero_states[h].alive = True
                alive_heroes = [h for h, _ in all_match_heroes]

            candidates = [h for h in alive_heroes if h != last_updated_hero]
            if not candidates:
                candidates = alive_heroes

            # Strongly prefer heroes with fewer mentions (tighter balancing)
            min_mentions = min(mention_counts[h] for h in candidates)
            # If any hero has <2 mentions, force them as priority
            urgent = [h for h in candidates if mention_counts[h] < 2]
            if urgent:
                candidates = urgent
            else:
                priority_candidates = [h for h in candidates if mention_counts[h] <= min_mentions + 1]
                if priority_candidates:
                    candidates = priority_candidates

            primary_hero = rng.choice(candidates)

            # Update computed attributes
            self._update_computed_attrs(hero_states, game_time, all_match_heroes, rng)

            # Step 8d: Execute event
            event_data = {}
            state_changes = []

            if event_type == "farming_update":
                event_data, state_changes = self._handle_farming_update(
                    primary_hero, hero_states, game_time, rng, config, tracked)
            elif event_type == "solo_kill":
                event_data, state_changes = self._handle_solo_kill(
                    primary_hero, hero_states, game_time, rng, config, tracked,
                    all_match_heroes)
            elif event_type == "teamfight":
                event_data, state_changes = self._handle_teamfight(
                    primary_hero, hero_states, game_time, rng, config, tracked,
                    all_match_heroes, team1_name, team2_name)
            elif event_type == "item_purchase":
                event_data, state_changes = self._handle_item_purchase(
                    primary_hero, hero_states, game_time, rng, config, tracked)
            elif event_type == "tower_kill":
                event_data, state_changes = self._handle_tower_kill(
                    primary_hero, hero_states, game_time, rng, config, tracked,
                    towers_destroyed, all_match_heroes, team1_name, team2_name)
            elif event_type == "roshan_kill":
                event_data, state_changes = self._handle_roshan_kill(
                    primary_hero, hero_states, game_time, rng, config, tracked,
                    roshan_count, all_match_heroes, team1_name, team2_name)
            elif event_type == "level_up":
                event_data, state_changes = self._handle_level_up(
                    primary_hero, hero_states, game_time, rng, config, tracked)
            elif event_type == "death_and_respawn":
                event_data, state_changes = self._handle_death_and_respawn(
                    primary_hero, hero_states, game_time, rng, config, tracked)
            elif event_type == "barracks_kill":
                event_data, state_changes = self._handle_barracks_kill(
                    primary_hero, hero_states, game_time, rng, config, tracked,
                    towers_destroyed, barracks_destroyed,
                    all_match_heroes, team1_name, team2_name)

            # Step 8e: Record state changes
            for hero_name, attr, old_val, new_val, mentioned in state_changes:
                full_state_log.append({
                    "time": round(game_time, 1),
                    "event": event_type,
                    "entity": hero_name,
                    "attribute": attr,
                    "old_value": str(old_val),
                    "new_value": str(new_val),
                    "mentioned_in_narrative": mentioned,
                })
                if mentioned and hero_name in tracked:
                    key = f"{hero_name} / {tracked[hero_name]}"
                    if key in entity_tracking:
                        if not entity_tracking[key] or entity_tracking[key][-1] != str(new_val):
                            entity_tracking[key].append(str(new_val))
                            mention_counts[hero_name] += 1

            event_log.append({
                "type": event_type,
                "time": round(game_time, 1),
                "data": event_data,
            })

            last_updated_hero = primary_hero

            # Step 8f: Update game-level state
            if event_type == "tower_kill":
                lane = event_data.get("lane")
                tier = event_data.get("tier")
                defending_team = event_data.get("defending_team")
                if lane and tier and defending_team:
                    towers_destroyed[defending_team].add((lane, tier))
            elif event_type == "barracks_kill":
                lane = event_data.get("lane")
                rax_type = event_data.get("rax_type")
                defending_team = event_data.get("defending_team")
                if lane and rax_type and defending_team:
                    barracks_destroyed[defending_team].add((lane, rax_type))
            elif event_type == "roshan_kill":
                roshan_kill_times.append(game_time)
                roshan_count += 1

            # Step 8g: Advance game_time
            time_range = EVENT_TIME_ADVANCE[event_type][phase]
            dt = rng.uniform(time_range[0], time_range[1])
            dt = max(0.3, min(3.0, dt))
            game_time += dt
            game_time = round(game_time, 1)

            # Step 8h: Handle respawns
            for hero_name, _ in all_match_heroes:
                h_state = hero_states[hero_name]
                if not h_state.alive and game_time >= h_state.respawn_at:
                    h_state.alive = True

        # Step 8i: Enforce minimum 2 mentions per entity
        # If any entity has <2 tracked values, force appropriate events
        COMBAT_ONLY_ATTRS = {"kills", "deaths", "assists", "kda_string", "kill_streak",
                             "total_damage_taken"}
        min_mentions_target = 2
        for force_pass in range(5):  # up to 5 passes
            under_mentioned = [
                h for h, _ in all_match_heroes
                if mention_counts[h] < min_mentions_target
            ]
            if not under_mentioned:
                break
            for hero_name in under_mentioned:
                if game_time >= 60.0:
                    break
                hero_states[hero_name].alive = True
                self._update_computed_attrs(hero_states, game_time, all_match_heroes, rng)
                attr_name = tracked[hero_name]

                # Pick the right event type based on tracked attribute
                if attr_name in COMBAT_ONLY_ATTRS:
                    # Use solo_kill to change combat stats
                    event_type = "solo_kill"
                    event_data, state_changes = self._handle_solo_kill(
                        hero_name, hero_states, game_time, rng, config, tracked,
                        all_match_heroes)
                else:
                    # Use farming for gold/CS/level etc
                    event_type = "farming_update"
                    event_data, state_changes = self._handle_farming_update(
                        hero_name, hero_states, game_time, rng, config, tracked)

                for hn, attr, old_val, new_val, mentioned in state_changes:
                    full_state_log.append({
                        "time": round(game_time, 1),
                        "event": event_type,
                        "entity": hn,
                        "attribute": attr,
                        "old_value": str(old_val),
                        "new_value": str(new_val),
                        "mentioned_in_narrative": mentioned,
                    })
                    if mentioned and hn in tracked:
                        key = f"{hn} / {tracked[hn]}"
                        if key in entity_tracking:
                            if not entity_tracking[key] or entity_tracking[key][-1] != str(new_val):
                                entity_tracking[key].append(str(new_val))
                                mention_counts[hn] += 1
                event_log.append({
                    "type": event_type,
                    "time": round(game_time, 1),
                    "data": event_data,
                })
                game_time += rng.uniform(0.5, 1.5)
                game_time = round(game_time, 1)

        # Step 9: Render narrative
        narrative_parts = []

        # 9a: Match header
        header_template = rng.choice(self.templates["header_templates"])
        header = self._render_header(header_template, team1_name, team2_name,
                                     radiant_heroes, dire_heroes,
                                     tournament_name, config, rng)
        narrative_parts.append(header)
        narrative_parts.append("")

        # 9b: Render events with filler and transitions
        current_phase = get_phase(event_log[0]["time"]) if event_log else "laning"
        filler_cfg = FILLER_CONFIG[config.filler_budget]

        phase_descriptions = {
            "mid_game": "the mid-game rotations and objective play",
            "late_game": "the late-game high-ground sieges and decisive teamfights",
        }

        for i, event in enumerate(event_log):
            event_phase = get_phase(event["time"])
            if event_phase != current_phase:
                transition = rng.choice(self.templates["segment_transition_templates"])
                narrative_parts.append(self.render_template(transition, {
                    "time": int(event["time"]),
                    "old_phase": current_phase.replace("_", " "),
                    "phase_description": phase_descriptions.get(event_phase, "a new phase"),
                }))
                narrative_parts.append("")
                current_phase = event_phase

            templates = self._get_templates_for_event(event["type"])
            template = rng.choice(templates)
            rendered = self.render_template(template, event["data"])
            narrative_parts.append(rendered)

            # Maybe insert filler
            if rng.random() < filler_cfg["insert_probability"]:
                n_sentences = filler_cfg["sentences_per_insertion"]
                if isinstance(n_sentences, tuple):
                    n_sentences = rng.randint(n_sentences[0], n_sentences[1])
                filler_templates = rng.sample(
                    self.templates["filler_templates"],
                    min(n_sentences, len(self.templates["filler_templates"]))
                )
                for ft in filler_templates:
                    filler_hero = rng.choice([h for h, _ in all_match_heroes])
                    filler_rendered = self._render_filler(
                        ft, filler_hero, hero_states, rng, tracked,
                        all_match_heroes, team1_name, team2_name)
                    narrative_parts.append(filler_rendered)

        # Step 10: Generate questions (before finalizing narrative, in case we need a bonus event)
        queried_hero_idx = min(config.queried_hero_idx, len(all_match_heroes) - 1)
        queried_hero = all_match_heroes[queried_hero_idx][0]
        queried_attr = tracked[queried_hero]
        tracking_key = f"{queried_hero} / {queried_attr}"
        values = entity_tracking[tracking_key]

        # Ensure RI != PI: find a hero with distinct first/last values
        if len(values) < 2 or values[0] == values[-1]:
            for alt_hero, _ in all_match_heroes:
                alt_key = f"{alt_hero} / {tracked[alt_hero]}"
                alt_values = entity_tracking[alt_key]
                if len(alt_values) >= 2 and alt_values[0] != alt_values[-1]:
                    queried_hero = alt_hero
                    queried_attr = tracked[alt_hero]
                    tracking_key = alt_key
                    values = alt_values
                    break

        # Last resort: if still no valid hero, force a bonus event to create divergence
        if len(values) < 2 or values[0] == values[-1]:
            # Pick the hero with the most tracked values
            best_hero = max(
                [h for h, _ in all_match_heroes],
                key=lambda h: len(entity_tracking[f"{h} / {tracked[h]}"])
            )
            best_attr = tracked[best_hero]
            best_key = f"{best_hero} / {best_attr}"
            hs = hero_states[best_hero]

            # Force a state change
            old_formatted = hs.format_attribute(best_attr)
            if best_attr in ("gold", "net_worth", "gpm", "xpm",
                             "total_damage_dealt", "total_damage_taken",
                             "last_hits", "tower_damage", "healing_done"):
                setattr(hs, best_attr if hasattr(hs, best_attr) else "gold",
                        hs.get_attribute(best_attr) + rng.randint(100, 500))
            elif best_attr in ("kills", "deaths", "assists", "level",
                               "denies", "wards_placed", "camps_stacked",
                               "runes_collected", "kill_streak"):
                cur = hs.get_attribute(best_attr)
                setattr(hs, best_attr, cur + 1)
            new_formatted = hs.format_attribute(best_attr)

            # Add a bonus narrative sentence and record the change
            bonus_text = f" By the end, {best_hero}'s {best_attr} had shifted to {new_formatted}."
            narrative_parts.append(bonus_text)
            if not entity_tracking[best_key] or entity_tracking[best_key][-1] != new_formatted:
                entity_tracking[best_key].append(new_formatted)
                mention_counts[best_hero] += 1

            queried_hero = best_hero
            queried_attr = best_attr
            tracking_key = best_key
            values = entity_tracking[best_key]

        ri_answer = values[0] if values else "unknown"
        pi_answer = values[-1] if values else "unknown"

        RI_QUESTION_TEMPLATES = [
            f"What was {queried_hero}'s {queried_attr} when first mentioned in this match?",
            f"What was {queried_hero}'s {queried_attr} at the first reference in the game?",
            f"At {queried_hero}'s first appearance in the narrative, what was the {queried_attr}?",
        ]
        PI_QUESTION_TEMPLATES = [
            f"What was {queried_hero}'s {queried_attr} at the most recent update?",
            f"In the last mention of {queried_hero}'s {queried_attr}, what was the value?",
            f"What was {queried_hero}'s final recorded {queried_attr}?",
        ]

        questions = {
            "RI": {
                "question": rng.choice(RI_QUESTION_TEMPLATES),
                "expected_answer": ri_answer,
                "target_entity": queried_hero,
                "target_attribute": queried_attr,
            },
            "PI": {
                "question": rng.choice(PI_QUESTION_TEMPLATES),
                "expected_answer": pi_answer,
                "target_entity": queried_hero,
                "target_attribute": queried_attr,
            },
        }

        # Finalize narrative (after potential bonus events)
        narrative = " ".join(p for p in narrative_parts if p != "")
        final_parts = []
        for p in narrative_parts:
            if p == "":
                if final_parts and not final_parts[-1].endswith("\n"):
                    final_parts.append("\n")
            else:
                final_parts.append(p)
        narrative = " ".join(final_parts).replace(" \n ", "\n\n")

        # Step 12: Assemble and return
        return {
            "id": f"dota2_{seed:06d}",
            "domain": "dota2",
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
                "tracked_attribute": config.tracked_attribute,
                "attribute_mode": config.attribute_mode,
                "filler_budget": config.filler_budget,
                "voice": config.voice,
                "match_scope": config.match_scope,
                "game_archetype": config.game_archetype,
                "queried_hero_idx": config.queried_hero_idx,
                "start_time": start_time,
                "team1": team1_name,
                "team2": team2_name,
                "tournament": tournament_name,
                "radiant_heroes": [[h, r] for h, r in radiant_heroes],
                "dire_heroes": [[h, r] for h, r in dire_heroes],
            },
        }

    # -------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------

    def validate_trial(self, trial: dict) -> bool:
        narrative = trial["narrative"]
        entity_tracking = trial["entity_tracking"]
        questions = trial["questions"]

        # Check that all tracked values appear in narrative
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
            f"Need at least 2 mentions for queried entity '{tracking_key}', got {len(values)}"
        )

        # Check RI answer is first value, PI answer is last
        assert values[0] == ri_answer, (
            f"RI answer should be first value '{values[0]}', got '{ri_answer}'"
        )
        assert values[-1] == pi_answer, (
            f"PI answer should be last value '{values[-1]}', got '{pi_answer}'"
        )

        return True
