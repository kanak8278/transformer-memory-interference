# Dota 2 Narrative Interference Generator — Complete Code Specification

**Purpose**: This document is a zero-ambiguity implementation spec. A developer reads this and writes code with no questions.

**Date**: 2026-02-27

---

## 1. HERO DATABASE

```python
# All 127 heroes mapped to their typical role(s).
# Roles: "pos1" (carry), "pos2" (mid), "pos3" (offlane), "pos4" (soft support), "pos5" (hard support)
# Heroes with multiple roles are listed with all viable roles; first role is primary.

HERO_ROLES = {
    # --- Position 1 (Hard Carry) primary ---
    "Anti-Mage": ["pos1"],
    "Arc Warden": ["pos1"],
    "Chaos Knight": ["pos1", "pos3"],
    "Clinkz": ["pos1", "pos3"],
    "Drow Ranger": ["pos1"],
    "Faceless Void": ["pos1", "pos3"],
    "Gyrocopter": ["pos1", "pos2"],
    "Juggernaut": ["pos1"],
    "Lifestealer": ["pos1"],
    "Luna": ["pos1"],
    "Medusa": ["pos1"],
    "Monkey King": ["pos1", "pos2"],
    "Morphling": ["pos1"],
    "Naga Siren": ["pos1"],
    "Phantom Assassin": ["pos1"],
    "Phantom Lancer": ["pos1"],
    "Slark": ["pos1"],
    "Spectre": ["pos1"],
    "Sven": ["pos1"],
    "Terrorblade": ["pos1"],
    "Troll Warlord": ["pos1"],
    "Ursa": ["pos1", "pos3"],
    "Wraith King": ["pos1", "pos3"],
    "Lone Druid": ["pos1", "pos3"],
    "Bloodseeker": ["pos1", "pos2"],
    "Muerta": ["pos1", "pos2"],
    "Weaver": ["pos1", "pos3"],

    # --- Position 2 (Midlane) primary ---
    "Ember Spirit": ["pos2"],
    "Invoker": ["pos2"],
    "Leshrac": ["pos2", "pos3"],
    "Lina": ["pos2", "pos4"],
    "Outworld Destroyer": ["pos2"],
    "Puck": ["pos2"],
    "Queen of Pain": ["pos2"],
    "Razor": ["pos2", "pos3"],
    "Shadow Fiend": ["pos2"],
    "Sniper": ["pos2", "pos1"],
    "Storm Spirit": ["pos2"],
    "Templar Assassin": ["pos2"],
    "Tinker": ["pos2"],
    "Void Spirit": ["pos2"],
    "Viper": ["pos2", "pos3"],
    "Zeus": ["pos2", "pos4"],
    "Death Prophet": ["pos2", "pos3"],
    "Dragon Knight": ["pos2", "pos3"],
    "Huskar": ["pos2"],
    "Kez": ["pos2", "pos3"],
    "Necrophos": ["pos2", "pos3"],
    "Pugna": ["pos2", "pos4"],
    "Windranger": ["pos2", "pos3", "pos4"],
    "Meepo": ["pos2"],

    # --- Position 3 (Offlane) primary ---
    "Axe": ["pos3"],
    "Beastmaster": ["pos3"],
    "Brewmaster": ["pos3"],
    "Bristleback": ["pos3"],
    "Broodmother": ["pos3"],
    "Centaur Warrunner": ["pos3"],
    "Doom": ["pos3"],
    "Earth Spirit": ["pos3", "pos4"],
    "Earthshaker": ["pos3", "pos4"],
    "Elder Titan": ["pos3", "pos4"],
    "Enigma": ["pos3", "pos4"],
    "Kunkka": ["pos3", "pos2"],
    "Legion Commander": ["pos3"],
    "Lycan": ["pos3"],
    "Magnus": ["pos3"],
    "Mars": ["pos3"],
    "Night Stalker": ["pos3"],
    "Omniknight": ["pos3", "pos5"],
    "Pangolier": ["pos3"],
    "Primal Beast": ["pos3"],
    "Sand King": ["pos3", "pos4"],
    "Slardar": ["pos3"],
    "Tidehunter": ["pos3"],
    "Timbersaw": ["pos3"],
    "Tiny": ["pos3", "pos2"],
    "Underlord": ["pos3"],
    "Undying": ["pos3", "pos5"],
    "Batrider": ["pos3"],
    "Dark Seer": ["pos3"],
    "Dawnbreaker": ["pos3", "pos5"],
    "Nature's Prophet": ["pos3"],
    "Spirit Breaker": ["pos3", "pos4"],
    "Largo": ["pos3"],

    # --- Position 4 (Soft Support) primary ---
    "Bounty Hunter": ["pos4"],
    "Clockwerk": ["pos4", "pos3"],
    "Dark Willow": ["pos4", "pos5"],
    "Disruptor": ["pos4", "pos5"],
    "Grimstroke": ["pos4", "pos5"],
    "Hoodwink": ["pos4"],
    "Io": ["pos4", "pos5"],
    "Jakiro": ["pos4", "pos5"],
    "Marci": ["pos4", "pos3"],
    "Mirana": ["pos4", "pos2"],
    "Nyx Assassin": ["pos4"],
    "Phoenix": ["pos4", "pos3"],
    "Ringmaster": ["pos4", "pos5"],
    "Rubick": ["pos4"],
    "Shadow Demon": ["pos4", "pos5"],
    "Skywrath Mage": ["pos4"],
    "Snapfire": ["pos4", "pos5"],
    "Techies": ["pos4"],
    "Tusk": ["pos4"],
    "Vengeful Spirit": ["pos4", "pos5"],
    "Venomancer": ["pos4", "pos3"],
    "Visage": ["pos4", "pos3"],

    # --- Position 5 (Hard Support) primary ---
    "Abaddon": ["pos5", "pos3"],
    "Ancient Apparition": ["pos5"],
    "Bane": ["pos5"],
    "Chen": ["pos5"],
    "Crystal Maiden": ["pos5"],
    "Dazzle": ["pos5"],
    "Enchantress": ["pos5", "pos3"],
    "Keeper of the Light": ["pos5"],
    "Lich": ["pos5"],
    "Lion": ["pos5"],
    "Ogre Magi": ["pos5"],
    "Oracle": ["pos5"],
    "Shadow Shaman": ["pos5"],
    "Silencer": ["pos5", "pos4"],
    "Treant Protector": ["pos5"],
    "Warlock": ["pos5"],
    "Witch Doctor": ["pos5"],
    "Winter Wyvern": ["pos5"],
    "Pudge": ["pos5", "pos4"],
    "Riki": ["pos5", "pos4"],
    "Alchemist": ["pos1", "pos3"],
}
# Total: 127 heroes
```

**Helper**: derive role pools from this dict at init time:

```python
ROLE_POOL = {"pos1": [], "pos2": [], "pos3": [], "pos4": [], "pos5": []}
for hero, roles in HERO_ROLES.items():
    for role in roles:
        ROLE_POOL[role].append(hero)
```

---

## 2. ITEM DATABASE

### 2.1 Items by tier

```python
ITEMS = {
    "starting": [
        {"name": "Tango", "cost": 90},
        {"name": "Healing Salve", "cost": 100},
        {"name": "Clarity", "cost": 50},
        {"name": "Faerie Fire", "cost": 70},
        {"name": "Iron Branch", "cost": 50},
        {"name": "Circlet", "cost": 155},
        {"name": "Slippers of Agility", "cost": 140},
        {"name": "Gauntlets of Strength", "cost": 140},
        {"name": "Mantle of Intelligence", "cost": 140},
        {"name": "Band of Elvenskin", "cost": 450},
        {"name": "Quelling Blade", "cost": 100},
        {"name": "Ring of Protection", "cost": 175},
        {"name": "Magic Stick", "cost": 200},
        {"name": "Boots of Speed", "cost": 500},
        {"name": "Wind Lace", "cost": 250},
        {"name": "Orb of Venom", "cost": 275},
    ],
    "early": [
        {"name": "Magic Wand", "cost": 450},
        {"name": "Bracer", "cost": 505},
        {"name": "Wraith Band", "cost": 505},
        {"name": "Null Talisman", "cost": 505},
        {"name": "Urn of Shadows", "cost": 880},
        {"name": "Medallion of Courage", "cost": 1025},
        {"name": "Power Treads", "cost": 1400},
        {"name": "Phase Boots", "cost": 1480},
        {"name": "Arcane Boots", "cost": 1300},
        {"name": "Hand of Midas", "cost": 2200},
        {"name": "Vanguard", "cost": 1825},
        {"name": "Hood of Defiance", "cost": 1500},
        {"name": "Drum of Endurance", "cost": 1700},
        {"name": "Vladmir's Offering", "cost": 2450},
    ],
    "core": [
        {"name": "Blink Dagger", "cost": 2250},
        {"name": "Force Staff", "cost": 2200},
        {"name": "Eul's Scepter", "cost": 2725},
        {"name": "Shadow Blade", "cost": 3000},
        {"name": "Desolator", "cost": 3500},
        {"name": "Maelstrom", "cost": 2700},
        {"name": "Diffusal Blade", "cost": 2500},
        {"name": "Orchid Malevolence", "cost": 3475},
        {"name": "Sange and Yasha", "cost": 4100},
        {"name": "Kaya and Sange", "cost": 4100},
        {"name": "Yasha and Kaya", "cost": 4100},
        {"name": "Black King Bar", "cost": 4050},
        {"name": "Battle Fury", "cost": 3900},
        {"name": "Aghanim's Scepter", "cost": 4200},
        {"name": "Manta Style", "cost": 4650},
        {"name": "Pipe of Insight", "cost": 3475},
        {"name": "Crimson Guard", "cost": 3600},
        {"name": "Lotus Orb", "cost": 3850},
        {"name": "Radiance", "cost": 4700},
    ],
    "luxury": [
        {"name": "Daedalus", "cost": 5100},
        {"name": "Butterfly", "cost": 5450},
        {"name": "Eye of Skadi", "cost": 5300},
        {"name": "Satanic", "cost": 5050},
        {"name": "Heart of Tarrasque", "cost": 5000},
        {"name": "Assault Cuirass", "cost": 5250},
        {"name": "Monkey King Bar", "cost": 4975},
        {"name": "Scythe of Vyse", "cost": 5675},
        {"name": "Refresher Orb", "cost": 5200},
        {"name": "Divine Rapier", "cost": 5950},
        {"name": "Abyssal Blade", "cost": 6250},
        {"name": "Overwhelming Blink", "cost": 6800},
        {"name": "Swift Blink", "cost": 6800},
        {"name": "Arcane Blink", "cost": 6800},
    ],
}
```

### 2.2 Role-specific item build progressions

```python
ROLE_ITEM_BUILDS = {
    "pos1": {
        "starting": ["Tango", "Quelling Blade", "Iron Branch", "Iron Branch", "Slippers of Agility"],
        "early": ["Power Treads", "Magic Wand", "Wraith Band"],
        "core": ["Battle Fury", "Manta Style", "Black King Bar"],
        "luxury": ["Butterfly", "Abyssal Blade", "Satanic"],
    },
    "pos2": {
        "starting": ["Tango", "Faerie Fire", "Iron Branch", "Iron Branch", "Circlet"],
        "early": ["Power Treads", "Magic Wand", "Null Talisman"],
        "core": ["Black King Bar", "Aghanim's Scepter", "Orchid Malevolence"],
        "luxury": ["Scythe of Vyse", "Refresher Orb", "Eye of Skadi"],
    },
    "pos3": {
        "starting": ["Tango", "Gauntlets of Strength", "Ring of Protection", "Iron Branch", "Iron Branch"],
        "early": ["Phase Boots", "Magic Wand", "Bracer", "Vanguard"],
        "core": ["Blink Dagger", "Black King Bar", "Pipe of Insight", "Crimson Guard"],
        "luxury": ["Assault Cuirass", "Heart of Tarrasque", "Overwhelming Blink"],
    },
    "pos4": {
        "starting": ["Tango", "Wind Lace", "Clarity", "Clarity", "Iron Branch", "Iron Branch"],
        "early": ["Arcane Boots", "Magic Wand", "Urn of Shadows"],
        "core": ["Blink Dagger", "Force Staff", "Eul's Scepter", "Aghanim's Scepter"],
        "luxury": ["Scythe of Vyse", "Refresher Orb", "Lotus Orb"],
    },
    "pos5": {
        "starting": ["Tango", "Healing Salve", "Clarity", "Clarity", "Iron Branch", "Iron Branch"],
        "early": ["Arcane Boots", "Magic Wand", "Medallion of Courage"],
        "core": ["Force Staff", "Blink Dagger", "Aghanim's Scepter"],
        "luxury": ["Scythe of Vyse", "Lotus Orb", "Refresher Orb"],
    },
}
```

**Usage**: When generating an item_purchase event, pick the next item from the role's build in order. If the hero already has all items from the current tier, advance to the next tier. If all items from the build are purchased, randomly pick from remaining items in the appropriate tier by cost.

---

## 3. COMPLETE TRACKABLE ATTRIBUTES LIST

25 trackable attributes. Each row: name, Python type, value range, format string for embedding in narrative, example value.

```python
TRACKABLE_ATTRIBUTES = [
    {
        "name": "gold",
        "type": "int",
        "range": [0, 40000],
        "format": "{value:,} gold",
        "example": "8,400 gold",
        "update_events": ["farming_update", "solo_kill", "teamfight", "item_purchase", "tower_kill", "roshan_kill", "death_and_respawn"],
        "direction": "volatile",  # goes up and down
    },
    {
        "name": "net_worth",
        "type": "int",
        "range": [0, 50000],
        "format": "{value:,} net worth",
        "example": "14,200 net worth",
        "update_events": ["farming_update", "solo_kill", "teamfight", "item_purchase", "tower_kill", "roshan_kill", "death_and_respawn"],
        "direction": "mostly_up",  # only decreases on death
    },
    {
        "name": "kills",
        "type": "int",
        "range": [0, 30],
        "format": "{value} kills",
        "example": "7 kills",
        "update_events": ["solo_kill", "teamfight"],
        "direction": "monotonic_up",
    },
    {
        "name": "deaths",
        "type": "int",
        "range": [0, 20],
        "format": "{value} deaths",
        "example": "4 deaths",
        "update_events": ["solo_kill", "teamfight", "death_and_respawn"],
        "direction": "monotonic_up",
    },
    {
        "name": "assists",
        "type": "int",
        "range": [0, 35],
        "format": "{value} assists",
        "example": "12 assists",
        "update_events": ["solo_kill", "teamfight"],
        "direction": "monotonic_up",
    },
    {
        "name": "kda_string",
        "type": "str",
        "range": "K/D/A format",
        "format": "a score line of {value}",
        "example": "a score line of 7/4/12",
        "update_events": ["solo_kill", "teamfight", "death_and_respawn"],
        "direction": "composite",
    },
    {
        "name": "last_hits",
        "type": "int",
        "range": [0, 500],
        "format": "{value} last hits",
        "example": "186 last hits",
        "update_events": ["farming_update"],
        "direction": "monotonic_up",
    },
    {
        "name": "denies",
        "type": "int",
        "range": [0, 40],
        "format": "{value} denies",
        "example": "14 denies",
        "update_events": ["farming_update"],
        "direction": "monotonic_up",
    },
    {
        "name": "level",
        "type": "int",
        "range": [1, 30],
        "format": "level {value}",
        "example": "level 14",
        "update_events": ["level_up", "farming_update", "solo_kill", "teamfight"],
        "direction": "monotonic_up",
    },
    {
        "name": "gpm",
        "type": "int",
        "range": [100, 900],
        "format": "{value} GPM",
        "example": "623 GPM",
        "update_events": ["farming_update", "solo_kill", "teamfight", "tower_kill"],
        "direction": "volatile",
    },
    {
        "name": "xpm",
        "type": "int",
        "range": [100, 900],
        "format": "{value} XPM",
        "example": "548 XPM",
        "update_events": ["farming_update", "solo_kill", "teamfight"],
        "direction": "volatile",
    },
    {
        "name": "latest_item",
        "type": "str",
        "range": "item name from ITEMS dict",
        "format": "{value}",
        "example": "Battle Fury",
        "update_events": ["item_purchase"],
        "direction": "categorical",
    },
    {
        "name": "total_damage_dealt",
        "type": "int",
        "range": [0, 80000],
        "format": "{value:,} total hero damage",
        "example": "23,400 total hero damage",
        "update_events": ["solo_kill", "teamfight"],
        "direction": "monotonic_up",
    },
    {
        "name": "total_damage_taken",
        "type": "int",
        "range": [0, 60000],
        "format": "{value:,} damage taken",
        "example": "18,700 damage taken",
        "update_events": ["solo_kill", "teamfight", "death_and_respawn"],
        "direction": "monotonic_up",
    },
    {
        "name": "tower_damage",
        "type": "int",
        "range": [0, 15000],
        "format": "{value:,} tower damage",
        "example": "4,200 tower damage",
        "update_events": ["tower_kill", "barracks_kill"],
        "direction": "monotonic_up",
    },
    {
        "name": "healing_done",
        "type": "int",
        "range": [0, 25000],
        "format": "{value:,} healing done",
        "example": "8,300 healing done",
        "update_events": ["teamfight", "farming_update"],
        "direction": "monotonic_up",
    },
    {
        "name": "wards_placed",
        "type": "int",
        "range": [0, 30],
        "format": "{value} wards placed",
        "example": "11 wards placed",
        "update_events": ["farming_update"],
        "direction": "monotonic_up",
    },
    {
        "name": "camps_stacked",
        "type": "int",
        "range": [0, 20],
        "format": "{value} camps stacked",
        "example": "6 camps stacked",
        "update_events": ["farming_update"],
        "direction": "monotonic_up",
    },
    {
        "name": "runes_collected",
        "type": "int",
        "range": [0, 15],
        "format": "{value} runes collected",
        "example": "5 runes collected",
        "update_events": ["farming_update"],
        "direction": "monotonic_up",
    },
    {
        "name": "kill_streak",
        "type": "int",
        "range": [0, 15],
        "format": "a {value}-kill streak",
        "example": "a 5-kill streak",
        "update_events": ["solo_kill", "teamfight", "death_and_respawn"],
        "direction": "volatile",  # resets on death
    },
    {
        "name": "buyback_status",
        "type": "str",
        "range": ["available", "on cooldown"],
        "format": "buyback {value}",
        "example": "buyback available",
        "update_events": ["death_and_respawn"],
        "direction": "categorical",
    },
    {
        "name": "respawn_timer",
        "type": "int",
        "range": [0, 120],
        "format": "{value}s respawn timer",
        "example": "62s respawn timer",
        "update_events": ["death_and_respawn"],
        "direction": "volatile",
    },
    {
        "name": "gold_advantage",
        "type": "int",
        "range": [-20000, 20000],
        "format": "{value:,} gold advantage",
        "example": "3,400 gold advantage",
        "update_events": ["farming_update", "solo_kill", "teamfight", "tower_kill"],
        "direction": "volatile",
    },
    {
        "name": "inventory_count",
        "type": "int",
        "range": [0, 6],
        "format": "{value} items in inventory",
        "example": "5 items in inventory",
        "update_events": ["item_purchase"],
        "direction": "mostly_up",
    },
    {
        "name": "net_worth_rank",
        "type": "int",
        "range": [1, 10],
        "format": "ranked #{value} in net worth",
        "example": "ranked #2 in net worth",
        "update_events": ["farming_update", "solo_kill", "teamfight"],
        "direction": "volatile",
    },
]

# Attribute assignment by role when attribute_mode == "mixed"
ROLE_ATTRIBUTE_AFFINITY = {
    "pos1": ["gold", "net_worth", "last_hits", "gpm", "latest_item"],
    "pos2": ["kills", "level", "xpm", "net_worth", "total_damage_dealt"],
    "pos3": ["deaths", "total_damage_taken", "net_worth", "tower_damage", "level"],
    "pos4": ["assists", "wards_placed", "camps_stacked", "kills", "gpm"],
    "pos5": ["assists", "wards_placed", "healing_done", "deaths", "runes_collected"],
}
# When attribute_mode == "mixed", for each hero pick rng.choice(ROLE_ATTRIBUTE_AFFINITY[hero_role])
```

---

## 4. TEAM AND TOURNAMENT NAME POOLS

```python
TEAM_NAMES = [
    "Storm Vanguard", "Eclipse Gaming", "Phantom Regiment", "Iron Crown",
    "Nebula Esports", "Shadow Crest", "Arctic Wolves", "Solar Flux",
    "Crimson Tide", "Void Reapers", "Thunder Legion", "Crystal Dominion",
    "Obsidian Order", "Frost Sentinel", "Blaze Horizon", "Neon Dynasty",
    "Titan Force", "Dark Meridian", "Apex Predators", "Nova Strike",
    "Steel Phoenix", "Lunar Eclipse", "Storm Breakers", "Ember Rising",
    "Quantum Drift", "Raven Claw", "Zenith Esports", "Hydra Gaming",
    "Vortex Squad", "Omega Alliance", "Sapphire Wolves", "Granite Order",
    "Silver Serpents", "Plasma Core", "Tundra United", "Inferno Blaze",
    "Shadow Veil", "Prism Esports", "Atlas Titans", "Horizon Walkers",
    "Jade Falcons", "Onyx Guard", "Rift Runners", "Tempest Rising",
    "Cobalt Five", "Ironclad Gaming", "Aether Syndicate", "Cerulean Storm",
]
# 48 entries. Sample 2 distinct per trial.

TOURNAMENT_NAMES = [
    "The Meridian Cup", "Vanguard Championship", "Apex Invitational",
    "The Crown Series", "Horizon Masters", "Eclipse League Season 4",
    "The Nexus Tournament", "Crimson Open", "Steel Summit",
    "Dynasty Championship", "The Forge Invitational", "Celestial Cup",
    "Northern Frontier", "The Obsidian Major", "Titan Series Grand Finals",
    "Omega League Playoffs", "The Pinnacle Clash", "Inferno Open Season 2",
    "Continental Masters", "The Ascendancy Cup", "Worldwide Arena Championship",
    "Regionals Week 7", "The Diamond Bracket", "Overture Invitational",
]
# 24 entries. Sample 1 per trial.
```

---

## 5. LOCATION POOL

```python
LOCATIONS = [
    # Lanes
    "the top lane",
    "the mid lane",
    "the bottom lane",
    "the safe lane",
    "the offlane",
    # Jungle areas
    "the Radiant jungle",
    "the Dire jungle",
    "the Radiant ancients camp",
    "the Dire ancients camp",
    "the Radiant triangle",
    "the Dire triangle",
    "the large camp near the offlane",
    "the small camp near the safe lane",
    # Key landmarks
    "the Roshan pit",
    "the river",
    "the Dire outpost",
    "the Radiant outpost",
    "the top power rune spot",
    "the bottom power rune spot",
    # Tower areas
    "the Radiant tier 1 tower",
    "the Dire tier 1 tower",
    "the Radiant high ground",
    "the Dire high ground",
    # Specific areas
    "the ward cliff",
    "the Radiant secret shop",
    "the Dire secret shop",
    "the Radiant base",
    "the Dire base",
]
# 28 entries. Sample randomly per event.
```

---

## 6. COMPLETE NARRATIVE TEMPLATES

All templates use `{placeholder}` syntax. Placeholders are filled from event data dicts.

### 6.1 Match Header Templates (5)

```python
HEADER_TEMPLATES = [
    "Game {game_id} of the {tournament}: {team1} ({team1_heroes}) versus {team2} ({team2_heroes}). {team1} secured first pick with {first_pick}, while {team2} responded with {second_pick} to anchor their draft.",

    "A {adjective} matchup in the {tournament} as {team1} drafts {team1_heroes} against {team2}'s {team2_heroes}. The analysts gave {favored_team} a slight edge heading into this one.",

    "Match {game_id} in the {tournament} bracket. {team1} ({team1_heroes}) locks horns with {team2} ({team2_heroes}). Both sides opted for {draft_style} drafts, setting up a {expected_pace} game.",

    "The {tournament} continues with {team1} fielding {team1_heroes} against {team2}'s lineup of {team2_heroes}. {team1}'s {pos1_hero} will be the focal point of their strategy.",

    "{tournament}, Game {game_id}. On the Radiant side, {team1} with {team1_heroes}. Dire responds: {team2} running {team2_heroes}. The stage is set.",
]
```

### 6.2 Farming Update Templates (8)

```python
FARMING_TEMPLATES = [
    "By the {time}-minute mark, {hero}'s efficient farming in {location} had pushed {pron_pos} {attr_name} to {attr_value}.",

    "{hero} continued to find farm in {location}, {pron_pos} {attr_name} sitting at {attr_value} at {time} minutes.",

    "With {cs} creeps secured by {time} minutes, {hero}'s {attr_name} stood at {attr_value} — {comparison} for a {role} at this stage.",

    "{location} was proving productive for {hero}, who had accumulated {attr_value} {attr_label} by the {time}-minute mark.",

    "Uncontested farm in {location} let {hero} climb to {attr_value} {attr_label} at {time} minutes, a strong pace for {pron_pos} {role} role.",

    "At {time} minutes, {hero} was quietly farming {location}. {pron_sub} {attr_name} had reached {attr_value}, keeping pace with the game plan.",

    "The scoreboard at {time} minutes showed {hero} at {attr_value} {attr_label} after a productive stint in {location}.",

    "Free farm in {location} translated to {attr_value} {attr_label} for {hero} at the {time}-minute mark, well within expectations for a {role}.",
]
```

### 6.3 Solo Kill Templates (8)

```python
SOLO_KILL_TEMPLATES = [
    "At {time}:{seconds}, {killer} found {victim} alone near {location} and {kill_verb} {pron_obj}. The {bounty}-gold bounty pushed {killer}'s {attr_name} to {killer_attr_value}. {victim}'s {attr_name} dropped to {victim_attr_value} with a {respawn}s respawn timer.",

    "A rotation from {killer} caught {victim} off-guard at {location}. The kill brought {killer}'s {attr_name} to {killer_attr_value}. {victim} fell to {victim_attr_value} {attr_label}, facing {respawn} seconds dead.",

    "{killer} {kill_verb} {victim} near {location} at {time} minutes, earning {bounty} gold for the takedown. {killer} now sat at {killer_attr_value} {attr_label} while {victim} dropped to {victim_attr_value}.",

    "An aggressive play from {killer} at {location} resulted in {victim}'s death at {time}:{seconds}. {killer}'s {attr_name} climbed to {killer_attr_value}; {victim}'s fell to {victim_attr_value}.",

    "{victim} was caught out of position near {location} and {killer} made {pron_obj} pay. Post-kill, {killer} moved to {killer_attr_value} {attr_label}, while {victim} sat at {victim_attr_value} awaiting respawn.",

    "At the {time}-minute mark, {killer} {kill_verb} {victim} in {location}. The {bounty}-gold swing left {killer} at {killer_attr_value} {attr_label} and {victim} at {victim_attr_value}.",

    "{killer} executed a clean kill on {victim} near {location} at {time}:{seconds}. {killer}'s {attr_name}: {killer_attr_value}. {victim}'s {attr_name}: {victim_attr_value}. Respawn in {respawn} seconds.",

    "A pickoff near {location} — {killer} {kill_verb} {victim} at {time} minutes. With the bounty gold, {killer} reached {killer_attr_value} {attr_label}. {victim} dropped to {victim_attr_value}.",
]
```

### 6.4 Teamfight Templates (8)

```python
TEAMFIGHT_TEMPLATES = [
    "A massive engagement broke out at {location} at the {time}-minute mark. {winning_team} came out ahead, trading {win_kills} for {loss_kills}. {mvp} was the standout with {mvp_kills} kills, climbing to {mvp_attr_value} {attr_label}. On the losing side, {worst}'s {attr_name} dropped to {worst_attr_value}.",

    "Chaos erupted near {location} at {time} minutes. {fight_description}. When the dust settled, {mvp} emerged at {mvp_attr_value} {attr_label}, while {worst} fell to {worst_attr_value}.",

    "A {size}-hero teamfight at {location} around the {time}-minute mark went {winning_team}'s way, {win_kills} to {loss_kills}. {mvp} hit {mvp_attr_value} {attr_label} after the engagement. {worst} was left at {worst_attr_value}.",

    "{winning_team} forced a fight at {location} at {time} minutes and came out with {win_kills} kills to {loss_kills}. {mvp} reached {mvp_attr_value} {attr_label}. {worst} dropped to {worst_attr_value} after dying.",

    "The teams clashed at {location} at {time} minutes. {winning_team} won the fight decisively — {win_kills} kills for {loss_kills}. Post-fight, {mvp} sat at {mvp_attr_value} {attr_label}. {worst} was down to {worst_attr_value}.",

    "A pivotal teamfight at {location} ({time} min) saw {winning_team} take {win_kills} kills while conceding {loss_kills}. {mvp}'s {attr_name} surged to {mvp_attr_value}. Meanwhile, {worst} fell to {worst_attr_value}.",

    "Fighting broke out near {location} at {time} minutes. {fight_description}. {mvp} came out on top at {mvp_attr_value} {attr_label}; {worst} was the biggest loser at {worst_attr_value}.",

    "At {time} minutes, a full-scale engagement at {location}. {winning_team} took the fight {win_kills}-{loss_kills}. {mvp} climbed to {mvp_attr_value} {attr_label}. On the other side, {worst}'s {attr_name} sank to {worst_attr_value}.",
]
```

### 6.5 Item Purchase Templates (8)

```python
ITEM_PURCHASE_TEMPLATES = [
    "{hero} completed {item} for {cost} gold at {time} minutes, leaving {pron_obj} with {remaining_gold} gold in reserve. {pron_pos} {attr_name} now stood at {attr_value}.",

    "A trip to the shop saw {hero} pick up {item} at the {time}-minute mark. After spending {cost} gold, {pron_pos} {attr_name} sat at {attr_value}.",

    "With {item} now in {pron_pos} inventory, {hero}'s {attr_name} reached {attr_value} at {time} minutes. The {cost}-gold purchase was a key timing.",

    "{hero} secured {item} at {time} minutes, a critical pickup for the {role}. {pron_pos} {attr_name} moved to {attr_value} after the {cost}-gold investment.",

    "At {time} minutes, {hero} finished {item} ({cost} gold). {pron_pos} {attr_name}: {attr_value}. This completed {pron_pos} {item_tier} item progression.",

    "The {item} came online for {hero} at {time} minutes — {cost} gold well spent. {pron_pos} {attr_name} was now {attr_value}.",

    "{hero} picked up {item} at the {time}-minute mark for {cost} gold, an important piece of the puzzle. {attr_name}: {attr_value}.",

    "Backing to complete {item} ({cost} gold) at {time} minutes, {hero}'s {attr_name} adjusted to {attr_value}. {item_significance}.",
]
```

### 6.6 Tower Kill Templates (8)

```python
TOWER_KILL_TEMPLATES = [
    "{team} pushed down the {lane} tier {tier} tower at {time} minutes. The {tower_gold}-gold team bounty boosted everyone — notably {hero}, whose {attr_name} rose to {attr_value}.",

    "The {lane} lane tier {tier} tower fell to {team} at the {time}-minute mark, distributing {tower_gold} gold to each player. {hero}'s {attr_name} climbed to {attr_value}.",

    "At {time} minutes, {team} took the {lane} T{tier}. The team gold pushed {hero}'s {attr_name} to {attr_value}. Map control shifted accordingly.",

    "{team} secured the {lane} tier {tier} tower at {time} minutes after a sustained push. {hero} hit {attr_value} {attr_label} from the {tower_gold}-gold injection.",

    "The {lane} T{tier} crumbled under {team}'s pressure at {time} minutes. {tower_gold} gold per hero. {hero}'s {attr_name}: {attr_value}.",

    "Objective-focused play from {team} netted the {lane} tier {tier} tower at the {time}-minute mark. {hero} benefited most, reaching {attr_value} {attr_label}.",

    "At {time} minutes, the {lane} T{tier} fell. {team} pocketed {tower_gold} gold each. {hero}'s {attr_name} ticked up to {attr_value}.",

    "{team}'s sustained siege paid off — {lane} tier {tier} tower down at {time} minutes. {hero} reached {attr_value} {attr_label} with the team bounty.",
]
```

### 6.7 Roshan Kill Templates (8)

```python
ROSHAN_TEMPLATES = [
    "{team} secured Roshan at {time} minutes, with {carrier} claiming the Aegis. The bonus gold pushed {hero}'s {attr_name} to {attr_value}.",

    "Roshan fell to {team} at the {time}-minute mark — their {nth} Roshan kill. {drops}. {hero}'s {attr_name} climbed to {attr_value}.",

    "A clean Roshan take by {team} at {time} minutes. {carrier} grabbed the Aegis. The shared bounty brought {hero}'s {attr_name} to {attr_value}.",

    "At {time} minutes, {team} snuck Roshan. {drops}. Post-Rosh, {hero} sat at {attr_value} {attr_label}.",

    "{team} committed to the Roshan pit at {time} minutes and secured the kill. {carrier} took the Aegis. {hero}'s {attr_name} reached {attr_value}.",

    "Roshan number {rosh_count} went to {team} at {time} minutes. {drops}. {hero} moved to {attr_value} {attr_label} from the bounty.",

    "The Roshan pit was the site of {team}'s {nth} Rosh kill at {time} minutes. {hero}'s {attr_name} jumped to {attr_value}. {carrier} held the Aegis.",

    "A quick Roshan by {team} at the {time}-minute mark. The gold distribution pushed {hero} to {attr_value} {attr_label}. {drops}.",
]
```

### 6.8 Level Up Templates (8)

```python
LEVEL_UP_TEMPLATES = [
    "{hero} hit {attr_value} at {time} minutes, unlocking {ability_note}.",

    "A wave of experience pushed {hero} to {attr_value} at the {time}-minute mark.",

    "{hero} reached {attr_value} at {time} minutes — {timing_assessment} for a {role}.",

    "At {time} minutes, {hero} dinged {attr_value}. {ability_note}.",

    "The experience from {xp_source} pushed {hero} to {attr_value} at {time} minutes.",

    "{hero}'s level ticked up to {attr_value} at the {time}-minute mark, keeping pace with the {role} benchmark.",

    "With {attr_value} now reached at {time} minutes, {hero} had {ability_note}.",

    "{hero} leveled to {attr_value} at {time} minutes after clearing {xp_source}.",
]
```

### 6.9 Death and Respawn Templates (8)

```python
DEATH_RESPAWN_TEMPLATES = [
    "{hero} was {kill_verb} at {location} at {time} minutes. {pron_pos} {attr_name} dropped to {attr_value}. Respawn: {respawn}s.",

    "A costly death for {hero} at {location} at {time} minutes — {pron_pos} {attr_name} fell to {attr_value}. {respawn} seconds off the map.",

    "{hero} fell at {location} ({time} min), losing gold and dropping to {attr_value} {attr_label}. The {respawn}-second death timer was punishing.",

    "At {time} minutes, {hero} died near {location}. {pron_pos} {attr_name} slid to {attr_value}. Back in {respawn} seconds.",

    "{hero}'s death at {location} at the {time}-minute mark cost {gold_lost} gold. {attr_name}: {attr_value}. Respawn in {respawn}s.",

    "Caught out at {location}, {hero} went down at {time} minutes. {pron_pos} {attr_name} dropped to {attr_value}. {respawn}s until return.",

    "A {death_context} death for {hero} at {location} at {time} minutes. {attr_name} now: {attr_value}. Off the map for {respawn} seconds.",

    "{hero} was eliminated near {location} at {time} minutes, {pron_pos} {attr_name} falling to {attr_value}. The team played 4v5 for {respawn} seconds.",
]
```

### 6.10 Barracks Kill Templates (8)

```python
BARRACKS_KILL_TEMPLATES = [
    "{team} destroyed the {lane} {rax_type} barracks at {time} minutes. Super creeps would now march down the {lane} lane. {hero}'s {attr_name} hit {attr_value}.",

    "The {lane} {rax_type} barracks fell to {team} at {time} minutes. {hero} reached {attr_value} {attr_label} from the {rax_gold}-gold bounty.",

    "At {time} minutes, {team} took the {lane} {rax_type} rax. {hero}'s {attr_name} moved to {attr_value}. The map pressure intensified.",

    "{team} broke through the {lane} high ground at {time} minutes, destroying the {rax_type} barracks. {hero} sat at {attr_value} {attr_label}.",

    "The {lane} {rax_type} barracks crumbled at {time} minutes. {team} now had super creeps in {lane}. {hero}'s {attr_name}: {attr_value}.",

    "High ground breached — {team} razed the {lane} {rax_type} rax at {time} minutes. {hero} climbed to {attr_value} {attr_label}.",

    "At the {time}-minute mark, {team} demolished the {lane} {rax_type} barracks. {rax_gold} gold distributed. {hero}'s {attr_name} reached {attr_value}.",

    "{team}'s push paid off: {lane} {rax_type} barracks down at {time} minutes. {hero} hit {attr_value} {attr_label}. The mega creep threat loomed.",
]
```

### 6.11 Filler / Context Templates (10)

These mention non-tracked attributes ONLY. They NEVER mention tracked attribute values for ANY entity.

```python
FILLER_TEMPLATES = [
    "{hero} had maxed out {ability} by this point, giving {pron_obj} strong {ability_type} presence.",

    "Ward vision from {support} revealed enemy movements near {location}, giving {team} the information advantage.",

    "The draft advantage was beginning to show — {team}'s {hero} was creating space across the map.",

    "Both teams had committed their teleport scrolls, leaving the opposite side of the map exposed.",

    "The gold graph showed {team} with a modest lead at this stage, though the lineup favored {other_team} in the late game.",

    "{hero} rotated to {location} looking for a gank opportunity, but found nothing and returned to farming.",

    "Smoke of Deceit was used by {team}, but the gank attempt was read and no kills came of it.",

    "A power rune spawned at the {rune_spot} — {hero} grabbed the {rune_type} rune.",

    "The {lane} lane equilibrium had shifted heavily toward {team}'s side after the recent engagement.",

    "{hero} stacked the ancients for {teammate}, setting up a future farming opportunity.",
]
```

### 6.12 Segment Transition Templates (5)

```python
SEGMENT_TRANSITION_TEMPLATES = [
    "As the game moved past the {time}-minute mark, the pace shifted toward {phase_description}.",

    "The {old_phase} phase was over. At {time} minutes, {phase_description} began in earnest.",

    "With the clock reading {time} minutes, the game transitioned into {phase_description}.",

    "{time} minutes in and the game entered a new stage — {phase_description}.",

    "The {old_phase} gave way to {phase_description} as the timer hit {time} minutes.",
]
```

### 6.13 Kill Description Variants (8)

```python
KILL_VERBS = [
    "ganked",
    "burst down",
    "picked off",
    "solo-killed",
    "dove and killed",
    "ambushed",
    "deleted",
    "caught and eliminated",
]
```

### 6.14 Pronoun Helpers

```python
# All heroes use "their/them/they" (gender-neutral) in this generator.
# This avoids needing a gender database.
PRONOUNS = {
    "pron_sub": "they",
    "pron_obj": "them",
    "pron_pos": "their",
}
```

---

## 7. TIME PROGRESSION MODEL

### 7.1 Game phases

```python
GAME_PHASES = {
    "laning": {"start": 0, "end": 12},      # minutes
    "mid_game": {"start": 12, "end": 26},
    "late_game": {"start": 26, "end": 60},
}

def get_phase(game_time_minutes: float) -> str:
    if game_time_minutes < 12:
        return "laning"
    elif game_time_minutes < 26:
        return "mid_game"
    else:
        return "late_game"
```

### 7.2 Time advancement per event type

Each event advances `game_time` by a sampled amount (minutes). Ranges below are inclusive.

```python
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
```

### 7.3 Rules

1. Sample `dt = rng.uniform(lo, hi)` for the event's phase.
2. `game_time += dt`. Round `game_time` to 1 decimal place.
3. **Minimum gap**: 0.3 minutes between any two events.
4. **Maximum gap**: 3.0 minutes between any two events. If sampled `dt > 3.0`, clamp to 3.0.
5. Game time never exceeds 60.0 minutes.
6. For `match_scope == "window"`, `game_time` starts at `config.start_time` (see Section 10), not 0.

---

## 8. FILLER INSERTION MODEL

### 8.1 Filler probability per budget level

After each event that mentions a tracked attribute value, roll to insert filler:

```python
FILLER_CONFIG = {
    "minimal": {
        "insert_probability": 0.05,   # 5% chance of filler after each tracked mention
        "sentences_per_insertion": 1,  # always 1 sentence when it does occur
    },
    "light": {
        "insert_probability": 0.25,
        "sentences_per_insertion": 1,  # 1 sentence
    },
    "medium": {
        "insert_probability": 0.45,
        "sentences_per_insertion": (1, 2),  # rng.randint(1, 2) sentences
    },
    "heavy": {
        "insert_probability": 0.65,
        "sentences_per_insertion": (1, 3),  # rng.randint(1, 3) sentences
    },
}
```

### 8.2 Rules

1. After rendering an event that mentions a tracked attribute, roll `rng.random() < insert_probability`.
2. If True, sample `n` filler sentences (1 if int, `rng.randint(lo, hi)` if tuple).
3. Pick `n` distinct filler templates from `FILLER_TEMPLATES` (no repeats within one insertion).
4. Fill template placeholders with hero names and non-tracked attribute values from the current state.
5. **Filler MUST NOT mention any tracked attribute value for any entity.** It can mention hero names. It can mention non-tracked attributes (e.g., if tracking gold, filler can say "level 14" but never a gold number).
6. Filler can mention any hero in the match, not just the ones being tracked. This is safe because filler never contains tracked values.

---

## 9. ARCHETYPE EFFECTS

### 9.1 Default event weights (baseline)

```python
DEFAULT_EVENT_WEIGHTS = {
    "laning": {
        "farming_update": 40,
        "solo_kill": 15,
        "item_purchase": 25,
        "level_up": 15,
        "tower_kill": 5,
        "teamfight": 0,
        "roshan_kill": 0,
        "death_and_respawn": 0,
        "barracks_kill": 0,
    },
    "mid_game": {
        "farming_update": 15,
        "solo_kill": 20,
        "teamfight": 15,
        "item_purchase": 20,
        "tower_kill": 10,
        "roshan_kill": 5,
        "level_up": 10,
        "death_and_respawn": 5,
        "barracks_kill": 0,
    },
    "late_game": {
        "farming_update": 5,
        "solo_kill": 10,
        "teamfight": 25,
        "item_purchase": 15,
        "tower_kill": 15,
        "roshan_kill": 10,
        "barracks_kill": 10,
        "level_up": 5,
        "death_and_respawn": 5,
    },
}
```

### 9.2 Per-archetype weight modifiers

Each archetype applies a multiplicative modifier to the base weights. After multiplication, renormalize to sum to 100.

```python
ARCHETYPE_MODIFIERS = {
    "stomp": {
        # One team dominates. More kills, faster towers, less farming.
        "description": "One-sided game. Winning team snowballs.",
        "gold_distribution": "winning_team_ahead_by_factor_2.0",
        # winning team heroes get 2x gold rate; losing team gets 0.6x
        "weight_mods": {
            "laning":    {"solo_kill": 1.5, "farming_update": 0.8, "tower_kill": 1.5},
            "mid_game":  {"solo_kill": 1.5, "teamfight": 1.3, "tower_kill": 1.5, "farming_update": 0.5},
            "late_game": {"teamfight": 1.5, "tower_kill": 1.5, "barracks_kill": 1.5, "farming_update": 0.3},
        },
        "special_events": [],
    },
    "comeback": {
        # One team behind early, comes back mid/late.
        "description": "Team A leads early. Team B comes back.",
        "gold_distribution": "early_leader_then_reversal",
        # Phase 1 (laning): team_a gets 1.5x gold. Phase 2+ (mid/late): team_b gets 1.3x gold.
        "weight_mods": {
            "laning":    {"solo_kill": 1.3, "farming_update": 1.0},
            "mid_game":  {"teamfight": 1.5, "roshan_kill": 1.5, "solo_kill": 0.8},
            "late_game": {"teamfight": 1.5, "roshan_kill": 1.3, "tower_kill": 1.3},
        },
        "special_events": [],
    },
    "close": {
        # Back and forth. Both teams trading.
        "description": "Close game, alternating advantages.",
        "gold_distribution": "even",
        # both teams within 10% of each other in gold
        "weight_mods": {
            "laning":    {"farming_update": 1.2, "solo_kill": 1.0},
            "mid_game":  {"teamfight": 1.3, "solo_kill": 1.0, "roshan_kill": 1.3},
            "late_game": {"teamfight": 1.5, "roshan_kill": 1.5},
        },
        "special_events": [],
    },
    "farmfest": {
        # Few kills, lots of farming, late-game focused.
        "description": "Passive game. High CS, low kills.",
        "gold_distribution": "even",
        "weight_mods": {
            "laning":    {"farming_update": 1.8, "solo_kill": 0.3, "item_purchase": 1.3},
            "mid_game":  {"farming_update": 1.5, "solo_kill": 0.4, "teamfight": 0.4, "item_purchase": 1.5},
            "late_game": {"farming_update": 1.0, "teamfight": 1.2, "item_purchase": 1.3},
        },
        "special_events": [],
    },
    "bloodbath": {
        # Many kills, aggressive playstyle.
        "description": "Aggressive game. Constant fighting.",
        "gold_distribution": "volatile",
        # gold swings wildly between teams
        "weight_mods": {
            "laning":    {"solo_kill": 2.0, "farming_update": 0.5, "death_and_respawn": 1.5},
            "mid_game":  {"solo_kill": 1.8, "teamfight": 1.8, "farming_update": 0.3, "death_and_respawn": 1.3},
            "late_game": {"teamfight": 2.0, "solo_kill": 1.5, "death_and_respawn": 1.5, "farming_update": 0.2},
        },
        "special_events": [],
    },
    "split_push": {
        # One team avoids fights, pushes lanes.
        "description": "Rat Dota. Split push avoids direct fights.",
        "gold_distribution": "even",
        "weight_mods": {
            "laning":    {"farming_update": 1.3, "tower_kill": 1.5},
            "mid_game":  {"tower_kill": 2.0, "farming_update": 1.3, "teamfight": 0.5, "solo_kill": 0.7},
            "late_game": {"tower_kill": 2.0, "barracks_kill": 2.0, "teamfight": 0.5, "farming_update": 1.0},
        },
        "special_events": [],
    },
    "roshan_centric": {
        # Game revolves around Roshan fights.
        "description": "Multiple Roshan fights define the game.",
        "gold_distribution": "even",
        "weight_mods": {
            "laning":    {"farming_update": 1.0},
            "mid_game":  {"roshan_kill": 3.0, "teamfight": 1.5},
            "late_game": {"roshan_kill": 3.0, "teamfight": 1.5, "barracks_kill": 1.3},
        },
        "special_events": [],
    },
    "base_race": {
        # Both teams pushing simultaneously in late game.
        "description": "Late-game base race. Both teams all-in on pushing.",
        "gold_distribution": "even",
        "weight_mods": {
            "laning":    {"farming_update": 1.2},
            "mid_game":  {"tower_kill": 1.5, "teamfight": 1.0},
            "late_game": {"tower_kill": 2.5, "barracks_kill": 2.5, "teamfight": 0.5, "farming_update": 0.2},
        },
        "special_events": [],
    },
}
```

### 9.3 Applying modifiers

```python
def get_event_weights(phase: str, archetype: str) -> dict:
    base = dict(DEFAULT_EVENT_WEIGHTS[phase])
    mods = ARCHETYPE_MODIFIERS[archetype]["weight_mods"].get(phase, {})
    for event_type, multiplier in mods.items():
        if event_type in base:
            base[event_type] = base[event_type] * multiplier
    # Renormalize
    total = sum(base.values())
    if total > 0:
        base = {k: (v / total) * 100 for k, v in base.items()}
    return base
```

### 9.4 Gold distribution by archetype

```python
def get_gold_multipliers(archetype: str, phase: str, team: str) -> float:
    """Returns a multiplier applied to gold gains for a team in a given phase."""
    if archetype == "stomp":
        # Pick winning team at trial init: rng.choice(["radiant", "dire"])
        # stored as config.stomp_winner
        if team == config.stomp_winner:
            return 1.5 if phase == "laning" else 2.0
        else:
            return 0.7 if phase == "laning" else 0.5
    elif archetype == "comeback":
        # config.early_leader set at init
        if phase == "laning":
            return 1.4 if team == config.early_leader else 0.7
        else:
            return 0.7 if team == config.early_leader else 1.3
    elif archetype in ("close", "farmfest", "split_push", "roshan_centric", "base_race"):
        return 1.0  # even
    elif archetype == "bloodbath":
        return 1.0  # volatile but centered on 1.0
    return 1.0
```

---

## 10. WINDOW MODE INITIALIZATION

When `match_scope == "window"`, the generator starts at a mid-game timestamp. Hero states must be initialized to realistic values.

### 10.1 Start time determination

```python
def determine_start_time(num_updates: int, rng) -> float:
    """Returns start_time in minutes."""
    if num_updates <= 5:
        return rng.uniform(5.0, 20.0)    # short window, can start anywhere
    elif num_updates <= 10:
        return rng.uniform(0.0, 15.0)    # medium coverage
    else:
        return 0.0                        # full match, always start at 0
```

### 10.2 Initial state formulas by role and start_time

Gold trajectories (from mechanics doc). For a given `start_time` and role, sample gold uniformly from the range:

```python
GOLD_TRAJECTORY = {
    "pos1": {
        5:  (1800, 2500),
        10: (3500, 5000),
        15: (5500, 8000),
        20: (8000, 12000),
        25: (11000, 16000),
        30: (14000, 22000),
        35: (17000, 28000),
        40: (20000, 35000),
    },
    "pos2": {
        5:  (1600, 2300),
        10: (3200, 4500),
        15: (5000, 7200),
        20: (7000, 11000),
        25: (9500, 14000),
        30: (12000, 19000),
        35: (14500, 24000),
        40: (17000, 30000),
    },
    "pos3": {
        5:  (1200, 1800),
        10: (2500, 3800),
        15: (4000, 6000),
        20: (5500, 8500),
        25: (7500, 11000),
        30: (9500, 14000),
        35: (11500, 17000),
        40: (13500, 20000),
    },
    "pos4": {
        5:  (900, 1400),
        10: (1800, 2800),
        15: (2800, 4200),
        20: (3800, 5800),
        25: (5000, 7500),
        30: (6200, 9500),
        35: (7500, 11500),
        40: (8500, 13500),
    },
    "pos5": {
        5:  (800, 1200),
        10: (1500, 2500),
        15: (2500, 4000),
        20: (3500, 5500),
        25: (4500, 7000),
        30: (6000, 9000),
        35: (7000, 11000),
        40: (8000, 13000),
    },
}

def interpolate_gold_range(role: str, start_time: float) -> tuple:
    """Linearly interpolate between the two nearest time anchors."""
    trajectory = GOLD_TRAJECTORY[role]
    times = sorted(trajectory.keys())
    if start_time <= times[0]:
        return trajectory[times[0]]
    if start_time >= times[-1]:
        return trajectory[times[-1]]
    # Find surrounding anchors
    for i in range(len(times) - 1):
        if times[i] <= start_time <= times[i + 1]:
            t0, t1 = times[i], times[i + 1]
            lo0, hi0 = trajectory[t0]
            lo1, hi1 = trajectory[t1]
            frac = (start_time - t0) / (t1 - t0)
            lo = lo0 + (lo1 - lo0) * frac
            hi = hi0 + (hi1 - hi0) * frac
            return (int(lo), int(hi))
    return trajectory[times[-1]]
```

### 10.3 Level initialization

```python
LEVEL_TRAJECTORY = {
    # (start_time_minutes): (role -> (min_level, max_level))
    "pos1": {5: (4, 5), 10: (8, 10),  15: (12, 14), 20: (16, 18), 25: (19, 22), 30: (22, 25), 35: (24, 27), 40: (26, 30)},
    "pos2": {5: (5, 6), 10: (9, 11),  15: (13, 15), 20: (17, 19), 25: (20, 23), 30: (23, 26), 35: (25, 28), 40: (27, 30)},
    "pos3": {5: (3, 5), 10: (7, 9),   15: (10, 13), 20: (14, 17), 25: (17, 20), 30: (20, 23), 35: (22, 26), 40: (24, 28)},
    "pos4": {5: (3, 4), 10: (5, 7),   15: (8, 11),  20: (11, 14), 25: (14, 17), 30: (16, 20), 35: (18, 22), 40: (20, 25)},
    "pos5": {5: (2, 4), 10: (4, 6),   15: (7, 10),  20: (10, 13), 25: (12, 16), 30: (15, 19), 35: (17, 21), 40: (19, 24)},
}
# Use same interpolation logic as gold.
```

### 10.4 KDA initialization

```python
def init_kda(role: str, start_time: float, rng) -> tuple:
    """Returns (kills, deaths, assists) for a hero at start_time."""
    # Average kills per minute by role
    kpm = {"pos1": 0.25, "pos2": 0.28, "pos3": 0.18, "pos4": 0.12, "pos5": 0.08}
    dpm = {"pos1": 0.12, "pos2": 0.15, "pos3": 0.18, "pos4": 0.20, "pos5": 0.25}
    apm = {"pos1": 0.25, "pos2": 0.30, "pos3": 0.35, "pos4": 0.40, "pos5": 0.45}

    kills = max(0, int(rng.gauss(kpm[role] * start_time, 1.0)))
    deaths = max(0, int(rng.gauss(dpm[role] * start_time, 1.0)))
    assists = max(0, int(rng.gauss(apm[role] * start_time, 1.5)))
    return (kills, deaths, assists)
```

### 10.5 Last hits initialization

```python
LAST_HIT_TRAJECTORY = {
    "pos1": {10: (55, 70), 20: (130, 170), 30: (220, 280), 40: (300, 380)},
    "pos2": {10: (50, 65), 20: (120, 150), 30: (200, 250), 40: (270, 340)},
    "pos3": {10: (30, 45), 20: (70, 100),  30: (120, 160), 40: (160, 220)},
    "pos4": {10: (5, 15),  20: (20, 40),   30: (30, 60),   40: (40, 80)},
    "pos5": {10: (0, 5),   20: (5, 15),    30: (10, 25),   40: (15, 35)},
}
# Use same interpolation logic as gold.
```

### 10.6 Items initialization

```python
def init_items(role: str, gold: int, rng) -> list:
    """Given a role and current gold, return a plausible item list."""
    build = ROLE_ITEM_BUILDS[role]
    items = []
    spent = 0

    # Always have boots by mid game
    items.append(build["early"][0])  # boots variant
    spent += get_item_cost(build["early"][0])

    # Add early items
    for item_name in build["early"][1:]:
        cost = get_item_cost(item_name)
        if spent + cost <= gold * 0.9:  # assume they spent ~90% of earned gold on items
            items.append(item_name)
            spent += cost

    # Add core items if gold allows
    for item_name in build["core"]:
        cost = get_item_cost(item_name)
        if spent + cost <= gold * 0.9 and len(items) < 6:
            items.append(item_name)
            spent += cost

    # Add luxury items if gold allows
    for item_name in build["luxury"]:
        cost = get_item_cost(item_name)
        if spent + cost <= gold * 0.9 and len(items) < 6:
            items.append(item_name)
            spent += cost

    return items[:6]  # max 6 slots
```

### 10.7 Full init function

```python
def initialize_hero_state(hero: str, role: str, start_time: float, archetype: str, team: str, rng) -> HeroState:
    if start_time == 0:
        return HeroState(
            name=hero, role=role, team=team,
            gold=600, net_worth=600, level=1,
            kills=0, deaths=0, assists=0,
            last_hits=0, denies=0,
            items=list(ROLE_ITEM_BUILDS[role]["starting"]),
            alive=True, kill_streak=0,
            total_damage_dealt=0, total_damage_taken=0,
            tower_damage=0, healing_done=0,
            wards_placed=0, camps_stacked=0, runes_collected=0,
        )

    gold_range = interpolate_gold_range(role, start_time)
    gold_multiplier = get_gold_multipliers(archetype, get_phase(start_time), team)
    raw_gold = rng.randint(int(gold_range[0] * gold_multiplier), int(gold_range[1] * gold_multiplier))
    kills, deaths, assists = init_kda(role, start_time, rng)
    level_range = interpolate_range(LEVEL_TRAJECTORY[role], start_time)
    level = rng.randint(level_range[0], level_range[1])
    lh_range = interpolate_range(LAST_HIT_TRAJECTORY[role], start_time)
    last_hits = rng.randint(lh_range[0], lh_range[1])
    items = init_items(role, raw_gold, rng)
    net_worth = sum(get_item_cost(i) for i in items) + rng.randint(200, 1500)

    return HeroState(
        name=hero, role=role, team=team,
        gold=rng.randint(200, 2000),  # current unspent gold
        net_worth=net_worth,
        level=level,
        kills=kills, deaths=deaths, assists=assists,
        last_hits=last_hits,
        denies=rng.randint(0, int(start_time * 0.5)),
        items=items,
        alive=True,
        kill_streak=max(0, kills - deaths) if kills > deaths else 0,
        total_damage_dealt=int(rng.gauss(start_time * 400, start_time * 80)),
        total_damage_taken=int(rng.gauss(start_time * 300, start_time * 60)),
        tower_damage=rng.randint(0, int(start_time * 100)),
        healing_done=rng.randint(0, int(start_time * 50)) if role in ("pos4", "pos5") else 0,
        wards_placed=rng.randint(0, int(start_time * 0.3)) if role in ("pos4", "pos5") else 0,
        camps_stacked=rng.randint(0, int(start_time * 0.15)) if role in ("pos4", "pos5") else 0,
        runes_collected=rng.randint(0, int(start_time * 0.1)),
    )
```

---

## 11. COMPLETE FLOW (PSEUDOCODE)

```python
def generate_dota_trial(num_keys: int, num_updates: int, condition: str, seed: int) -> dict:
    """
    Main entry point.

    Args:
        num_keys: number of heroes to track (2-10)
        num_updates: target tracked-value mentions per hero (3-50)
        condition: "RI" or "PI" (only affects question generation)
        seed: random seed for full reproducibility

    Returns:
        dict matching OUTPUT JSON SCHEMA (Section 12)
    """

    # ── STEP 1: Create RNG ──
    rng = random.Random(seed)

    # ── STEP 2: Auto-sample config ──
    config = auto_config(num_keys, num_updates, condition, seed)
    # auto_config samples: tracked_attribute, attribute_mode, filler_budget,
    # voice, match_scope, game_archetype, queried_hero_idx
    # See Section 12 of GENERATOR_DESIGN.md for full auto_config logic.

    # ── STEP 3: Pick heroes, assign teams and roles ──
    all_heroes = list(HERO_ROLES.keys())
    rng.shuffle(all_heroes)

    # Assign to teams
    radiant_count = (num_keys + 1) // 2
    dire_count = num_keys - radiant_count
    radiant_heroes = []
    dire_heroes = []

    # Fill roles for each team
    # Priority: pos1, pos2, pos3, pos4, pos5
    roles_needed_radiant = ["pos1", "pos2", "pos3", "pos4", "pos5"][:radiant_count]
    roles_needed_dire = ["pos1", "pos2", "pos3", "pos4", "pos5"][:dire_count]

    used_heroes = set()
    for role in roles_needed_radiant:
        candidates = [h for h in ROLE_POOL[role] if h not in used_heroes]
        hero = rng.choice(candidates)
        radiant_heroes.append((hero, role))
        used_heroes.add(hero)

    for role in roles_needed_dire:
        candidates = [h for h in ROLE_POOL[role] if h not in used_heroes]
        hero = rng.choice(candidates)
        dire_heroes.append((hero, role))
        used_heroes.add(hero)

    all_match_heroes = radiant_heroes + dire_heroes

    # ── STEP 4: Assign tracked attributes ──
    if config.attribute_mode == "same":
        tracked = {hero: config.tracked_attribute for hero, role in all_match_heroes}
    else:  # "mixed"
        tracked = {}
        for hero, role in all_match_heroes:
            tracked[hero] = rng.choice(ROLE_ATTRIBUTE_AFFINITY[role])

    # ── STEP 5: Determine start_time ──
    if config.match_scope == "window":
        start_time = determine_start_time(num_updates, rng)
    else:
        start_time = 0.0

    # ── STEP 6: Initialize hero states ──
    # Pick stomp_winner / early_leader if needed
    if config.game_archetype == "stomp":
        config.stomp_winner = rng.choice(["radiant", "dire"])
    if config.game_archetype == "comeback":
        config.early_leader = rng.choice(["radiant", "dire"])

    hero_states = {}
    for hero, role in all_match_heroes:
        team = "radiant" if (hero, role) in radiant_heroes else "dire"
        hero_states[hero] = initialize_hero_state(
            hero, role, start_time, config.game_archetype, team, rng
        )

    # ── STEP 7: Pick team and tournament names ──
    team1_name, team2_name = rng.sample(TEAM_NAMES, 2)
    tournament_name = rng.choice(TOURNAMENT_NAMES)

    # ── STEP 8: Generate event sequence ──
    game_time = start_time
    event_log = []          # list of Event objects
    full_state_log = []     # list of state change records
    entity_tracking = {f"{hero} / {tracked[hero]}": [] for hero, _ in all_match_heroes}
    mention_counts = {hero: 0 for hero, _ in all_match_heroes}
    total_target_mentions = num_keys * num_updates

    # Track which hero was last updated (no consecutive same-hero)
    last_updated_hero = None

    # Track game state for impossible-event filtering
    towers_destroyed = {"radiant": set(), "dire": set()}  # set of (lane, tier) tuples
    barracks_destroyed = {"radiant": set(), "dire": set()}
    roshan_kill_times = []  # list of game_time when Roshan was killed
    roshan_count = 0

    while sum(mention_counts.values()) < total_target_mentions and game_time < 60.0:

        phase = get_phase(game_time)
        weights = get_event_weights(phase, config.game_archetype)

        # ── STEP 8a: Filter impossible events ──
        possible_weights = dict(weights)

        # No barracks in laning or if no T3 towers destroyed
        if phase == "laning":
            possible_weights.pop("barracks_kill", None)
            possible_weights.pop("teamfight", None)
            possible_weights.pop("roshan_kill", None)

        # Roshan cooldown: min 8 minutes since last kill
        if roshan_kill_times and (game_time - roshan_kill_times[-1]) < 8:
            possible_weights.pop("roshan_kill", None)

        # No tower_kill if all 11 towers on one side destroyed
        # (simplified: check if any tower is available to destroy)
        # Tower destruction order: T1 before T2 before T3 per lane, per team
        # Implementation: maintain towers_destroyed set; filter if no valid target

        # No barracks_kill if the corresponding T3 tower is not destroyed
        # AND no T3 towers are destroyed for either team
        can_kill_barracks = False
        for team_key in ["radiant", "dire"]:
            for lane in ["top", "mid", "bot"]:
                if (lane, 3) in towers_destroyed[team_key]:
                    can_kill_barracks = True
        if not can_kill_barracks:
            possible_weights.pop("barracks_kill", None)

        # No level_up if all heroes are max level (30)
        if all(hero_states[h].level >= 30 for h, _ in all_match_heroes):
            possible_weights.pop("level_up", None)

        # Dead heroes can't farm/kill — check at event execution, not here

        # Renormalize
        total_w = sum(possible_weights.values())
        if total_w == 0:
            break
        probs = {k: v / total_w for k, v in possible_weights.items()}

        # ── STEP 8b: Pick event type ──
        event_type = weighted_random_choice(probs, rng)

        # ── STEP 8c: Pick involved heroes ──
        # Constraint: primary hero != last_updated_hero
        # Constraint: heroes must be alive for most events
        alive_heroes = [h for h, _ in all_match_heroes if hero_states[h].alive]

        # Select primary hero (the one whose tracked value will be mentioned)
        # Weight by inverse mention_count to balance mentions
        candidates = [h for h in alive_heroes if h != last_updated_hero]
        if not candidates:
            candidates = alive_heroes  # relax constraint if only 1 alive

        # Prefer heroes with fewer mentions (balancing)
        min_mentions = min(mention_counts[h] for h in candidates)
        priority_candidates = [h for h in candidates if mention_counts[h] <= min_mentions + 2]
        if priority_candidates:
            candidates = priority_candidates

        primary_hero = rng.choice(candidates)

        # ── STEP 8d: Execute event, update hero states ──
        # Each event handler returns:
        #   - event_data: dict with all template placeholders filled
        #   - state_changes: list of (hero, attribute, old_value, new_value, mentioned_in_narrative)

        event_data, state_changes = execute_event(
            event_type, primary_hero, hero_states, game_time, rng, config, tracked,
            towers_destroyed, barracks_destroyed, roshan_count,
            all_match_heroes, team1_name, team2_name
        )

        # ── STEP 8e: Record state changes ──
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
            if mentioned:
                key = f"{hero_name} / {tracked[hero_name]}"
                if key in entity_tracking:
                    # Only record if value actually changed from last recorded
                    if not entity_tracking[key] or entity_tracking[key][-1] != str(new_val):
                        entity_tracking[key].append(str(new_val))
                        mention_counts[hero_name] += 1

        event_log.append({
            "type": event_type,
            "time": round(game_time, 1),
            "data": event_data,
        })

        last_updated_hero = primary_hero

        # ── STEP 8f: Update game-level state ──
        if event_type == "tower_kill":
            lane = event_data["lane"]
            tier = event_data["tier"]
            defending_team = event_data["defending_team"]
            towers_destroyed[defending_team].add((lane, tier))
        elif event_type == "barracks_kill":
            lane = event_data["lane"]
            rax_type = event_data["rax_type"]
            defending_team = event_data["defending_team"]
            barracks_destroyed[defending_team].add((lane, rax_type))
        elif event_type == "roshan_kill":
            roshan_kill_times.append(game_time)
            roshan_count += 1

        # ── STEP 8g: Advance game_time ──
        time_range = EVENT_TIME_ADVANCE[event_type][phase]
        dt = rng.uniform(time_range[0], time_range[1])
        dt = max(0.3, min(3.0, dt))
        game_time += dt
        game_time = round(game_time, 1)

        # ── STEP 8h: Handle respawns ──
        for hero_name, _ in all_match_heroes:
            hs = hero_states[hero_name]
            if not hs.alive and hasattr(hs, 'respawn_at') and game_time >= hs.respawn_at:
                hs.alive = True

    # ── STEP 9: Render narrative ──
    narrative_parts = []

    # 9a: Match header
    header_template = rng.choice(HEADER_TEMPLATES)
    header = render_header(header_template, team1_name, team2_name,
                           radiant_heroes, dire_heroes, tournament_name,
                           config, rng)
    narrative_parts.append(header)
    narrative_parts.append("")  # blank line

    # 9b: Render events with filler and transitions
    current_phase = get_phase(event_log[0]["time"]) if event_log else "laning"
    filler_cfg = FILLER_CONFIG[config.filler_budget]

    for i, event in enumerate(event_log):
        # Check for phase transition
        event_phase = get_phase(event["time"])
        if event_phase != current_phase:
            transition = rng.choice(SEGMENT_TRANSITION_TEMPLATES)
            phase_descriptions = {
                "mid_game": "the mid-game rotations and objective play",
                "late_game": "the late-game high-ground sieges and decisive teamfights",
            }
            narrative_parts.append(render_template(transition, {
                "time": int(event["time"]),
                "old_phase": current_phase.replace("_", " "),
                "phase_description": phase_descriptions.get(event_phase, "a new phase"),
            }))
            narrative_parts.append("")
            current_phase = event_phase

        # Render event
        templates = get_templates_for_event(event["type"])
        template = rng.choice(templates)

        # Voice adjustment: if "caster", use present tense variants
        # if "analyst", use past tense (templates are written in past tense by default)
        # For caster voice, replace "pushed" -> "pushes", "fell" -> "falls", etc.
        # Implementation: maintain a TENSE_MAP dict for common verbs and apply substitution.
        # This is a cosmetic post-process; does not affect ground truth.

        rendered = render_template(template, event["data"])
        narrative_parts.append(rendered)

        # Maybe insert filler
        if rng.random() < filler_cfg["insert_probability"]:
            n_sentences = filler_cfg["sentences_per_insertion"]
            if isinstance(n_sentences, tuple):
                n_sentences = rng.randint(n_sentences[0], n_sentences[1])
            filler_templates = rng.sample(FILLER_TEMPLATES, min(n_sentences, len(FILLER_TEMPLATES)))
            for ft in filler_templates:
                filler_hero = rng.choice([h for h, _ in all_match_heroes])
                filler_rendered = render_filler(ft, filler_hero, hero_states, rng, tracked)
                narrative_parts.append(filler_rendered)

    narrative = " ".join(narrative_parts)

    # ── STEP 10: Generate questions ──
    queried_hero = all_match_heroes[config.queried_hero_idx][0]
    queried_attr = tracked[queried_hero]
    tracking_key = f"{queried_hero} / {queried_attr}"
    values = entity_tracking[tracking_key]

    # Ensure RI != PI (first != last). If equal, pick a different hero.
    if len(values) < 2 or values[0] == values[-1]:
        # Try other heroes
        for alt_hero, _ in all_match_heroes:
            alt_key = f"{alt_hero} / {tracked[alt_hero]}"
            alt_values = entity_tracking[alt_key]
            if len(alt_values) >= 2 and alt_values[0] != alt_values[-1]:
                queried_hero = alt_hero
                queried_attr = tracked[alt_hero]
                tracking_key = alt_key
                values = alt_values
                break

    ri_answer = values[0]
    pi_answer = values[-1]

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

    # ── STEP 11: Validate ──
    for key, vals in entity_tracking.items():
        for val in vals:
            assert val in narrative, f"Value '{val}' for '{key}' not found in narrative text"
    assert ri_answer != pi_answer, "RI and PI answers must differ"
    assert len(values) >= 2, "Need at least 2 mentions for queried entity"

    # ── STEP 12: Assemble and return ──
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
            "radiant_heroes": [(h, r) for h, r in radiant_heroes],
            "dire_heroes": [(h, r) for h, r in dire_heroes],
        },
    }
```

---

## 12. OUTPUT JSON SCHEMA

```json
{
    "id": "dota2_000042",
    "domain": "dota2",
    "num_keys": 5,
    "num_updates": 7,
    "narrative": "Game 1 of the Meridian Cup: Storm Vanguard (Anti-Mage, Lina, ...) ...",
    "questions": {
        "RI": {
            "question": "What was Anti-Mage's net_worth when first mentioned in this match?",
            "expected_answer": "1850",
            "target_entity": "Anti-Mage",
            "target_attribute": "net_worth"
        },
        "PI": {
            "question": "What was Anti-Mage's final recorded net_worth?",
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
    },
    "full_state_log": [
        {
            "time": 3.2,
            "event": "farming_update",
            "entity": "Anti-Mage",
            "attribute": "net_worth",
            "old_value": "600",
            "new_value": "1850",
            "mentioned_in_narrative": true
        },
        {
            "time": 3.2,
            "event": "farming_update",
            "entity": "Anti-Mage",
            "attribute": "last_hits",
            "old_value": "0",
            "new_value": "28",
            "mentioned_in_narrative": false
        }
    ],
    "mention_counts": {
        "Anti-Mage": 7,
        "Lina": 7,
        "Tidehunter": 7,
        "Invoker": 7,
        "Crystal Maiden": 7
    },
    "config": {
        "seed": 42,
        "num_keys": 5,
        "num_updates": 7,
        "condition": "RI",
        "tracked_attribute": "net_worth",
        "attribute_mode": "same",
        "filler_budget": "light",
        "voice": "analyst",
        "match_scope": "full",
        "game_archetype": "close",
        "queried_hero_idx": 0,
        "start_time": 0.0,
        "team1": "Storm Vanguard",
        "team2": "Eclipse Gaming",
        "tournament": "The Meridian Cup",
        "radiant_heroes": [["Anti-Mage", "pos1"], ["Lina", "pos2"], ["Tidehunter", "pos3"]],
        "dire_heroes": [["Invoker", "pos2"], ["Crystal Maiden", "pos5"]]
    }
}
```

**Field types and constraints:**

| Field | Type | Constraint |
|-------|------|-----------|
| `id` | str | `"dota2_{seed:06d}"` |
| `domain` | str | Always `"dota2"` |
| `num_keys` | int | 2-10 |
| `num_updates` | int | 3-50 |
| `narrative` | str | The full narrative text |
| `questions` | dict | Always has `"RI"` and `"PI"` keys |
| `questions.*.question` | str | Natural language question |
| `questions.*.expected_answer` | str | String representation of the answer |
| `questions.*.target_entity` | str | Hero name |
| `questions.*.target_attribute` | str | Attribute name from TRACKABLE_ATTRIBUTES |
| `entity_tracking` | dict | Keys: `"HeroName / attribute"`, Values: list of str |
| `entity_tracking` values | list[str] | All values appear in narrative, no consecutive duplicates, len >= 2 for queried entity |
| `full_state_log` | list[dict] | Every state change, including unmentioned ones |
| `full_state_log[].time` | float | Game time in minutes, 1 decimal |
| `full_state_log[].mentioned_in_narrative` | bool | True if value appears in narrative text |
| `mention_counts` | dict | Keys: hero names, Values: int count of tracked mentions |
| `config` | dict | Full config for reproducibility |

---

## 13. FILE STRUCTURE

```
mechanistic_probing_v2/
  core/
    narrative_generator/
      __init__.py              # Exports: DotaTrialGenerator, generate_dota_trial
      base.py                  # Abstract base class: NarrativeTrialGenerator
                               #   - abstract methods: generate_trial(), validate_trial()
                               #   - shared utilities: weighted_random_choice(),
                               #     render_template(), interpolate_range()
      dota2.py                 # DotaTrialGenerator(NarrativeTrialGenerator)
                               #   - generate_trial(num_keys, num_updates, condition, seed) -> dict
                               #   - All event handlers as methods
                               #   - All state management
                               #   - Narrative rendering
      data/
        dota2_heroes.json      # HERO_ROLES dict serialized as JSON
        dota2_items.json       # ITEMS dict + ROLE_ITEM_BUILDS serialized as JSON
        dota2_templates.json   # All template lists serialized as JSON:
                               #   header_templates, farming_templates, solo_kill_templates,
                               #   teamfight_templates, item_purchase_templates,
                               #   tower_kill_templates, roshan_templates, level_up_templates,
                               #   death_respawn_templates, barracks_kill_templates,
                               #   filler_templates, segment_transition_templates, kill_verbs
        dota2_names.json       # TEAM_NAMES, TOURNAMENT_NAMES, LOCATIONS
```

### base.py skeleton

```python
from abc import ABC, abstractmethod
import random

class NarrativeTrialGenerator(ABC):
    """Abstract base class for all domain-specific narrative generators."""

    @abstractmethod
    def generate_trial(self, num_keys: int, num_updates: int,
                       condition: str, seed: int) -> dict:
        """Generate a single trial. Returns dict matching output schema."""
        ...

    @abstractmethod
    def validate_trial(self, trial: dict) -> bool:
        """Validate all ground truth constraints. Raises AssertionError on failure."""
        ...

    @staticmethod
    def weighted_random_choice(weights: dict, rng: random.Random) -> str:
        items = list(weights.keys())
        cumulative = []
        total = 0
        for item in items:
            total += weights[item]
            cumulative.append(total)
        r = rng.random() * total
        for i, c in enumerate(cumulative):
            if r <= c:
                return items[i]
        return items[-1]

    @staticmethod
    def render_template(template: str, data: dict) -> str:
        """Fill {placeholder} in template with values from data dict.
        Missing keys are left as-is (for debugging)."""
        result = template
        for key, value in data.items():
            result = result.replace("{" + key + "}", str(value))
        return result

    @staticmethod
    def interpolate_range(trajectory: dict, time: float) -> tuple:
        """Linearly interpolate between two nearest time anchors in a trajectory dict."""
        times = sorted(trajectory.keys())
        if time <= times[0]:
            return trajectory[times[0]]
        if time >= times[-1]:
            return trajectory[times[-1]]
        for i in range(len(times) - 1):
            if times[i] <= time <= times[i + 1]:
                t0, t1 = times[i], times[i + 1]
                lo0, hi0 = trajectory[t0]
                lo1, hi1 = trajectory[t1]
                frac = (time - t0) / (t1 - t0)
                lo = int(lo0 + (lo1 - lo0) * frac)
                hi = int(hi0 + (hi1 - hi0) * frac)
                return (lo, hi)
        return trajectory[times[-1]]
```

### dota2.py class structure

```python
from dataclasses import dataclass, field
from typing import Optional
import random

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
    respawn_at: float = 0.0  # game_time when hero respawns

    def get_attribute(self, attr_name: str):
        """Return current value of any trackable attribute."""
        if attr_name == "kda_string":
            return f"{self.kills}/{self.deaths}/{self.assists}"
        elif attr_name == "gpm":
            # Must be passed game_time externally; stored as computed property
            return getattr(self, '_gpm', 0)
        elif attr_name == "xpm":
            return getattr(self, '_xpm', 0)
        elif attr_name == "latest_item":
            return self.items[-1] if self.items else "none"
        elif attr_name == "inventory_count":
            return len(self.items)
        elif attr_name == "buyback_status":
            return "available"  # simplified
        elif attr_name == "respawn_timer":
            return int(5 + 3.8 * self.level)
        elif attr_name == "net_worth_rank":
            return getattr(self, '_nw_rank', 1)
        elif attr_name == "gold_advantage":
            return getattr(self, '_gold_advantage', 0)
        else:
            return getattr(self, attr_name, 0)


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


class DotaTrialGenerator(NarrativeTrialGenerator):
    def __init__(self):
        # Load data from JSON files
        self.heroes = load_json("dota2_heroes.json")
        self.items = load_json("dota2_items.json")
        self.templates = load_json("dota2_templates.json")
        self.names = load_json("dota2_names.json")
        # Build role pools
        self.role_pool = self._build_role_pool()

    def generate_trial(self, num_keys, num_updates, condition, seed):
        # Implements the full flow from Section 11
        ...

    def validate_trial(self, trial):
        # Implements validation from Section 11 Step 11
        ...

    # Event handlers (one per event type):
    def _handle_farming_update(self, hero, state, game_time, rng, config, tracked): ...
    def _handle_solo_kill(self, killer, victim, states, game_time, rng, config, tracked): ...
    def _handle_teamfight(self, heroes, states, game_time, rng, config, tracked): ...
    def _handle_item_purchase(self, hero, state, game_time, rng, config, tracked): ...
    def _handle_tower_kill(self, team, states, game_time, rng, config, tracked, towers): ...
    def _handle_roshan_kill(self, team, states, game_time, rng, config, tracked, rosh_count): ...
    def _handle_level_up(self, hero, state, game_time, rng, config, tracked): ...
    def _handle_death_and_respawn(self, hero, state, game_time, rng, config, tracked): ...
    def _handle_barracks_kill(self, team, states, game_time, rng, config, tracked, towers, barracks): ...
```

---

## END OF SPECIFICATION

This document contains every data structure, constant, template, formula, and algorithmic step needed to implement the Dota 2 narrative interference generator. The developer should:

1. Create the file structure from Section 13.
2. Populate JSON data files from Sections 1-5.
3. Implement `base.py` from the skeleton in Section 13.
4. Implement `dota2.py` following the flow in Section 11, using the data structures and rules from Sections 7-10.
5. Write templates from Section 6 into `dota2_templates.json`.
6. Run validation (Section 11, Step 11) on every generated trial.
