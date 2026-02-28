# Remaining Domains -- Complete Mechanics Reference

This document provides exhaustive game/domain mechanics data for five domains used in narrative interference experiments.

---
---

# DOMAIN 1: StarCraft 2 (Legacy of the Void) -- Complete Game Mechanics Reference

All values are for the multiplayer Legacy of the Void (LotV) balance patch 5.0.14+ unless noted.

---

## 1. Overview

StarCraft 2 is a real-time strategy (RTS) game with three asymmetric races: **Terran**, **Protoss**, and **Zerg**. Each race has unique units, buildings, mechanics, and macro abilities. Players gather resources, build armies, and attempt to destroy their opponent's structures.

---

## 2. Resource Mechanics

### Minerals
- Workers mine **5 minerals per trip**
- Each standard base has **8 mineral patches**: 4 large (1800 minerals each) + 4 small (900 minerals each) = **10,800 minerals total per base**
- A worker occupies a mineral patch for ~2.0 seconds, pauses ~0.36 seconds, then returns
- Optimal saturation: **16 workers on minerals** (2 per patch); 3rd worker per patch yields diminishing returns (~30% efficiency)
- Fully saturated (16 workers): ~**1050-1140 minerals/min**; oversaturated (24 workers): ~**1200 minerals/min**

### Vespene Gas
- Workers gather **4 gas per trip**
- Each standard base has **2 Vespene Geysers** with **2250 gas each**
- A worker occupies a geyser for ~1.4 seconds before returning
- Optimal: **3 workers per geyser** (6 per base)
- Fully saturated (6 workers, 2 geysers): ~**325 gas/min**

### Starting Resources
- Each player starts with **50 minerals, 0 gas** (LotV standard, some maps vary)
- **12 starting workers** (LotV, up from 6 in WoL/HotS)

---

## 3. Supply System

- Maximum supply cap: **200** for all races
- Supply providers:
  - **Terran**: Supply Depot (100 min, 8 supply, build time 21s) -- can lower into ground
  - **Protoss**: Pylon (100 min, 8 supply, build time 18s) -- provides power field
  - **Zerg**: Overlord (100 min, 8 supply, build time 18s) -- flying unit, provides vision
- Base structures provide supply: Command Center/Nexus/Hatchery each provide 15 supply (Hatchery provides 6)
- Lair provides 6 supply, Hive provides 6 supply
- Overseer (morph from Overlord: 50 min 50 gas) provides 8 supply

---

## 4. Race Macro Mechanics

### Terran: MULE / Calldown Supply / Scanner Sweep
- **MULE**: Orbital Command ability, costs 50 energy. Drops a temporary worker that mines **25 minerals per trip** for 64 seconds (~4x SCV income). Cannot mine gas.
- **Calldown Extra Supplies**: Orbital Command ability, 50 energy. Permanently adds 8 supply to a Supply Depot.
- **Scanner Sweep**: Orbital Command ability, 50 energy. Reveals area and detects cloaked/burrowed units for 12 seconds.

### Protoss: Chrono Boost
- **Chrono Boost**: Nexus ability, costs 50 energy. Speeds targeted building's production by **50% for 20 seconds**. Effectively reduces build time to 2/3 for the boosted period. Energy regenerates at 0.5625 per second.

### Zerg: Spawn Larva (Inject)
- **Spawn Larva**: Queen ability, costs 25 energy. Injects 3 larva eggs into a Hatchery/Lair/Hive; they hatch after **29 seconds** into 3 larvae.
- Hatcheries passively produce 1 larva every **11 seconds** up to a cap of 3 (without injection).
- Maximum larvae with perfect inject: effectively unlimited production throughput.
- Queens auto-generate energy at 0.5625/s; 25 energy per inject means ~44s between injects per Queen.

---

## 5. Terran Units

### Workers
| Unit | Min | Gas | Supply | HP | Armor | Damage | Attack Speed | Range | Speed | Build Time |
|------|-----|-----|--------|----|-------|--------|-------------|-------|-------|------------|
| SCV | 50 | 0 | 1 | 45 | 0 | 5 | 1.07 | 0.1 (melee) | 3.94 | 12s |

### Infantry (Barracks)
| Unit | Min | Gas | Supply | HP | Armor | Damage | Attack Speed | Range | Speed | Build Time | Notes |
|------|-----|-----|--------|----|-------|--------|-------------|-------|-------|------------|-------|
| Marine | 50 | 0 | 1 | 45 | 0 | 6 (normal) | 0.61 | 5 | 3.15 | 18s | +Stim: +50% attack speed, -10 HP |
| Marauder | 100 | 25 | 2 | 125 | 1 | 10 (concussive) | 1.07 | 6 | 3.15 | 21s | +10 vs armored; Stim available |
| Reaper | 50 | 50 | 1 | 60 | 0 | 4x2 (normal) | 0.79 | 5 | 5.25 | 32s | Cliff jumping; HP regen out of combat |
| Ghost | 150 | 125 | 2 | 100 | 0 | 10 (normal) | 1.07 | 6 | 3.94 | 29s | Cloak, Snipe (170 dmg to bio), EMP, Nuke |

### Factory Units
| Unit | Min | Gas | Supply | HP | Armor | Damage | Attack Speed | Range | Speed | Build Time | Notes |
|------|-----|-----|--------|----|-------|--------|-------------|-------|-------|------------|-------|
| Hellion | 100 | 0 | 2 | 90 | 0 | 8 (line splash) | 1.79 | 5 | 5.95 | 21s | +6 vs light; transforms to Hellbat |
| Hellbat | 100 | 0 | 2 | 135 | 0 | 18 (cone splash) | 1.43 | 2 | 3.15 | 21s | +12 vs light; biological, healable |
| Widow Mine | 75 | 25 | 2 | 90 | 0 | 125+40 splash | 29s (rearm) | 5 (burrow) | 3.94 | 21s | Burrows; splash 40 to friendlies; +35 shields dmg |
| Siege Tank | 150 | 125 | 3 | 175 | 1 | 15 (normal) / 40+15 splash (siege) | 0.74 / 2.14 | 7 / 13 | 3.15 / 0 | 32s | Siege mode: immobile, massive splash |
| Cyclone | 150 | 100 | 3 | 120 | 1 | 18 (normal) | 0.71 | 5 | 4.73 | 32s | Lock-On ability: 400 dmg over 14s to air |
| Thor | 300 | 200 | 6 | 400 | 2 | 30+6 splash (ground) | 0.91 | 7 | 2.62 | 43s | Anti-air: 6+6 splash or 25+25 single target (switchable) |

### Starport Units
| Unit | Min | Gas | Supply | HP | Armor | Damage | Attack Speed | Range | Speed | Build Time | Notes |
|------|-----|-----|--------|----|-------|--------|-------------|-------|-------|------------|-------|
| Viking | 150 | 75 | 2 | 135 (air) / 135 (ground) | 0 | 10x2 (air mode, vs air) / 12 (ground mode) | 1.43 / 0.71 | 9 / 6 | 3.85 / 3.15 | 30s | Transforms air/ground |
| Medivac | 100 | 100 | 2 | 150 | 1 | 0 (healer) | -- | -- | 3.5 | 30s | Heals 12.6 HP/s; carries 8 units; Afterburners boost |
| Liberator | 150 | 150 | 3 | 180 | 0 | 5x2 (air-to-air splash) / 75 (defender mode, ground) | 1.29 / 1.14 | 5 / 10 | 4.72 / 0 | 43s | Defender mode: single-target zone lockdown |
| Banshee | 150 | 100 | 3 | 140 | 0 | 12x2 (ground only) | 0.89 | 6 | 3.85 | 43s | Cloak ability |
| Raven | 100 | 200 | 2 | 140 | 1 | 0 (support) | -- | -- | 3.85 | 34s | Auto-Turret, Interference Matrix, Anti-Armor Missile |
| Battlecruiser | 400 | 300 | 6 | 550 | 3 | 8 (air+ground) | 0.16 | 6 | 2.62 | 64s | Yamato Cannon: 240 dmg; Tactical Jump (teleport) |

### Terran Buildings (Tech Tree)
| Building | Min | Gas | Build Time | Prerequisite | Unlocks |
|----------|-----|-----|------------|-------------|---------|
| Command Center | 400 | 0 | 71s | None | SCV; morphs to Orbital/Planetary |
| Orbital Command | 150 | 0 | 25s | Barracks | MULE, Scanner, Supply Drop |
| Planetary Fortress | 150 | 150 | 36s | Engineering Bay | Ground attack (40 dmg) |
| Supply Depot | 100 | 0 | 21s | None | +8 supply |
| Barracks | 150 | 0 | 46s | Supply Depot | Marine, Marauder, Reaper, Ghost (w/ Ghost Academy) |
| Barracks Reactor | 50 | 50 | 36s | Barracks | Dual production (basic units only) |
| Barracks Tech Lab | 50 | 25 | 18s | Barracks | Marauder, Ghost; Stim, Combat Shield, Concussive |
| Factory | 150 | 100 | 43s | Barracks | Hellion, Widow Mine, Siege Tank, Cyclone, Thor (w/ Armory) |
| Starport | 150 | 100 | 36s | Factory | Viking, Medivac, Liberator, Banshee, Raven, BC (w/ Fusion Core) |
| Engineering Bay | 125 | 0 | 25s | Command Center | Infantry weapon/armor upgrades; Planetary Fortress |
| Armory | 150 | 100 | 46s | Factory | Vehicle/ship weapon/armor upgrades; Thor |
| Ghost Academy | 150 | 50 | 29s | Barracks | Ghost; Nuke (100/100, 43s) |
| Fusion Core | 150 | 150 | 46s | Starport | Battlecruiser; Yamato, Weapon Refit upgrades |
| Bunker | 100 | 0 | 29s | Barracks | Holds 4 infantry; Neosteel Frame (+2) |
| Missile Turret | 100 | 0 | 18s | Engineering Bay | Anti-air defense (42 DPS) |
| Sensor Tower | 125 | 100 | 18s | Engineering Bay | Reveals unit movement on minimap (30 range) |

---

## 6. Protoss Units

### Workers
| Unit | Min | Gas | Supply | HP | Shield | Armor | Damage | Attack Speed | Range | Speed | Build Time |
|------|-----|-----|--------|----|--------|-------|--------|-------------|-------|-------|------------|
| Probe | 50 | 0 | 1 | 20 | 20 | 0 | 5 | 1.07 | 0.1 | 3.94 | 12s |

### Gateway / Warp Gate Units
| Unit | Min | Gas | Supply | HP | Shield | Armor | Damage | Attack Speed | Range | Speed | Build Time | Notes |
|------|-----|-----|--------|----|--------|-------|--------|-------------|-------|-------|------------|-------|
| Zealot | 100 | 0 | 2 | 100 | 50 | 1 | 8x2 (melee) | 0.86 | 0.1 | 3.15 | 27s | Charge upgrade: dash to target, +8 impact dmg |
| Adept | 100 | 25 | 2 | 70 | 70 | 1 | 10 (+12 vs light) | 1.61 | 4 | 3.5 | 27s | Psionic Transfer: shade teleport |
| Stalker | 125 | 50 | 2 | 80 | 80 | 1 | 13 (+5 vs armored) | 1.34 | 6 | 4.13 | 30s | Blink: 8-range teleport |
| Sentry | 50 | 100 | 2 | 40 | 40 | 1 | 6 | 0.71 | 5 | 3.15 | 23s | Force Field, Guardian Shield (-2 ranged dmg), Hallucination |
| High Templar | 50 | 150 | 2 | 40 | 40 | 0 | 4 | 1.25 | 6 | 2.62 | 39s | Psionic Storm: 80 dmg over 2.85s (AoE); Feedback |
| Dark Templar | 125 | 125 | 2 | 40 | 80 | 1 | 45 (melee) | 1.21 | 0.1 | 3.94 | 39s | Permanently cloaked; Shadow Stride (blink) |
| Archon | 0* | 0* | 4 | 10 | 350 | 0 | 25 (+10 vs bio) (splash) | 1.25 | 3 | 3.94 | 9s | Merged from 2 HT or 2 DT; massive shield pool |

*Archon cost = cost of the two Templar merged

### Robotics Facility Units
| Unit | Min | Gas | Supply | HP | Shield | Armor | Damage | Attack Speed | Range | Speed | Build Time | Notes |
|------|-----|-----|--------|----|--------|-------|--------|-------------|-------|-------|------------|-------|
| Observer | 25 | 75 | 1 | 40 | 20 | 0 | 0 | -- | -- | 2.63 | 21s | Permanently cloaked detector |
| Warp Prism | 200 | 0 | 2 | 80 | 100 | 0 | 0 | -- | -- | 4.13 | 36s | Transport (4 slots); Phase mode: power field |
| Immortal | 275 | 100 | 4 | 200 | 100 | 1 | 20 (+30 vs armored) | 1.04 | 6 | 3.15 | 39s | Barrier: absorbs 100 dmg for 2s |
| Colossus | 300 | 200 | 6 | 200 | 150 | 1 | 10x2 (line splash) | 1.07 | 7 | 3.15 | 54s | Walks over cliffs; targetable by air attacks |
| Disruptor | 150 | 150 | 3 | 100 | 100 | 1 | 145 (Purification Nova AoE) | 14.3s cooldown | 13 | 3.15 | 36s | Nova: devastating AoE burst |

### Stargate Units
| Unit | Min | Gas | Supply | HP | Shield | Armor | Damage | Attack Speed | Range | Speed | Build Time | Notes |
|------|-----|-----|--------|----|--------|-------|--------|-------------|-------|-------|------------|-------|
| Phoenix | 150 | 100 | 2 | 120 | 60 | 0 | 5x2 (+5x2 vs light) | 0.79 | 5 | 5.95 | 25s | Graviton Beam: lifts unit; air-to-air only |
| Void Ray | 250 | 150 | 4 | 150 | 100 | 0 | 6 (+4 vs armored) | 0.36 | 6 | 3.5 | 43s | Prismatic Alignment: +6 vs armored for 14s |
| Oracle | 150 | 150 | 3 | 100 | 60 | 0 | 15 (+7 vs light, Pulsar Beam) | 0.61 | 4 | 5.6 | 37s | Revelation (detection); Stasis Ward (trap) |
| Tempest | 250 | 175 | 5 | 150 | 125 | 2 | 30 (+22 vs structures, air) / 40 (ground) | 2.36 / 2.36 | 14 / 10 | 3.5 | 43s | Extreme range siege unit |
| Carrier | 350 | 250 | 6 | 250 | 150 | 2 | 8 x8 Interceptors | varies | 8 (release) | 3.15 | 64s | Builds Interceptors (15 min each, max 8) |
| Mothership | 400 | 400 | 8 | 250 | 250 | 2 | 6x2 (splash) | 1.58 | 7 | 3.15 | 114s | Cloaking Field, Mass Recall, Time Warp; limit 1 |

### Protoss Buildings (Tech Tree)
| Building | Min | Gas | Build Time | Prerequisite | Unlocks |
|----------|-----|-----|------------|-------------|---------|
| Nexus | 400 | 0 | 71s | None | Probe; Chrono Boost; Mothership Core (removed) |
| Pylon | 100 | 0 | 18s | None | +8 supply; power field |
| Gateway | 150 | 0 | 46s | Nexus + Pylon | Zealot, Adept, Stalker, Sentry, HT, DT |
| Warp Gate | 0 | 0 | 7s (transform) | Cybernetics Core research | Warp-in units at any pylon |
| Cybernetics Core | 150 | 0 | 36s | Gateway | Warp Gate, Air Weapons/Armor, research |
| Forge | 150 | 0 | 32s | Nexus | Ground Weapons/Armor/Shields; Photon Cannon |
| Photon Cannon | 150 | 0 | 29s | Forge | Static defense (22 dmg, range 7) |
| Shield Battery | 100 | 0 | 29s | Cybernetics Core | Restores shields (50 energy) |
| Twilight Council | 150 | 100 | 36s | Cybernetics Core | Charge, Blink, Resonating Glaives; Dark Shrine, Templar Archives |
| Templar Archives | 150 | 200 | 36s | Twilight Council | High Templar; Psionic Storm |
| Dark Shrine | 150 | 150 | 71s | Twilight Council | Dark Templar; Shadow Stride |
| Robotics Facility | 200 | 100 | 46s | Cybernetics Core | Observer, Warp Prism, Immortal, Colossus, Disruptor |
| Robotics Bay | 200 | 200 | 46s | Robotics Facility | Colossus, Disruptor; Extended Thermal Lance |
| Stargate | 150 | 150 | 43s | Cybernetics Core | Phoenix, Void Ray, Oracle, Tempest, Carrier |
| Fleet Beacon | 300 | 200 | 43s | Stargate | Tempest, Carrier; Mothership; upgrades |

---

## 7. Zerg Units

### Workers / Support
| Unit | Min | Gas | Supply | HP | Armor | Damage | Attack Speed | Range | Speed | Build Time | Notes |
|------|-----|-----|--------|----|-------|--------|-------------|-------|-------|------------|-------|
| Drone | 50 | 0 | 1 | 40 | 0 | 5 | 1.07 | 0.1 | 3.94 | 12s | Morphs into buildings (consumed) |
| Overlord | 100 | 0 | 0 | 200 | 0 | 0 | -- | -- | 0.902 | 18s | +8 supply; slow flyer; upgrades: speed, transport |
| Overseer | 50 | 50 | 0 | 200 | 1 | 0 | -- | -- | 2.62 | 12s | Detector; Contaminate, Changeling |
| Queen | 150 | 0 | 2 | 175 | 1 | 9 (ground) / 9 (air) | 0.71 / 0.71 | 5 / 7 | 1.31 (off creep) / 2.67 (on creep) | 36s | Inject Larva, Transfuse (75 HP heal), Creep Tumor |

### Hatchery Units (Larvae-based)
| Unit | Min | Gas | Supply | HP | Armor | Damage | Attack Speed | Range | Speed | Build Time | Notes |
|------|-----|-----|--------|----|-------|--------|-------------|-------|-------|------------|-------|
| Zergling | 50/pair | 0 | 0.5 each | 35 | 0 | 5 (melee) | 0.497 | 0.1 | 4.72 (6.58 w/ speed) | 17s | Produced in pairs; Metabolic Boost: +60% speed; Adrenal Glands: +18% attack |
| Baneling | +25 | +25 | 0.5 | 30 | 0 | 16 (+19 vs light, splash) | suicide | 0.25 (splash) | 4.13 (4.72 w/ speed) | 14s (morph) | Suicide unit; morphs from Zergling |
| Roach | 75 | 25 | 2 | 145 | 1 | 16 | 1.43 | 4 | 3.15 (4.2 off creep w/ speed) | 19s | Burrow heal; Tunneling Claws (move while burrowed) |
| Ravager | +25 | +75 | 3 | 120 | 1 | 16 | 1.14 | 6 | 3.85 | 9s (morph) | Corrosive Bile: 60 dmg AoE, destroys Force Fields |
| Hydralisk | 100 | 50 | 2 | 90 | 0 | 12 (+5 vs air) | 0.59 | 5 (6 w/ range) | 3.15 | 24s | Muscular Augments: +speed on creep; Grooved Spines: +1 range |
| Lurker | +50 | +100 | 3 | 200 | 1 | 20 (+10 vs armored, line splash) | 1.43 | 8 (10 w/ upgrade) | 4.13 | 18s (morph) | Must burrow to attack; morphs from Hydralisk |
| Infestor | 100 | 150 | 2 | 90 | 0 | 0 | -- | -- | 3.15 | 36s | Fungal Growth (30 dmg AoE, roots), Neural Parasite, Microbial Shroud |
| Swarm Host | 100 | 75 | 3 | 160 | 1 | 0 (spawns Locusts) | 43s | -- | 3.15 | 29s | Spawns 2 Locusts (12 dmg each, 18s lifespan); Flying Locusts upgrade |
| Ultralisk | 300 | 200 | 6 | 500 | 2 | 35 (cleave splash) | 0.61 | 1 | 4.13 | 39s | Massive; Chitinous Plating (+2 armor); Anabolic Synthesis (+speed off creep) |

### Air Units
| Unit | Min | Gas | Supply | HP | Armor | Damage | Attack Speed | Range | Speed | Build Time | Notes |
|------|-----|-----|--------|----|-------|--------|-------------|-------|-------|------------|-------|
| Mutalisk | 100 | 100 | 2 | 120 | 0 | 9 (bounce: 9 → 3 → 1) | 0.72 | 3 | 5.6 | 24s | Attacks air+ground; bouncing glaive |
| Corruptor | 150 | 100 | 2 | 200 | 2 | 14 (+6 vs massive) | 1.36 | 6 | 4.72 | 29s | Air-to-air only; Caustic Spray (structure dmg) |
| Brood Lord | +150 | +150 | 4 | 225 | 1 | 20 (spawns Broodlings) | 1.79 | 9.5 | 2.24 | 24s (morph) | Morphs from Corruptor; spawns 2 melee Broodlings per attack |
| Viper | 100 | 200 | 3 | 150 | 1 | 0 | -- | -- | 4.13 | 29s | Abduct (pull), Blinding Cloud, Parasitic Bomb; consumes buildings for energy |

### Zerg Buildings (Tech Tree)
| Building | Min | Gas | Build Time | Prerequisite | Unlocks |
|----------|-----|-----|------------|-------------|---------|
| Hatchery | 300 | 0 | 71s | None | Drone, Overlord, Zergling (w/ Pool), Queen |
| Lair | 150 | 100 | 57s | Spawning Pool | Morphs from Hatchery; Overlord speed; Overseer; tier 2 |
| Hive | 200 | 150 | 71s | Infestation Pit | Morphs from Lair; tier 3 |
| Spawning Pool | 200 | 0 | 46s | Hatchery | Zergling, Queen; Metabolic Boost, Adrenal Glands |
| Extractor | 25 | 0 | 21s | None | Gas harvesting |
| Evolution Chamber | 75 | 0 | 25s | Hatchery | Melee/Ranged/Carapace upgrades; Spore Crawler |
| Baneling Nest | 100 | 50 | 43s | Spawning Pool | Baneling morph; Centrifugal Hooks |
| Roach Warren | 150 | 0 | 39s | Spawning Pool | Roach; Glial Reconstitution, Tunneling Claws |
| Hydralisk Den | 100 | 100 | 29s | Lair | Hydralisk; Muscular Augments, Grooved Spines |
| Lurker Den | 100 | 150 | 57s | Hydralisk Den | Lurker morph; Seismic Spines (+2 range) |
| Infestation Pit | 100 | 100 | 36s | Lair | Infestor, Swarm Host; Hive requirement |
| Spire | 200 | 200 | 71s | Lair | Mutalisk, Corruptor; air upgrades |
| Greater Spire | 100 | 150 | 71s | Hive + Spire | Brood Lord morph |
| Ultralisk Cavern | 150 | 200 | 46s | Hive | Ultralisk; Chitinous Plating, Anabolic Synthesis |
| Nydus Network | 150 | 200 | 36s | Lair | Nydus Worm (50/50): teleport units anywhere |
| Spine Crawler | 100 | 0 | 36s | Spawning Pool | Ground defense (25 dmg, range 7) |
| Spore Crawler | 75 | 0 | 21s | Evolution Chamber | Air defense (15 +15 vs bio, range 7); detector |

---

## 8. Standard Openings

### Terran Common Openers
- **1-1-1 (Barracks-Factory-Starport)**: Standard safe opener; Scout with Reaper, tech into any composition
- **2-1-1 (Marine-Medivac Timing)**: Double Barracks, Factory for Reactor swap, 2 Medivac Marine push ~5:00
- **CC First (Fast Expand)**: Greedy economic opener; Command Center before second Barracks
- **Proxy 2 Rax**: Aggressive; build 2 Barracks near opponent for early Marine/Marauder rush
- **Hellion/Banshee into Mech**: Factory-first into siege composition
- **Cloaked Banshee**: Early Starport with Tech Lab for harassment

### Protoss Common Openers
- **Gate-Expand (Gateway into Nexus)**: Standard macro opener
- **2-Gate Adept Pressure**: Early aggression with Adepts + Psionic Transfer
- **Stargate Oracle**: Oracle harassment into expand; Revelation for scouting
- **Robo Expand**: Fast Observer for scouting, into Immortal-based army
- **Blink Stalker All-In**: Twilight Council rush into mass Blink Stalkers
- **Proxy Void Ray**: Cheese strategy; Stargate near opponent

### Zerg Common Openers
- **Hatch First (16 Hatch)**: Standard; take natural expansion before Spawning Pool
- **Pool First (13/12 Pool)**: Aggressive; early Zerglings for pressure or defense
- **3 Hatch before Pool**: Extremely greedy economy opener
- **Ling-Bane-Muta (vs Terran)**: Speedling + Baneling aggression, Mutalisks for harassment
- **Roach-Ravager (vs Protoss)**: Robust ground army with Corrosive Bile
- **Ling Flood**: Mass Zerglings for early game kills

---

## 9. Macro / Micro Basics

### Macro Fundamentals
- **Spending**: Never let minerals/gas bank above 500-800 (floating = waste)
- **Worker production**: Continuous worker production until ~70-80 workers total
- **Supply management**: Build supply structures ahead of demand (avoid supply block)
- **Expansion timing**: Typically 3-4 bases saturated by mid-game
- **Upgrades**: Start combat upgrades by ~5:00; +1 attack is ~160s research time
- **Production**: Match production structures to income (rule of thumb: 3 Barracks per active base)

### Micro Fundamentals
- **Stutter-step**: Attack-move-attack to maximize DPS while kiting
- **Focus fire**: Concentrate fire on single targets to remove them faster
- **Concave vs convex**: Spread army in arc to maximize surface area
- **Spell usage**: Storm, EMP, Fungal in engagements can decide fights
- **Drop harassment**: Medivac/Warp Prism drops in multiple locations to divide opponent
- **Split**: Spread units vs splash damage (Banelings, Disruptors, Storms)

---
---

# DOMAIN 2: Among Us -- Complete Game Mechanics Reference

All values are for the current version (v2024.11+).

---

## 1. Overview

Among Us is a social deduction game for **4-15 players** set on a spaceship/base/airship. Players are divided into **Crewmates** and **Impostors**. Crewmates complete tasks and identify Impostors; Impostors kill Crewmates and sabotage.

---

## 2. Player Counts and Win Conditions

### Player Count
- Minimum: **4 players**
- Maximum: **15 players**
- Impostor count: **1, 2, or 3** (host configurable)

### Crewmate Win Conditions
1. **Complete all tasks** (taskbar fills to 100%)
2. **Vote out all Impostors** via emergency meetings

### Impostor Win Conditions
1. **Kill enough Crewmates** so that Impostors equal or outnumber surviving Crewmates
2. **Critical sabotage not resolved** in time (Reactor Meltdown, O2 Depletion -- NOT Lights or Comms)

---

## 3. Roles

### Crewmate Roles

| Role | Description | Special Ability |
|------|-------------|-----------------|
| **Crewmate** (default) | Standard role | Complete tasks, call meetings, vote |
| **Engineer** | Can use vents | Use vent system like Impostors (configurable cooldown: 0-60s) |
| **Scientist** | Remote vitals access | View Vitals panel from anywhere; battery depletes, recharges by completing tasks |
| **Noisemaker** | Alert on death | When killed, emits a noise/alert visible to all players showing death location |
| **Tracker** | Track player movement | Place a tracking device on a player to see their movement on minimap |

### Impostor Roles

| Role | Description | Special Ability |
|------|-------------|-----------------|
| **Impostor** (default) | Standard killer | Kill, Sabotage, Vent, fake tasks |
| **Shapeshifter** | Disguise | Temporarily take the appearance of another player (configurable duration: 10-30s); leaves evidence (shapeshifting animation and evidence pile) |
| **Phantom** | Invisibility | Temporarily become invisible to other players; partially visible at close range |

### Ghost Roles

| Role | Description | Special Ability |
|------|-------------|-----------------|
| **Guardian Angel** | Protect from beyond | Dead Crewmate can protect one living player from being killed; cooldown 60s, protection lasts ~12s |

### Role Counts (Configurable)
- Each role count: 0-15 (host sets max per game)
- Role chance: 0-100% per game

---

## 4. Core Gameplay Mechanics

### Kill Mechanics
- **Kill Cooldown**: Configurable 10s, 12.5s, 15s, 17.5s, 20s, 22.5s, 25s, 27.5s, 30s, 32.5s, 35s, 37.5s, 40s, 42.5s, 45s, 47.5s, 50s, 52.5s, 55s, 57.5s, 60s
- **Kill Distance**: Short, Medium, or Long
- Kill cooldown starts at match beginning AND after each meeting
- Impostors must be within kill range and have direct line of sight

### Vision Settings
- **Crewmate Vision**: 0.25x to 5x (default 1.0x)
- **Impostor Vision**: 0.25x to 5x (default 1.5x)
- Lights sabotage reduces Crewmate vision dramatically; does NOT affect Impostor vision

### Movement
- **Player Speed**: 0.5x to 3x (default 1.0x)
- All players have the same speed regardless of role

### Meeting Mechanics
- **Emergency Meetings**: 0-9 per player (configurable; default 1)
- **Emergency Cooldown**: 0-60s after start of game / after previous meeting
- **Report**: Any player can report a dead body (no cooldown)
- **Discussion Time**: 0-120s (mandatory discussion before voting opens)
- **Voting Time**: 0-300s (time to cast votes; 0 = unlimited)
- **Anonymous Votes**: On/Off (configurable)
- **Confirm Ejects**: On/Off (announces if ejected player was Impostor)

---

## 5. Task System

### Task Categories
- **Common Tasks**: Assigned to ALL players if enabled. If one player has it, everyone does. This is a key way to catch fakers.
- **Short Tasks**: Single-step or quick tasks (a few seconds)
- **Long Tasks**: Multi-step tasks that require visiting multiple locations or waiting
- **Visual Tasks**: Produce a visible animation that other players can witness (proves innocence). Can be toggled on/off.

### Task Count Settings
- Common Tasks: 0-2
- Short Tasks: 0-5
- Long Tasks: 0-3

### The Skeld Tasks

**Common Tasks:**
- Fix Wiring (3 stages across random rooms)
- Swipe Card (Admin)

**Short Tasks:**
- Clean O2 Filter (O2)
- Clear Asteroids (Weapons) [VISUAL]
- Divert Power (Electrical → destination)
- Empty Garbage (Cafeteria or O2 → Storage) [VISUAL at Storage stage]
- Prime Shields (Shields) [VISUAL]
- Stabilize Steering (Navigation)
- Unlock Manifolds (Reactor)
- Calibrate Distributor (Electrical)
- Chart Course (Navigation)
- Align Engine Output (Upper/Lower Engine)
- Start Reactor (Reactor -- memory game, may be long depending on classification)

**Long Tasks:**
- Submit Scan (MedBay) [VISUAL]
- Fuel Engines (Storage → Upper/Lower Engine)
- Upload Data (various rooms → Admin)
- Inspect Sample (MedBay -- 60 second wait)

### MIRA HQ Tasks

**Common Tasks:**
- Fix Wiring (3 stages)
- Enter ID Code (Admin)

**Short Tasks:**
- Divert Power (Reactor → destination)
- Clear Asteroids (Balcony)
- Sort Samples (Laboratory)
- Prime Shields (Admin)
- Buy Beverage (Cafeteria)
- Clean O2 Filter (Greenhouse)
- Chart Course (Admin)
- Measure Weather (Balcony)
- Fuel Engines (Launchpad -- short on this map)

**Long Tasks:**
- Submit Scan (MedBay) [VISUAL]
- Water Plants (Storage → Greenhouse)
- Assemble Artifact (various rooms)
- Start Reactor (Reactor -- memory game)
- Process Data (Office)
- Run Diagnostics (Launchpad)
- Unlock Manifolds (Reactor)

### Polus Tasks

**Common Tasks:**
- Fix Wiring (3 stages)
- Swipe Card (Office)
- Insert Keys (Dropship)
- Scan Boarding Pass (Office)

**Short Tasks:**
- Fix Weather Node (outside locations)
- Fill Canisters (O2)
- Record Temperature (outside locations)
- Repair Drill (outside, near Laboratory)
- Align Telescope (outside, near Communications)
- Chart Course (Dropship)
- Clear Asteroids (Weapons)
- Monitor Tree (O2)
- Unlock Manifolds (Specimen Room)
- Store Artifacts (Specimen Room)
- Open Waterways (Boiler Room)
- Reboot WiFi (Communications)

**Long Tasks:**
- Submit Scan (MedBay) [VISUAL]
- Fuel Engines (Storage → outside locations)
- Upload Data (various rooms → Office)
- Inspect Sample (Laboratory -- 60 second wait)
- Replace Water Jug (Boiler Room → Office)
- Start Reactor (Specimen Room -- memory game)

### The Airship Tasks

**Common Tasks:**
- Fix Wiring (3 stages)
- Enter ID Code (various)

**Short Tasks:**
- Decontaminate (Ventilation)
- Dress Mannequin (Main Hall)
- Develop Photos (Main Hall)
- Divert Power (Electrical → destination)
- Empty Garbage (Kitchen, Main Hall)
- Make Burger (Kitchen)
- Pick Up Towels (various rooms)
- Polish Ruby (Vault)
- Put Away Pistols (Armory)
- Put Away Rifles (Armory)
- Reset Breakers (Electrical)
- Rewind Tapes (Security)
- Sort Records (Records)
- Stabilize Steering (Cockpit)
- Unlock Safe (Cargo Bay)

**Long Tasks:**
- Upload Data (various rooms → Cockpit/Records)
- Fuel Engines (Storage → Engine Room)
- Calibrate Distributor (Electrical)
- Download Data (various → Cockpit)

**Note**: The Airship has NO visual tasks.

### The Fungle Tasks

**Common Tasks:**
- Fix Wiring (3 stages)

**Short Tasks:**
- Catch Fish (Dock)
- Collect Vegetables (Greenhouse)
- Cook (Kitchen)
- Crank Generator (Jungle)
- Grind Gems (Mining Pit)
- Monitor Mushroom (Jungle)
- Play Video Game (Dorm)
- Polish Gem (Laboratory)
- Record Temperature (Splash Zone)
- Retrieve Data (Communications)
- Sort Samples (Laboratory)
- Store Artifacts (Storage)
- Test Samples (Laboratory)

**Long Tasks:**
- Hoist Supplies (Lookout → Cliffs)
- Replace Parts (Laboratory → various)
- Fuel (Storage → Upper Engine)
- Extract Fuel (Mining Pit)
- Assemble Artifact (multiple rooms)

---

## 6. Sabotage Types

### Critical Sabotages (can cause Impostor win)
| Sabotage | Timer | Fix Method | Maps |
|----------|-------|------------|------|
| **Reactor Meltdown** | 30-45s (map dependent) | Two players simultaneously hold buttons at Reactor | Skeld, MIRA HQ, Polus (Seismic Stabilizers), Fungle |
| **O2 Depletion** | 30s | Enter codes at two separate O2 panels | Skeld, MIRA HQ |
| **Mushroom Mixup** | ~10s | Cosmetic disguise sabotage | Fungle (non-lethal) |

### Utility Sabotages (cannot directly win)
| Sabotage | Effect | Fix Method | Maps |
|----------|--------|------------|------|
| **Lights** | Reduces Crewmate vision to near-zero | Toggle switches at Electrical | Skeld, Polus, Airship, Fungle |
| **Communications** | Disables task list and Admin/Doorlog/Vitals | Turn dial to match frequency | Skeld, MIRA HQ, Polus, Fungle |
| **Doors** | Closes room doors temporarily (~10s) | Wait or toggle switches | Skeld, Polus, Airship |
| **Crash Course** | Map-specific | Varies | Airship |
| **Avert Crash Course** | Cockpit sabotage | Fix at cockpit | Airship |

### Sabotage Cooldown
- **30 seconds** between sabotages (default, some maps vary)
- Sabotage is available during meetings only for door closures
- Cannot sabotage during meetings

---

## 7. Maps -- Complete Room Lists

### The Skeld (14 rooms)
1. Cafeteria (start, emergency button)
2. Weapons
3. Navigation
4. O2
5. Shields
6. Communications
7. Storage
8. Admin
9. Electrical
10. Lower Engine
11. Upper Engine
12. Reactor
13. MedBay
14. Security

**Vent Connections**: Cafeteria ↔ Admin ↔ Hallway (above Shields); MedBay ↔ Electrical ↔ Security; Weapons ↔ Navigation; Upper Engine ↔ Reactor ↔ Lower Engine

### MIRA HQ (14 rooms)
1. Launchpad
2. Medbay
3. Communications
4. Locker Room
5. Decontamination
6. Laboratory
7. Reactor
8. Office
9. Admin
10. Greenhouse
11. Balcony
12. Cafeteria (emergency button)
13. Storage
14. Hallway (large Y-shaped corridor connecting all areas)

**Vent Connections**: ALL vents are interconnected in one system. Vents in: Balcony, Cafeteria, MedBay, Launchpad, Admin, Greenhouse, Office, Laboratory, Reactor, Decontamination. NOT in: Locker Room, Communications, Storage.

**Unique Feature**: Door Log (tracks who passes through Y-corridor sensors) replaces Security cameras.

### Polus (16 rooms/areas)
1. Dropship
2. Office (emergency button)
3. Admin
4. Communications
5. Weapons
6. O2
7. Electrical
8. Storage
9. Boiler Room
10. Laboratory
11. Specimen Room
12. MedBay (inside Laboratory)
13. Decontamination (x2: one near Admin, one near Laboratory)
14. Outside (multiple outdoor areas between buildings)
15. Security (camera room near Electrical)
16. Toilet (near Dropship)

**Vent Connections**: Security ↔ Electrical ↔ O2; Admin ↔ outside near Electrical; Laboratory ↔ outside near Admin; Office ↔ Storage ↔ Communications; Specimen Room ↔ outside near Specimen

### The Airship (21 rooms)
1. Brig
2. Engine Room
3. Kitchen
4. Main Hall (spawn option)
5. Meeting Room (emergency button)
6. Records (spawn option)
7. Cargo Bay
8. Communications
9. Cockpit
10. Armory
11. Viewing Deck
12. Vault
13. Gap Room
14. Lounge
15. Showers
16. Medical
17. Electrical
18. Storage
19. Security
20. Ventilation
21. Engine Room (spawn option)

**Unique Features**: Spawn choice (3 locations: Records, Main Hall, Engine Room); Ladders between floors; Moving platform connecting left-right sides; Largest map.

### The Fungle (18 rooms/areas)
1. Beach (start)
2. Splash Zone
3. Kitchen
4. Dock
5. Meeting Room (emergency button, conch shell)
6. Dorm
7. Storage
8. Cafeteria
9. Jungle
10. Greenhouse
11. Laboratory
12. Lookout
13. Mining Pit
14. Reactor
15. Communications
16. Upper Engine
17. The Cliffs
18. Campfire (near Beach)

**Vent Connections**: Splash Zone ↔ Cafeteria ↔ Storage ↔ Laboratory; Kitchen ↔ Jungle ↔ outside Greenhouse; Lookout ↔ Reactor ↔ Communications.

**Unique Features**: Mushroom Mixup sabotage (all players look identical); first outdoor map; hide-in-mushroom mechanic.

---

## 8. Security Systems by Map

| Map | Cameras | Admin Table | Vitals | Door Log |
|-----|---------|-------------|--------|----------|
| The Skeld | Yes (Security) | Yes (Admin) | No | No |
| MIRA HQ | No | Yes (Admin) | No | Yes (Communications) |
| Polus | Yes (outside, Security) | Yes (Admin) | Yes (Office) | No |
| The Airship | Yes (Security) | Yes (Cockpit) | Yes (Medical) | No |
| The Fungle | Yes (Lookout) | Yes (Office) | Yes (Laboratory) | No |

---
---

# DOMAIN 3: FIFA / Football Manager -- Complete Mechanics Reference

This domain covers both the FIFA/EA FC video game player attribute system and the Football Manager simulation. Values and structures reference the Premier League model as a canonical example.

---

## 1. League Structure (English Premier League Model)

### Basic Format
- **20 teams** in the top division
- **38 matchdays** per season (each team plays every other team twice: once home, once away)
- Season runs **August to May**

### Points System
| Result | Points |
|--------|--------|
| Win | **3** |
| Draw | **1** |
| Loss | **0** |

### Standings / Tiebreakers (in order)
1. Total points
2. Goal difference (goals scored minus goals conceded)
3. Goals scored
4. Head-to-head points
5. Head-to-head goal difference
6. Head-to-head away goals
7. Playoff match at neutral venue

### Promotion / Relegation
- Bottom **3 teams** are relegated to the Championship (second tier)
- Top **2 Championship teams** are automatically promoted
- Championship places **3rd-6th** enter a **playoff** (semi-finals + final at Wembley) for the third promotion spot

### European Qualification (from Premier League, 2024-25 onward)
| Position | Competition |
|----------|-------------|
| 1st-4th | UEFA Champions League (group/league stage) |
| 5th | UEFA Europa League (group stage) |
| 6th | UEFA Conference League (qualifying/group) |
| FA Cup winner | Europa League (if not already qualified for CL) |
| League Cup winner | Conference League (if not already qualified) |

---

## 2. Match Simulation Mechanics

### Match Duration
- **90 minutes** regulation (two 45-minute halves)
- **Added time** (stoppage time): typically 1-6 minutes per half (referee discretion)
- Extra time (knockout only): 2 x 15 minutes
- Penalty shootout (if still tied after extra time): best of 5, then sudden death

### Substitutions
- **5 substitutes** allowed per match (implemented 2022+), used in **3 windows** (plus halftime)
- Additional substitute allowed in extra time
- Squad registration: typically **25 players** + unlimited under-21 homegrown players

### Cards and Discipline
| Card | Consequence |
|------|-------------|
| **Yellow Card** | Caution; 2 yellows in one match = red card; accumulation (5 yellows = 1-match ban in PL) |
| **Red Card** | Immediate dismissal; minimum 1-match ban; violent conduct = 3+ match ban |
| **Straight Red** | Serious foul play, violent conduct, denying goal-scoring opportunity, offensive language |

### Goals and Events
- **Goal**: ball fully crosses goal line between posts and under crossbar
- **Offside**: attacking player nearer to opponent's goal line than both the ball and second-to-last defender when ball is played (excludes arms/hands)
- **Penalty**: foul inside 18-yard box; kick from penalty spot (12 yards / 11 meters from goal)
- **Free Kick**: direct (can score) or indirect (must touch another player)
- **Corner Kick**: when ball crosses goal line last touched by defending team
- **Throw-In**: ball crosses touchline; thrown in with both hands from behind head

---

## 3. Player Attributes -- FIFA / EA FC System (1-99 Scale)

### Six Face Card Stats
Each outfield player has 6 headline attributes displayed on their card, each rated **1-99**:

#### PAC (Pace)
| Sub-Attribute | Description |
|---------------|-------------|
| Acceleration | How quickly a player reaches top speed (1-99) |
| Sprint Speed | Maximum running velocity (1-99) |

#### SHO (Shooting)
| Sub-Attribute | Description |
|---------------|-------------|
| Positioning | Ability to find space in the box (1-99) |
| Finishing | Accuracy of shots inside the box (1-99) |
| Shot Power | Strength behind strikes (1-99) |
| Long Shots | Accuracy of shots from outside the box (1-99) |
| Volleys | Ability to hit clean volleys (1-99) |
| Penalties | Skill at taking penalty kicks (1-99) |

#### PAS (Passing)
| Sub-Attribute | Description |
|---------------|-------------|
| Vision | Ability to see and execute creative passes (1-99) |
| Crossing | Accuracy of crosses from wide areas (1-99) |
| Free Kick Accuracy | Precision on free kicks (1-99) |
| Short Passing | Accuracy of short passes (1-99) |
| Long Passing | Accuracy of long-range passes (1-99) |
| Curve | Ability to curl the ball (1-99) |

#### DRI (Dribbling)
| Sub-Attribute | Description |
|---------------|-------------|
| Agility | Ability to change direction quickly (1-99) |
| Balance | Ability to stay upright under pressure (1-99) |
| Reactions | Speed of response to events (1-99) |
| Ball Control | First touch and close control (1-99) |
| Dribbling | Skill moving with the ball (1-99) |
| Composure | Calmness under pressure (1-99) |

#### DEF (Defending)
| Sub-Attribute | Description |
|---------------|-------------|
| Interceptions | Reading and intercepting passes (1-99) |
| Heading Accuracy | Ability to direct headers (1-99) |
| Defensive Awareness | Positional intelligence in defense (1-99) |
| Standing Tackle | Success rate of standing tackles (1-99) |
| Sliding Tackle | Success rate of sliding tackles (1-99) |

#### PHY (Physical)
| Sub-Attribute | Description |
|---------------|-------------|
| Jumping | Leaping ability (1-99) |
| Stamina | Endurance over 90 minutes (1-99) |
| Strength | Physical power in duels (1-99) |
| Aggression | Intensity of challenges and pressing (1-99) |

### Goalkeeper-Specific Attributes
| Attribute | Description |
|-----------|-------------|
| Diving (DIV) | Ability to make diving saves (1-99) |
| Handling (HAN) | Ability to catch and hold the ball (1-99) |
| Kicking (KIC) | Distribution accuracy and distance (1-99) |
| Reflexes (REF) | Reaction speed to close-range shots (1-99) |
| Speed (SPD) | Movement speed off the line (1-99) |
| Positioning (POS) | Shot-stopping positioning (1-99) |

Total: **6 face stats**, **29 detailed sub-attributes** for outfield players, **6 GK attributes**.

---

## 4. Player Attributes -- Football Manager System (1-20 Scale)

### Technical Attributes (14)
| Attribute | Description |
|-----------|-------------|
| Corners | Ability to take corner kicks |
| Crossing | Ability to deliver crosses from wide positions |
| Dribbling | Ability to run with the ball and beat opponents |
| Finishing | Ability to put the ball in the net from scoring chances |
| First Touch | Quality of first touch when receiving the ball |
| Free Kick Taking | Ability to take free kicks |
| Heading | Ability to head the ball |
| Long Shots | Ability to shoot from distance |
| Long Throws | Ability to throw the ball long distances |
| Marking | Ability to mark opposing players |
| Passing | Ability to find teammates with passes |
| Penalty Taking | Ability to take penalty kicks |
| Tackling | Ability to win the ball through tackles |
| Technique | Technical skill level for ball manipulation |

### Mental Attributes (14)
| Attribute | Description |
|-----------|-------------|
| Aggression | Intensity and willingness to engage physically |
| Anticipation | Ability to predict events and react |
| Bravery | Willingness to put body on the line |
| Composure | Calmness in pressure situations |
| Concentration | Focus maintained throughout the match |
| Decisions | Quality of choices made on the pitch |
| Determination | Drive and desire to succeed |
| Flair | Tendency to try creative or unexpected moves |
| Leadership | Ability to inspire and organize teammates |
| Off the Ball | Movement and positioning without the ball |
| Positioning | Defensive positioning intelligence |
| Teamwork | Willingness to follow team instructions |
| Vision | Ability to see potential passes and opportunities |
| Work Rate | Effort and industry throughout the match |

### Physical Attributes (8)
| Attribute | Description |
|-----------|-------------|
| Acceleration | How quickly a player reaches top speed |
| Agility | Ability to change direction at speed |
| Balance | Ability to remain stable on feet |
| Jumping Reach | Vertical leap ability |
| Natural Fitness | Natural recovery rate and injury resistance |
| Pace | Top speed when running |
| Stamina | Endurance over a full match |
| Strength | Physical power in challenges |

### Goalkeeper Attributes (FM-specific, 1-20)
| Attribute | Description |
|-----------|-------------|
| Aerial Reach | Ability to reach high balls |
| Command of Area | Authority in the penalty area |
| Communication | Organizing the defense vocally |
| Eccentricity | Tendency to do unexpected things |
| First Touch (GK) | Handling incoming passes |
| Handling | Ability to hold onto the ball |
| Kicking | Distribution quality |
| One on Ones | Ability in 1v1 situations |
| Passing (GK) | Distribution accuracy |
| Punching | Tendency and ability to punch the ball |
| Reflexes | Reaction speed |
| Rushing Out | Willingness to leave the goal line |
| Throwing | Throwing distribution quality |

**Total**: 14 Technical + 14 Mental + 8 Physical = **36 outfield attributes** + **13 GK attributes**, all on **1-20 scale** (where 1 = terrible, 10 = average, 15 = very good, 18-20 = world class).

---

## 5. Formations

### Common Formations (Defense - Midfield - Attack)

| Formation | Structure | Style | Key Strengths | Key Weaknesses |
|-----------|-----------|-------|---------------|----------------|
| **4-4-2** | 4 def, 4 mid, 2 fwd | Balanced, traditional | Solid width, 2 strikers for aerial threat | Outnumbered in midfield (2v3) |
| **4-3-3** | 4 def, 3 mid, 3 fwd | Possession, attacking | Width from wingers, midfield triangle | Requires disciplined wide forwards |
| **4-2-3-1** | 4 def, 2 DM, 3 AM, 1 fwd | Modern balanced | Double pivot security, creative #10 | Lone striker can be isolated |
| **3-5-2** | 3 CB, 2 WB, 3 mid, 2 fwd | Wing-back heavy | Midfield overload, 5 across middle | Wing-backs must cover huge ground |
| **3-4-3** | 3 CB, 4 mid, 3 fwd | Attacking, wing-back | Outnumbers in attack, wide overloads | Vulnerable on flanks if WBs caught high |
| **4-1-4-1** | 4 def, 1 DM, 4 mid, 1 fwd | Defensive, counter | DM shield, compact midfield | Limited attacking numbers |
| **5-3-2** | 5 def (3CB + 2WB), 3 mid, 2 fwd | Defensive, catenaccio | Very solid defensively | Can struggle to create going forward |
| **4-3-2-1** | 4 def, 3 mid, 2 AM, 1 fwd | "Christmas Tree" | Packed midfield, creative options | Narrow, lacks natural width |
| **4-4-2 Diamond** | 4 def, 1 DM, 2 CM, 1 AM, 2 fwd | Central overload | Dominant through the middle | No natural wide players |
| **4-1-2-1-2** | 4 def, 1 DM, 2 CM, 1 AM, 2 fwd | Narrow, midfield control | Strong central presence | Requires overlapping fullbacks for width |
| **3-4-1-2** | 3 CB, 2 WB, 2 CM, 1 AM, 2 fwd | Balanced 3-at-back | Good blend of defense and attack | WB fitness demands |
| **4-2-2-2** | 4 def, 2 DM, 2 AM, 2 fwd | Box midfield | Dual striker threat, midfield rectangle | Can be exposed in wide areas |

---

## 6. Injury Types and Recovery Times

| Injury | Severity | Typical Recovery |
|--------|----------|-----------------|
| **ACL Tear** | Severe | 6-12 months (surgery required) |
| **MCL Sprain (Grade I)** | Mild | 1-3 weeks |
| **MCL Sprain (Grade II)** | Moderate | 4-6 weeks |
| **MCL Tear (Grade III)** | Severe | 3-4 months (possible surgery) |
| **Hamstring Strain (Grade I)** | Mild | 1-3 weeks |
| **Hamstring Strain (Grade II)** | Moderate | 4-8 weeks |
| **Hamstring Tear (Grade III)** | Severe | 2-3 months |
| **Groin Strain** | Mild-Moderate | 2-6 weeks |
| **Calf Strain** | Mild-Moderate | 1-6 weeks |
| **Ankle Sprain (Grade I)** | Mild | 1-2 weeks |
| **Ankle Sprain (Grade II)** | Moderate | 3-4 weeks |
| **Ankle Sprain (Grade III)** | Severe | 6-8 weeks |
| **High Ankle Sprain** | Severe | 6-12 weeks |
| **Metatarsal Fracture** | Severe | 6-10 weeks |
| **Tibia/Fibula Fracture** | Severe | 3-6 months |
| **Concussion** | Varies | 1-4 weeks (graduated return protocol) |
| **Knee Meniscus Tear** | Moderate-Severe | 4-8 weeks (partial); 3-6 months (full repair) |
| **Achilles Tendon Rupture** | Severe | 6-9 months |
| **Quadriceps Strain** | Mild-Moderate | 2-6 weeks |
| **Hip Flexor Strain** | Mild-Moderate | 2-6 weeks |
| **Dislocated Shoulder** | Moderate | 2-4 weeks (non-surgical) |
| **Broken Collarbone** | Moderate | 6-8 weeks |
| **Bruised Ribs** | Mild | 2-4 weeks |
| **Back Muscle Spasm** | Mild | 1-2 weeks |

---

## 7. Transfer System

### Transfer Windows
- **Summer Window**: typically June 9 - August 31 (varies by country)
- **January Window**: January 1 - January 31
- **Free Agents**: can be signed anytime outside windows

### Transfer Types
| Type | Description |
|------|-------------|
| **Permanent Transfer** | Full ownership changes clubs; fee negotiated |
| **Loan** | Temporary move (6 months or 1 season); may include loan fee, wage contribution, option/obligation to buy |
| **Free Transfer** | Player out of contract; no transfer fee (signing bonus and agent fees still apply) |
| **Player Exchange** | Players swap clubs (with or without additional cash) |
| **Buy-back Clause** | Selling club retains right to re-sign player at set price |
| **Sell-on Clause** | Selling club receives % of future transfer fee |
| **Release Clause** | Pre-agreed fee that club MUST accept if met |

### Contract Parameters (Football Manager)
| Parameter | Range |
|-----------|-------|
| Wage | Per week (e.g., $5k-$500k+/week) |
| Contract Length | 1-5 years |
| Signing-on Fee | Lump sum or annual installments |
| Squad Status | Key Player, First Team, Rotation, Backup, Hot Prospect |
| Release Clause | Optional; minimum fee for mandatory sale |
| Performance Bonuses | Goals, assists, appearances, clean sheets |
| Loyalty Bonus | Paid if player stays for full contract duration |
| Agent Fee | Percentage of transfer or flat fee |

---

## 8. Squad Management

### Squad Size Limits
- **Premier League**: 25 registered players (max 17 non-homegrown, min 8 homegrown slots)
- **Homegrown**: trained at a club in the same national association for 3+ years before age 21
- **Under-21**: unlimited registration (players under 21 do not need to be registered)

### Player Positions
- GK (Goalkeeper)
- CB (Centre-Back), LB (Left-Back), RB (Right-Back), LWB (Left Wing-Back), RWB (Right Wing-Back)
- CDM (Central Defensive Midfielder), CM (Central Midfielder), CAM (Central Attacking Midfielder)
- LM (Left Midfielder), RM (Right Midfielder), LW (Left Winger), RW (Right Winger)
- CF (Centre Forward), ST (Striker)

### Player Conditions (Football Manager)
| Condition | Match Readiness |
|-----------|----------------|
| Superb | 95-100% fitness, peak form |
| Good | 85-95% fitness |
| Adequate | 70-85% fitness |
| Poor | Below 70% fitness |
| Injured | Cannot play |
| Suspended | Serving ban |

### Morale Levels (Football Manager)
Superb → Very Good → Good → Okay → Poor → Very Poor → Awful

---
---

# DOMAIN 4: Stock Trading Desk -- Complete Mechanics Reference

---

## 1. Overview

A trading desk is the operational center where traders execute buy and sell orders for financial instruments. This reference covers order types, position management, margin, P&L calculation, risk metrics, common instruments, daily patterns, and options Greeks.

---

## 2. Order Types

### Basic Order Types

| Order Type | Description | Execution | Risk |
|------------|-------------|-----------|------|
| **Market Order** | Buy/sell at best available current price | Immediate (near-guaranteed fill) | Price uncertainty (slippage) |
| **Limit Order** | Buy at or below / sell at or above a specified price | Only at limit price or better; may never fill | No fill risk if price moves away |
| **Stop Order (Stop-Loss)** | Triggers a market order when price reaches stop price | Becomes market order at trigger; guaranteed trigger, not guaranteed price | Gap/slippage risk |
| **Stop-Limit Order** | Triggers a limit order when price reaches stop price | Only fills at limit price or better after trigger | May not fill if price gaps through |
| **Trailing Stop** | Stop price adjusts ("trails") as price moves favorably | Dynamic; e.g., trail by $2 or 5% from peak | Same gap risk as stop orders |

### Time-in-Force Instructions

| Instruction | Description |
|-------------|-------------|
| **DAY** | Expires at end of trading day if unfilled |
| **GTC (Good 'til Canceled)** | Remains active until filled or manually canceled (brokers may impose 60-90 day limits) |
| **IOC (Immediate or Cancel)** | Fill whatever can be filled immediately; cancel remainder |
| **FOK (Fill or Kill)** | Fill entire order immediately or cancel entire order |
| **GTD (Good 'til Date)** | Active until a specified expiration date |
| **MOO (Market on Open)** | Execute at market open price |
| **MOC (Market on Close)** | Execute at market close price |
| **LOO (Limit on Open)** | Limit order for opening auction only |
| **LOC (Limit on Close)** | Limit order for closing auction only |

### Advanced Order Types

| Order Type | Description |
|------------|-------------|
| **AON (All or None)** | Entire order must fill or nothing; no partial fills |
| **Iceberg / Reserve** | Only shows a small "display quantity" in the order book; hidden portion refills as displayed portion fills |
| **Bracket Order** | Simultaneously places take-profit (limit) and stop-loss around a position |
| **OCO (One Cancels Other)** | Two linked orders; when one executes, the other is automatically canceled |
| **TWAP (Time-Weighted Average Price)** | Algorithmic; splits large order into equal slices over time to minimize market impact |
| **VWAP (Volume-Weighted Average Price)** | Algorithmic; executes proportionally to historical volume pattern throughout the day |
| **Pegged Order** | Price tracks the bid, ask, or midpoint dynamically |
| **Conditional / Contingent** | Only activates when a specified condition is met (e.g., "if AAPL hits $200, buy MSFT") |

---

## 3. Position Management

### Long vs Short

| Position | Entry | Profit When | Max Loss | Max Gain |
|----------|-------|-------------|----------|----------|
| **Long** | Buy asset | Price rises | Purchase price (asset goes to $0) | Unlimited |
| **Short** | Sell borrowed asset | Price falls | Unlimited (price can rise indefinitely) | Purchase price (asset goes to $0) |

### Position Sizing
- **Shares/Contracts**: Quantity of the instrument held
- **Notional Value**: Price x Quantity (total dollar exposure)
- **Cost Basis**: Average price paid per share (including commissions)
- **Unrealized P&L**: (Current Price - Cost Basis) x Quantity
- **Realized P&L**: Profit/Loss from closed positions

### P&L Calculation

**Long Position**:
```
P&L = (Exit Price - Entry Price) x Quantity - Commissions - Fees
```

**Short Position**:
```
P&L = (Entry Price - Exit Price) x Quantity - Commissions - Fees - Borrow Cost
```

**Return on Investment (ROI)**:
```
ROI = (P&L / Initial Investment) x 100%
```

**For Leveraged/Margin Positions**:
```
ROI = (P&L / Margin Deposited) x 100%
```
(Leverage amplifies both gains and losses)

---

## 4. Margin Requirements

### Regulation T (Federal Reserve, US)

| Position Type | Initial Margin | Maintenance Margin (FINRA minimum) | Typical Broker Maintenance |
|---------------|---------------|-------------------------------------|---------------------------|
| **Long Equity** | 50% of purchase price | 25% of current market value | 30-40% |
| **Short Equity** | 150% of short sale value (100% proceeds + 50% margin) | 30% of current market value | 30-40% |
| **Day Trading (Pattern Day Trader)** | $25,000 minimum equity | 25% of highest open position value | Varies |

### Margin Call Process
1. Account equity falls below maintenance margin requirement
2. Broker issues **margin call**
3. Trader must deposit additional cash/securities within **2-5 business days** (broker dependent)
4. If not met, broker can **liquidate positions** at their discretion without notice

### Buying Power
```
Buying Power (Reg T) = Excess Equity / Initial Margin Rate
Example: $100,000 account, $60,000 in positions, $40,000 excess equity
Buying Power = $40,000 / 0.50 = $80,000 additional purchasing capacity
```

### Portfolio Margin (for qualified accounts, FINRA)
- Uses risk-based model (theoretical scenarios) rather than fixed percentages
- Typically allows **6:1 leverage** or more (vs 2:1 for Reg T)
- Minimum account equity: **$100,000+**
- Stress-tested against +/- 15% market moves (or more for concentrated positions)

---

## 5. Risk Metrics

### Value at Risk (VaR)
- **Definition**: Maximum expected loss over a specified time period at a given confidence level under normal market conditions
- **Typical Parameters**: 1-day 95% VaR or 1-day 99% VaR

**Calculation Methods**:

| Method | Description |
|--------|-------------|
| **Historical Simulation** | Use actual past returns; sort from worst to best; VaR = return at the (1-confidence)th percentile |
| **Parametric (Variance-Covariance)** | Assumes normal distribution; VaR = Portfolio Value x Z-score x σ x sqrt(time) |
| **Monte Carlo Simulation** | Generate thousands of random scenarios from assumed distributions; VaR = percentile of simulated P&L |

**Z-scores**: 95% = 1.645; 99% = 2.326; 99.9% = 3.090

**Example**:
```
Portfolio Value: $10,000,000
Daily σ (volatility): 1.5%
1-day 95% VaR = $10,000,000 x 1.645 x 0.015 = $246,750
Interpretation: 95% confidence that daily loss will not exceed $246,750
```

### Conditional VaR (CVaR / Expected Shortfall)
- **Definition**: Average loss in the worst (1-confidence)% of scenarios
- Always >= VaR; captures tail risk
- Example: If 99% VaR = $500K, CVaR might = $750K (average of the worst 1% of outcomes)

### Sharpe Ratio
```
Sharpe Ratio = (Rp - Rf) / σp
```
Where:
- Rp = portfolio return (annualized)
- Rf = risk-free rate (e.g., T-bill rate, ~4-5% in 2024-25)
- σp = standard deviation of portfolio returns (annualized)

| Sharpe Ratio | Interpretation |
|-------------|----------------|
| < 0 | Worse than risk-free; losing money on risk-adjusted basis |
| 0 - 0.5 | Below average |
| 0.5 - 1.0 | Acceptable |
| 1.0 - 2.0 | Good |
| 2.0 - 3.0 | Very good |
| > 3.0 | Excellent (rare, often signals overfitting or low-frequency strategies) |

### Sortino Ratio
```
Sortino Ratio = (Rp - Rf) / σ_downside
```
- Same as Sharpe but only penalizes downside volatility (below target return)
- Better for asymmetric return distributions

### Maximum Drawdown (MDD)
```
MDD = (Trough Value - Peak Value) / Peak Value x 100%
```
- **Definition**: Largest peak-to-trough decline before a new peak is achieved
- Measures worst-case loss experience over a period
- **Calmar Ratio** = Annualized Return / Maximum Drawdown

**Example**:
```
Portfolio peak: $1,000,000
Portfolio trough: $750,000
MDD = ($750,000 - $1,000,000) / $1,000,000 = -25%
```

### Beta
```
β = Covariance(Rp, Rm) / Variance(Rm)
```
- Measures portfolio sensitivity to market movements
- β = 1.0: moves with market; β = 1.5: 50% more volatile than market; β = 0.5: half as volatile

### Treynor Ratio
```
Treynor Ratio = (Rp - Rf) / β
```
- Return per unit of systematic (market) risk

### Information Ratio
```
IR = (Rp - Rb) / Tracking Error
```
- Measures manager skill relative to a benchmark (Rb)

---

## 6. Common Instruments

### Equities (Stocks)
| Feature | Details |
|---------|---------|
| What it is | Ownership shares in a company |
| Trading hours | NYSE/NASDAQ: 9:30 AM - 4:00 PM ET |
| Pre-market | 4:00 AM - 9:30 AM ET |
| After-hours | 4:00 PM - 8:00 PM ET |
| Settlement | T+1 (trade date + 1 business day, since May 2024) |
| Tick size | $0.01 for stocks over $1 |

### Bonds (Fixed Income)
| Feature | Details |
|---------|---------|
| What it is | Debt instrument; lender receives interest (coupon) + principal |
| Types | Government (Treasuries), Corporate, Municipal, High-Yield |
| Trading hours | OTC; ~8:00 AM - 5:00 PM ET |
| Yield calculation | Yield = (Coupon / Price) x 100; Yield to Maturity (YTM) accounts for price premium/discount |
| Duration | Measure of price sensitivity to interest rate changes |

### Exchange-Traded Funds (ETFs)
| Feature | Details |
|---------|---------|
| What it is | Basket of securities (stocks, bonds, commodities) traded like a stock |
| Examples | SPY (S&P 500), QQQ (NASDAQ-100), IWM (Russell 2000), TLT (Long-term Treasuries) |
| Trade just like stocks | Same hours, same order types |
| Expense ratio | Annual fee (e.g., 0.03% for SPY) |

### Options
| Feature | Details |
|---------|---------|
| What it is | Contract giving right (not obligation) to buy/sell underlying at a strike price by expiration |
| Types | Call (right to buy), Put (right to sell) |
| Contract size | 100 shares per contract (standard equity options) |
| Expiration | Weekly, monthly, quarterly, LEAPS (1-2+ years) |
| Settlement | American-style (exercise anytime); European-style (exercise only at expiration) |
| Trading hours | 9:30 AM - 4:00 PM ET (some index options until 4:15 PM) |

### Futures
| Feature | Details |
|---------|---------|
| What it is | Obligation to buy/sell an asset at a set price on a future date |
| Types | Equity index (ES, NQ), Commodity (CL, GC, ZC), Bond (ZB, ZN), Currency (6E, 6J) |
| Trading hours | Nearly 24 hours, Sun 6:00 PM - Fri 5:00 PM ET (with 1-hour daily break) |
| Margin | Much lower than equities; e.g., ES futures: ~$12,000 initial margin for ~$250K notional |
| Settlement | Cash-settled or physically delivered |

### Foreign Exchange (Forex)
| Feature | Details |
|---------|---------|
| What it is | Trading currency pairs (e.g., EUR/USD, GBP/JPY) |
| Trading hours | 24 hours, Sun 5:00 PM - Fri 5:00 PM ET |
| Major pairs | EUR/USD, USD/JPY, GBP/USD, USD/CHF, AUD/USD, USD/CAD |
| Lot sizes | Standard: 100,000 units; Mini: 10,000; Micro: 1,000 |
| Leverage | Up to 50:1 (US), 30:1 (EU), 500:1 (some offshore) |

### Commodities
| Category | Examples |
|----------|---------|
| Energy | Crude Oil (WTI, Brent), Natural Gas, Gasoline |
| Precious Metals | Gold, Silver, Platinum, Palladium |
| Base Metals | Copper, Aluminum, Zinc, Nickel |
| Agriculture | Corn, Wheat, Soybeans, Coffee, Cocoa, Sugar, Cotton |
| Livestock | Live Cattle, Lean Hogs |

---

## 7. Typical Daily Trading Patterns

### US Equity Market Daily Timeline (Eastern Time)

| Time | Event | Characteristics |
|------|-------|-----------------|
| 4:00 AM | Pre-market opens | Very thin liquidity; wide spreads; earnings reactions |
| 7:00-8:00 AM | Economic data releases | Key reports (jobs, CPI, GDP) cause volatility spikes |
| 8:30 AM | Major economic releases | Non-Farm Payrolls (1st Friday), CPI, GDP |
| 9:30 AM | **Opening Bell** | Highest volatility of the day; large volume burst; gap fills |
| 9:30-10:30 AM | "Opening Range" | First hour sets daily high/low ~35% of the time; most volatile |
| 10:00 AM | Some economic data | Consumer confidence, home sales, etc. |
| 10:30-11:30 AM | Mid-morning fade | Volatility decreases; many opening moves reverse |
| 11:30 AM - 1:00 PM | "Lunch hour" | Lowest volume/volatility of the day; choppy; avoid trading |
| 1:00 PM | Bond market activity | Treasury auctions (1:00 PM); Fed announcements (2:00 PM FOMC days) |
| 2:00 PM | FOMC decisions (8x/year) | Major volatility events when they occur |
| 3:00-3:30 PM | Pre-close positioning | Volume picks up; institutional rebalancing begins |
| 3:30-4:00 PM | **Power Hour** | Second-highest volume; MOC orders execute; trend continuation or reversal |
| 4:00 PM | **Closing Bell** | Final prices set; closing auction |
| 4:00-4:15 PM | Index options close | SPX, VIX options settlement |
| 4:00-8:00 PM | After-hours trading | Thin liquidity; earnings reactions; wide spreads |

### Intraday Volume Profile (U-Shape)
- **Highest volume**: First 30 min (~15-20% of daily volume) and last 30 min (~15-20%)
- **Lowest volume**: 12:00-1:00 PM (~5-8% of daily volume)
- Shape resembles a "U" or "smile" when charted

---

## 8. Options Greeks

### Overview
The Greeks measure how an option's price responds to changes in underlying factors. Derived from the Black-Scholes-Merton model for European options.

### Delta (Δ)
| Property | Value |
|----------|-------|
| **Definition** | Rate of change of option price per $1 move in underlying |
| **Call range** | 0 to +1.0 |
| **Put range** | -1.0 to 0 |
| **ATM call** | ~+0.50 |
| **ATM put** | ~-0.50 |
| **Deep ITM** | Approaches ±1.0 |
| **Deep OTM** | Approaches 0 |
| **Probability proxy** | Delta ≈ probability option expires ITM |

**Example**: Delta = 0.40 means option price increases ~$0.40 for each $1.00 increase in the underlying.

**Position Delta** (portfolio level): Sum of all position deltas. Delta-neutral = position delta near 0.

### Gamma (Γ)
| Property | Value |
|----------|-------|
| **Definition** | Rate of change of Delta per $1 move in underlying |
| **Always positive** for long options (calls and puts) |
| **Highest** | ATM options near expiration |
| **Lowest** | Deep ITM or OTM options |
| **Formula identical** for calls and puts |

**Example**: If Delta = 0.40 and Gamma = 0.05, then after a $1 increase in underlying, new Delta = 0.45.

**Gamma Risk**: Near expiration, ATM options have enormous gamma, causing delta to swing wildly. Called "gamma exposure" or "GEX" at market level.

### Theta (Θ)
| Property | Value |
|----------|-------|
| **Definition** | Rate of option price decay per day (time decay) |
| **Always negative** for long options (lose value as time passes) |
| **Acceleration** | Theta accelerates as expiration approaches (non-linear) |
| **ATM options** | Highest theta (most time value to lose) |
| **30 DTE** | Moderate theta |
| **7 DTE** | Rapid theta decay |
| **1 DTE** | Extreme theta for ATM |

**Example**: Theta = -0.05 means the option loses $0.05 per day (per share; $5.00 per contract).

**Rule of Thumb**: An ATM option loses roughly 1/3 of its time value in the last week before expiration.

### Vega (ν)
| Property | Value |
|----------|-------|
| **Definition** | Rate of change of option price per 1% change in implied volatility (IV) |
| **Always positive** for long options (higher IV = higher option price) |
| **Highest** | ATM options with long time to expiration |
| **Same formula** for calls and puts |
| **Decreases** as expiration approaches |

**Example**: Vega = 0.12 means if IV increases by 1% (e.g., 20% to 21%), option price increases by $0.12 per share.

**Implied Volatility (IV)**: Market's expectation of future price movement. IV Rank and IV Percentile compare current IV to historical range.

### Rho (ρ)
| Property | Value |
|----------|-------|
| **Definition** | Rate of change of option price per 1% change in risk-free interest rate |
| **Calls** | Positive rho (higher rates → higher call prices) |
| **Puts** | Negative rho (higher rates → lower put prices) |
| **Significance** | Generally least impactful Greek; matters more for LEAPS (long-dated options) |

**Example**: Rho = 0.03 means if interest rates rise by 1%, option price increases by $0.03 per share.

### Greeks Summary Table

| Greek | Measures | Long Call | Long Put | Short Call | Short Put |
|-------|----------|-----------|----------|------------|-----------|
| **Delta** | Price sensitivity | + (0 to 1) | - (-1 to 0) | - (0 to -1) | + (0 to 1) |
| **Gamma** | Delta sensitivity | + | + | - | - |
| **Theta** | Time decay | - | - | + | + |
| **Vega** | Volatility sensitivity | + | + | - | - |
| **Rho** | Interest rate sensitivity | + | - | - | + |

### Black-Scholes Formula Reference
```
Call Price: C = S*N(d1) - K*e^(-rT)*N(d2)
Put Price:  P = K*e^(-rT)*N(-d2) - S*N(-d1)

Where:
  d1 = [ln(S/K) + (r + σ²/2)*T] / (σ*sqrt(T))
  d2 = d1 - σ*sqrt(T)

  S = current stock price
  K = strike price
  r = risk-free rate
  T = time to expiration (years)
  σ = implied volatility
  N() = cumulative standard normal distribution
```

---
---

# DOMAIN 5: Cooking Competition -- Complete Mechanics Reference

---

## 1. Overview

Cooking competitions pit chefs against each other under time pressure with specific constraints (mystery ingredients, theme requirements). This reference covers the format of major cooking competition shows (primarily Chopped, MasterChef, and generic competition standards), judging systems, food safety temperatures, plating, and techniques.

---

## 2. Competition Formats

### Chopped (Food Network)
| Element | Details |
|---------|---------|
| Contestants | 4 chefs per episode |
| Rounds | 3: Appetizer, Entree, Dessert |
| Elimination | 1 chef "chopped" (eliminated) per round |
| Final | Last 2 chefs compete in Dessert round for $10,000 |
| Timing | Appetizer: **20 minutes**, Entree: **30 minutes**, Dessert: **30 minutes** |
| Mystery Basket | 4 mandatory ingredients per round (must use ALL four) |
| Platings | 4 plates: 3 for judges + 1 "beauty plate" for display |
| Judging Criteria | Taste, Presentation, Creativity, Use of basket ingredients |

### MasterChef (Fox)
| Element | Details |
|---------|---------|
| Contestants | Varies (typically 18-24 home cooks per season) |
| Round Types | Mystery Box, Team Challenge, Pressure Test, Elimination |
| Mystery Box | Identical ingredients for all; **60 minutes** to cook |
| Team Challenge | Groups cook for large groups of diners; **2+ hours** |
| Pressure Test | Replicate a complex dish; **60-90 minutes** |
| Elimination | Bottom performers face Pressure Test; worst performer eliminated |
| Judges | 3 (e.g., Gordon Ramsay, Joe Bastianich, Graham Elliot) |
| Scoring | Subjective; no formal numeric scores displayed |

### Iron Chef / Iron Chef America
| Element | Details |
|---------|---------|
| Contestants | 1 challenger vs 1 Iron Chef |
| Time Limit | **60 minutes** |
| Secret Ingredient | Revealed at start; must be featured in every dish |
| Dishes | Multiple courses (typically 4-5 dishes each) |
| Judges | Panel of 3-5 celebrity/food expert judges |
| Scoring | Taste (out of 10), Plating (out of 10), Originality (out of 10) per dish |

### Generic Competition Standard (ACF / World Chefs)
| Element | Details |
|---------|---------|
| Setup Time | 15 minutes |
| Cooking Time | 60 minutes |
| Plating Time | 10 minutes |
| Cleanup Time | 15 minutes |
| Portions | 3 for tasting + 1 for display |
| Judging Categories | Taste (50%), Execution/Technique (35%), Presentation/Appearance (15%) |

---

## 3. Judging Criteria -- Detailed Breakdown

### Taste (Weighted Highest: 40-50%)

| Criterion | Score Range | What Judges Evaluate |
|-----------|------------|---------------------|
| Flavor Balance | 1-10 | Sweet, salty, sour, bitter, umami in harmony |
| Seasoning | 1-10 | Proper salt levels; seasoned throughout (not just surface) |
| Depth of Flavor | 1-10 | Layers of flavor; complexity vs one-note |
| Doneness of Protein | 1-10 | Properly cooked (not over/undercooked) |
| Sauce/Accompaniment | 1-10 | Complements the dish; proper consistency |
| Temperature | 1-10 | Hot food hot, cold food cold; not lukewarm |

### Presentation / Plating (15-25%)

| Criterion | Score Range | What Judges Evaluate |
|-----------|------------|---------------------|
| Visual Appeal | 1-10 | Overall beauty; appetizing appearance |
| Color Contrast | 1-10 | Varied colors; not monochromatic |
| Plate Cleanliness | 1-10 | No drips, smears, fingerprints on rim |
| Height/Dimension | 1-10 | Varied vertical elements; not flat |
| Portion Size | 1-10 | Appropriate for the course; not too much/little |
| Garnish | 1-10 | Functional (edible, adds flavor); not purely decorative |

### Creativity / Originality (15-25%)

| Criterion | Score Range | What Judges Evaluate |
|-----------|------------|---------------------|
| Ingredient Use | 1-10 | Inventive use of mystery/required ingredients |
| Technique Range | 1-10 | Multiple techniques demonstrated (3+ preferred) |
| Concept | 1-10 | Cohesive idea; tells a story on the plate |
| Risk-Taking | 1-10 | Ambitious but well-executed |
| Surprise Element | 1-10 | Unexpected flavor combination that works |

### Execution / Technique (20-35%)

| Criterion | Score Range | What Judges Evaluate |
|-----------|------------|---------------------|
| Knife Skills | 1-10 | Uniform cuts, brunoise/julienne/chiffonade precision |
| Cooking Methods | 1-10 | Proper application of techniques |
| Timing | 1-10 | All components finished simultaneously |
| Consistency | 1-10 | Each plate identical (across multiple servings) |
| Waste Minimization | 1-10 | Use of trim, bones, scraps |

---

## 4. Protein Internal Temperatures (USDA Safe Minimums -- Fahrenheit)

### Poultry

| Protein | Safe Minimum Temp | Notes |
|---------|------------------|-------|
| Chicken (whole) | **165°F** (74°C) | Measured at thickest part of thigh |
| Chicken (breast) | **165°F** (74°C) | Often pulled at 160°F and rested (carryover) |
| Chicken (thigh/leg) | **165°F** (74°C) | Better at 175-180°F for collagen breakdown |
| Turkey (whole) | **165°F** (74°C) | Breast and thigh; stuffing must also reach 165°F |
| Duck (breast) | **165°F** USDA; chefs serve at **135-145°F** (medium-rare to medium) | Duck breast often treated like steak in competition |
| Ground Poultry | **165°F** (74°C) | No exceptions |

### Beef

| Doneness | Internal Temp (°F) | Internal Temp (°C) | Appearance |
|----------|-------------------|-------------------|------------|
| Blue Rare | 115-120°F | 46-49°C | Cool red center; seared outside only |
| Rare | 120-130°F | 49-54°C | Cool-to-warm red center |
| Medium-Rare | 130-135°F | 54-57°C | Warm red center (competition ideal for steaks) |
| Medium | 135-145°F | 57-63°C | Warm pink center |
| Medium-Well | 145-155°F | 63-68°C | Slightly pink center |
| Well-Done | 155-165°F+ | 68-74°C+ | No pink; fully cooked throughout |
| USDA Safe Minimum | 145°F + 3 min rest | 63°C | For steaks, chops, roasts |
| Ground Beef | 160°F | 71°C | No exceptions; no resting credit |

### Pork

| Protein | Temp | Notes |
|---------|------|-------|
| Pork chops/tenderloin | **145°F** (63°C) + 3 min rest | USDA updated from 160°F; slightly pink is safe |
| Pork shoulder/butt (BBQ) | **195-205°F** (90-96°C) | Low and slow; collagen fully rendered |
| Ground Pork | **160°F** (71°C) | No resting credit |
| Pork ribs | **195-203°F** (90-95°C) | Fall-off-the-bone tender |
| Ham (fresh) | **145°F** + 3 min rest | Pre-cooked ham: 140°F to reheat |

### Lamb

| Protein | Temp | Notes |
|---------|------|-------|
| Rack of lamb (medium-rare) | **130-135°F** (54-57°C) | Competition standard |
| Rack of lamb (medium) | **135-145°F** (57-63°C) | Still pink |
| Lamb chops | **145°F** USDA | Chefs often serve at 130-140°F |
| Leg of lamb | **145°F** USDA minimum | |
| Ground Lamb | **160°F** (71°C) | |
| Lamb shank (braised) | **190-200°F** (88-93°C) | Collagen breakdown |

### Fish and Seafood

| Protein | Temp | Notes |
|---------|------|-------|
| Salmon (medium) | **125-130°F** (52-54°C) | Translucent pink center; competition standard |
| Salmon (USDA safe) | **145°F** (63°C) | Fully opaque |
| Tuna (seared rare) | **110-115°F** (43-46°C) | Red center; seared outside |
| White fish (cod, halibut) | **140-145°F** (60-63°C) | Opaque, flakes easily |
| Shrimp | **120°F** (49°C) | Turns pink and opaque; VERY easy to overcook |
| Lobster | **140°F** (60°C) | Opaque white; rubbery if overcooked |
| Scallops | **120-130°F** (49-54°C) | Translucent center; golden sear outside |
| Shellfish (mussels, clams) | N/A | Cook until shells open; discard unopened |

### Eggs

| Preparation | Temp | Notes |
|-------------|------|-------|
| Eggs (general USDA) | **160°F** (71°C) | Firm yolk and white |
| Soft-boiled | ~145-150°F yolk | Runny yolk, set white |
| Hard-boiled | ~170°F throughout | Fully set yolk |
| Custard/sauce | **160°F** | Thickens properly; kills bacteria |
| Sous vide egg (63°C egg) | **145.4°F** (63°C) for 60-75 min | Silky custard-like texture |

### Carryover Cooking
- **Rule**: Remove protein **3-5°F below target** temperature
- Carryover is caused by residual heat from the exterior continuing to cook the interior
- Larger/thicker cuts have more carryover (up to 10°F for large roasts)
- Resting time: **5-10 minutes** for steaks/chops; **15-30 minutes** for roasts

---

## 5. Cooking Techniques -- Complete Reference

### Dry-Heat Methods

| Technique | Description | Temperature Range | Best For |
|-----------|-------------|------------------|----------|
| **Sauteing** | Quick cooking in thin layer of fat over high heat; toss ingredients | 350-450°F pan surface | Vegetables, thin proteins, stir-fry |
| **Pan-Searing** | High-heat browning in oil; minimal movement; develop crust | 400-500°F pan surface | Steaks, fish fillets, scallops, duck breast |
| **Roasting** | Dry heat in oven; uncovered; convective heat all around | 300-450°F oven | Whole chickens, vegetables, prime rib |
| **Broiling** | High radiant heat from above (oven broiler) | 500°F+ | Gratins, finishing dishes, melting cheese |
| **Grilling** | Radiant heat from below (charcoal or gas) | 400-700°F | Steaks, burgers, vegetables, kebabs |
| **Baking** | Dry heat in oven; typically for doughs/batters | 325-425°F oven | Bread, pastries, casseroles |
| **Deep-Frying** | Full submersion in hot oil | 325-375°F oil | French fries, fried chicken, tempura, doughnuts |
| **Stir-Frying** | Very high heat wok cooking with constant motion | 500°F+ | Asian-style vegetables and proteins |

### Moist-Heat Methods

| Technique | Description | Temperature Range | Best For |
|-----------|-------------|------------------|----------|
| **Boiling** | Cooking in liquid at 212°F (100°C); vigorous bubbles | 212°F | Pasta, potatoes, blanching |
| **Simmering** | Gentle bubbling just below boiling point | 185-205°F | Stocks, soups, sauces, stews |
| **Poaching** | Submerged in liquid below simmering; very gentle | 140-180°F | Eggs, fish, chicken breast, fruit |
| **Steaming** | Cooking over (not in) boiling water using steam | 212°F | Vegetables, fish, dumplings, custards |
| **Blanching** | Brief immersion in boiling water, then ice bath | 212°F (30s-3min) | Vegetables (set color, remove bitterness), peeling tomatoes |

### Combination Methods

| Technique | Description | Temperature/Duration | Best For |
|-----------|-------------|---------------------|----------|
| **Braising** | Sear at high heat, then cook covered in small amount of liquid at low heat | Sear at 400°F+; braise at 275-325°F for 2-4 hours | Short ribs, osso buco, lamb shanks, pork shoulder |
| **Stewing** | Similar to braising but protein is cut into smaller pieces, fully submerged | 300-325°F for 1.5-3 hours | Beef stew, chili, curry, coq au vin |

### Advanced / Specialty Techniques

| Technique | Description | Key Details |
|-----------|-------------|-------------|
| **Sous Vide** | Vacuum-sealed food cooked in precisely controlled water bath | Temperature accuracy to ±0.1°F; cook times 1-72 hours; perfect edge-to-edge doneness |
| **Deglazing** | Adding liquid (wine, stock, vinegar) to hot pan to dissolve fond (browned bits) | Do immediately after searing; scrape bottom; forms base for pan sauce |
| **Flambe** | Igniting alcohol (typically 80-proof spirits) in pan for theatrical effect + flavor | Use cognac, rum, brandy (40% ABV ideal); tilt pan toward flame or use lighter; burns off in 10-30s |
| **Confit** | Cooking and preserving in fat at low temperature | 200-275°F for 2-10 hours; duck legs, garlic, tomatoes |
| **Curing** | Preserving with salt, sugar, and/or nitrates | Gravlax (salmon): 24-48h; duck prosciutto: 1-4 weeks |
| **Smoking** | Exposing food to wood smoke for flavor and preservation | Cold smoke: 68-86°F; hot smoke: 200-300°F |
| **Emulsification** | Combining two immiscible liquids (oil + water/acid) into stable mixture | Vinaigrettes, mayonnaise, hollandaise; lecithin or egg yolk as emulsifier |
| **Reduction** | Simmering liquid uncovered to evaporate water and concentrate flavor | Reduce by half = "demi"; reduce to syrup = "glace" |
| **Tempering** | Gradually heating a cold ingredient with a hot one to prevent curdling/seizure | Egg yolks into hot cream; chocolate to specific temperature curves |
| **Spherification** | Using sodium alginate and calcium to create gel spheres | Molecular gastronomy; "caviar" pearls, ravioli effect |
| **Brining** | Soaking protein in salt-water solution for moisture and seasoning | Wet brine: 3-6% salt solution, 4-24 hours; dry brine: salt directly on surface, 1-48 hours |
| **Caramelization** | Browning sugar by heating above 320°F | Dry caramel: sugar only; wet caramel: sugar + water; stages from light to dark |
| **Maillard Reaction** | Browning reaction between amino acids and sugars at 280-330°F | Not caramelization; responsible for crust on seared meats, toasted bread |

---

## 6. Plating Principles

### Composition Rules
- **Rule of Odds**: Odd numbers of elements (3, 5) are more visually appealing than even
- **White Space**: Don't overcrowd the plate; use negative space
- **Clock Method**: Protein at 6 o'clock (nearest diner), starch at 10, vegetable at 2
- **Height**: Build vertically; creates visual interest and dimension
- **Focal Point**: One dominant element draws the eye; everything else supports it

### Color Guidelines
- **Minimum 3 colors** on the plate (green, brown/sear, white/starch is baseline)
- **Contrasting colors** against plate color (dark food on white plate; colorful food on dark/slate plate)
- Avoid monochromatic plates (all brown = unappetizing)
- Sauces as art: swoosh, dot pattern, mirror (puddle underneath)

### Sauce Application Methods
| Method | Description |
|--------|-------------|
| **Swoosh/Smear** | Drag spoon or offset spatula across plate |
| **Dots** | Squeeze bottle; precise dots in pattern |
| **Pool/Mirror** | Thin layer covering base of plate |
| **Drizzle** | Thin stream from spoon or squeeze bottle |
| **Quenelle** | Elegant three-sided oval of sauce/puree (formed between two spoons) |
| **Ring mold** | Sauce or puree inside ring for clean circle |

### Common Plating Mistakes (Judge Deductions)
- Dirty plate rim (fingerprints, splashes)
- Overcrowded plate
- Monotone colors
- Random or sloppy arrangement
- Non-functional garnish (inedible herbs still on stem)
- Improper portion for the course (competition appetizer should be 3-4 bites)
- Food touching plate rim

---

## 7. Competition Kitchen Equipment

### Standard Equipment Available
- Gas range (6-8 burners)
- Convection oven
- Deep fryer
- Blender (standard + immersion)
- Stand mixer (KitchenAid)
- Food processor
- Mandoline slicer
- Microplane
- Ring molds (various sizes)
- Squeeze bottles
- Sheet pans, saucepans, saute pans, cast iron
- Chinois (fine mesh strainer)
- Piping bags and tips
- Torch (brulee)
- Ice cream machine (sometimes)
- Sous vide circulator (sometimes)
- Pressure cooker

### Standard Pantry
- Proteins (chicken, beef, pork, fish, shrimp -- varies)
- Eggs, butter, cream, milk
- All-purpose flour, sugar, brown sugar, powdered sugar
- Salt (kosher, flaky), black pepper
- Olive oil, vegetable oil, sesame oil
- Vinegars (red wine, white wine, balsamic, rice, sherry)
- Soy sauce, fish sauce, Worcestershire
- Fresh herbs (parsley, cilantro, thyme, rosemary, basil, chives, mint, dill)
- Dried spices (cumin, coriander, paprika, cayenne, cinnamon, nutmeg, etc.)
- Onions, garlic, shallots, ginger
- Lemons, limes, oranges
- Stock (chicken, beef, vegetable)
- Canned tomatoes, tomato paste
- Mustard (Dijon, whole grain)
- Honey, maple syrup

---

## 8. Timing Strategy for Competition

### 30-Minute Round (Appetizer)
| Time Block | Minutes | Activity |
|------------|---------|----------|
| 0:00-2:00 | 2 | Assess basket; plan dish; gather ingredients |
| 2:00-5:00 | 3 | Prep (mise en place): chop, slice, measure |
| 5:00-10:00 | 5 | Start protein; begin sauce base |
| 10:00-20:00 | 10 | Main cooking; multiple components simultaneously |
| 20:00-25:00 | 5 | Taste, adjust seasoning; cook garnish components |
| 25:00-28:00 | 3 | Plate all 4 plates |
| 28:00-30:00 | 2 | Final garnish; clean plate rims; last-second adjustments |

### 45-Minute Round (Entree)
| Time Block | Minutes | Activity |
|------------|---------|----------|
| 0:00-3:00 | 3 | Assess, plan, gather |
| 3:00-8:00 | 5 | Full mise en place |
| 8:00-15:00 | 7 | Start longest-cooking items (braise, reduction, roast) |
| 15:00-30:00 | 15 | Main cooking phase; manage multiple components |
| 30:00-38:00 | 8 | Finish sauces; sear proteins; cook final elements |
| 38:00-43:00 | 5 | Plate |
| 43:00-45:00 | 2 | Garnish; clean rims |

### Critical Time Management Rules
1. **Longest-cooking item goes on first** (reduce, braise, roast, confits)
2. **Taste constantly** (minimum 5 times during cooking)
3. **Season in layers** (salt at each stage, not just at the end)
4. **Start plating 5 minutes before time** (never less)
5. **Have a backup plan** if primary technique fails
6. **Do not attempt techniques you have not practiced** (risk of failure too high)

---

## 9. Common Competition Pitfalls

| Mistake | Consequence | Frequency |
|---------|-------------|-----------|
| **Underseasoned** | #1 judge complaint; flat, dull flavors | Very Common |
| **Overcooked protein** | Dry, tough; shows lack of fundamental skill | Common |
| **Raw/undercooked** | Food safety issue; automatic penalty | Occasional |
| **Not using all mystery ingredients** | Automatic penalty or elimination (Chopped) | Occasional |
| **Incomplete plate** | Missing component; looks unfinished | Common under time pressure |
| **Dirty plate** | Sloppy appearance; unprofessional | Common |
| **Over-ambitious menu** | Too many components; none executed well | Common |
| **Soggy/broken textures** | Soggy bun, broken emulsion, wilted greens | Common |
| **No acid** | Dish feels heavy/flat without brightness | Very Common |
| **Forgot to rest protein** | Juices run out when sliced; dry meat | Common |
