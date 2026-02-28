# League of Legends -- Domain Mechanics Reference

## Overview

League of Legends (LoL) is a 5v5 multiplayer online battle arena (MOBA) on Summoner's Rift. Two teams of five champions compete to destroy the opposing Nexus. Games last 25-40 minutes on average. As of November 2025, there are **172 released champions**.

---

## Champion Pool (~172 Champions)

Champions are categorized by class and role:

### Classes
- **Fighters (Bruisers)**: Darius, Fiora, Garen, Irelia, Jax, Mordekaiser, Renekton, Riven, Sett, Aatrox, Camille, Gwen, Urgot, Volibear, Wukong, Yone
- **Tanks**: Amumu, Cho'Gath, Malphite, Maokai, Nautilus, Ornn, Sejuani, Shen, Sion, Tahm Kench, Zac, Leona, Alistar, Braum, Rell, K'Sante
- **Mages**: Ahri, Annie, Azir, Brand, Cassiopeia, Hwei, LeBlanc, Lissandra, Lux, Malzahar, Orianna, Ryze, Syndra, Taliyah, Veigar, Viktor, Xerath, Ziggs, Zyra, Aurora
- **Assassins**: Akali, Ekko, Evelynn, Fizz, Kassadin, Katarina, Kha'Zix, Naafiri, Qiyana, Rengar, Shaco, Talon, Zed, Ambessa
- **Marksmen (ADC)**: Aphelios, Ashe, Caitlyn, Draven, Ezreal, Jhin, Jinx, Kai'Sa, Kog'Maw, Lucian, Miss Fortune, Samira, Sivir, Tristana, Twitch, Varus, Vayne, Xayah, Zeri, Smolder
- **Supports**: Bard, Janna, Karma, Lulu, Milio, Morgana, Nami, Pyke, Rakan, Renata Glasc, Senna, Seraphine, Sona, Soraka, Thresh, Yuumi, Zilean
- **Specialists**: Azir, Heimerdinger, Singed, Teemo, Fiddlesticks, Ivern

### Primary Roles on Summoner's Rift
| Role | Position | Typical Classes | Typical Stats at 20 min |
|------|----------|----------------|------------------------|
| Top | Top lane (solo) | Fighters, Tanks | 150-180 CS, 5000-7000 gold |
| Jungle | Jungle camps | Fighters, Tanks, Assassins | 120-150 CS (4 per camp), 5000-6500 gold |
| Mid | Mid lane (solo) | Mages, Assassins | 150-180 CS, 5500-7500 gold |
| ADC (Bot carry) | Bot lane (duo) | Marksmen | 160-190 CS, 6000-8000 gold |
| Support | Bot lane (duo) | Supports, Tanks | 20-40 CS, 3500-5000 gold |

---

## Game Phases

| Phase | Time Window | Key Characteristics |
|-------|------------|---------------------|
| Early Game (Laning) | 0:00 - 14:00 | Lane farming, turret plates, first dragon, Rift Herald |
| Mid Game | 14:00 - 25:00 | Tower trading, dragon soul race, Baron spawns at 20:00 |
| Late Game | 25:00+ | Baron plays, Elder Dragon, teamfights, death timers 40-60s |

### Key Timers
- Minions spawn: 1:05
- Jungle camps spawn: 1:30
- Rift Herald spawns: 14:00
- First dragon spawns: 5:00
- Baron Nashor spawns: 20:00
- Elder Dragon spawns: after one team gets Dragon Soul (or 35:00)
- Turret plates fall: 14:00
- Void Grubs spawn: 5:00 (removed when Rift Herald spawns)

---

## Gold Mechanics

### Passive Gold Generation
- **Starting gold**: 500
- **Passive gold rate**: 20.4 gold per 10 seconds (starting at 1:05)
- **Passive gold per minute**: 122.4

### CS (Creep Score) Gold -- Minion Values

| Minion Type | Base Gold | Scaling | Notes |
|------------|-----------|---------|-------|
| Melee minion | 21 gold | +0.125 gold/min after 15:00 | 3 per wave |
| Caster minion | 14 gold | +0.125 gold/min after 15:00 | 3 per wave |
| Siege/Cannon minion | 60 gold | +3 gold per upgrade tick (3:35 onwards), caps at 90 gold | Every 3 waves early, every 2 waves after 15:05, every wave after 25:05 |
| Super minion | 60-90 gold | Same as siege minion | Spawns when inhibitor destroyed |

### Wave Gold Values
- **Normal wave** (3 melee + 3 caster): ~105 gold base
- **Cannon wave** (3 melee + 3 caster + 1 cannon): ~165 gold base
- **Wave cycle** (3 waves, 2 normal + 1 cannon): ~375 gold
- **Perfect CS at 10:00**: ~107 CS, ~1200-1400 gold from minions alone

### Jungle Camp Gold

| Camp | Gold | Respawn Timer |
|------|------|---------------|
| Gromp | 80 | 2:15 |
| Blue Sentinel | 90 | 5:00 |
| Red Brambleback | 90 | 5:00 |
| Wolves (camp) | 95 | 2:15 |
| Raptors (camp) | 95 | 2:15 |
| Krugs (camp) | 130 | 2:15 |
| Rift Scuttler | 70 | 2:30 |

### Champion Kill Gold

#### Base Kill Gold
- **Base kill bounty**: 300 gold to the killer

#### Kill Streak Bounty (Shutdown System)

| Kill Streak | Bounty Tier | Total Kill Gold | Bounty Name |
|------------|-------------|-----------------|-------------|
| 0-1 kills | Tier 0-1 | 300 gold | -- |
| 2 kills | Tier 2 | 325 gold | -- |
| 3 kills | Tier 3 | 400 gold | Killing Spree |
| 4 kills | Tier 4 | 500 gold | Rampage |
| 5 kills | Tier 5 | 600 gold | Unstoppable |
| 6 kills | Tier 6 | 700 gold | Dominating |
| 7 kills | Tier 7 | 800 gold | Godlike |
| 8 kills | Tier 8 (cap) | 1000 gold | Legendary |
| 9+ kills | Extended | +100 gold per kill beyond 8 | Max 3600 gold at 33 kills |

#### Assist Gold
- If a kill is assisted, an extra **150 gold** (50% of base 300) is generated and split among all assisters
- Assist bounties: half of the kill bounty, capped at 150 gold
- Assists require dealing damage or applying CC to the target within 10 seconds of the kill

#### First Blood
- **First Blood bonus**: +100 gold to the killer (400 total for first kill)
- If assisted: +100 gold split among assisters

#### Death Streak (Behind Gold)
- Champions on a death streak are worth less gold:
  - 1 death: 274 gold
  - 2 deaths: 220 gold
  - 3 deaths: 176 gold
  - 4 deaths: 140 gold
  - 5+ deaths: 112 gold (minimum)

### Structure Gold

| Structure | Local Gold | Global Gold | Notes |
|-----------|-----------|-------------|-------|
| Outer turret | 250 (split) | 50 per player | First tower bonus: +300 local gold |
| Inner turret | 300 (split) | 50 per player | |
| Inhibitor turret | 50 (split) | 50 per player | |
| Inhibitor | 50 (split) | 0 | Respawns in 5:00 |
| Nexus turret | 50 (split) | 0 | |
| Turret plate (x5) | 125 (local) | 0 | Plates fall at 14:00; 5 plates = 625 potential gold |

### Objective Gold

| Objective | Gold | Notes |
|-----------|------|-------|
| Elemental Drake | 25 global per player | 125 total team gold |
| Rift Herald | 100 to killer | Eye drop for tower damage |
| Baron Nashor | 300 global per player | 1500 total team gold + buff |
| Elder Dragon | 25 global per player | + execute buff |
| Void Grub | 30-50 per grub | 6 total grubs available |

---

## XP Mechanics

### Champion Level XP Requirements

| Level | Cumulative XP | XP for This Level |
|-------|--------------|-------------------|
| 1 -> 2 | 280 | 280 |
| 2 -> 3 | 660 | 380 |
| 3 -> 4 | 1140 | 480 |
| 4 -> 5 | 1720 | 580 |
| 5 -> 6 | 2400 | 680 |
| 6 -> 7 | 3180 | 780 |
| 7 -> 8 | 4060 | 880 |
| 8 -> 9 | 5040 | 980 |
| 9 -> 10 | 6120 | 1080 |
| 10 -> 11 | 7300 | 1180 |
| 11 -> 12 | 8580 | 1280 |
| 12 -> 13 | 9960 | 1380 |
| 13 -> 14 | 11440 | 1480 |
| 14 -> 15 | 13020 | 1580 |
| 15 -> 16 | 14700 | 1680 |
| 16 -> 17 | 16480 | 1780 |
| 17 -> 18 | 18360 | 1880 |

### Minion XP

| Minion Type | Solo XP | Notes |
|------------|---------|-------|
| Melee minion | 60.45 | 95% of base 63.6 |
| Caster minion | 29.76 | 95% of base 31.3 |
| Siege/Cannon minion | 92.82 | 95% of base 97.7 |

- XP is **shared** among nearby champions; splitting with 1 other reduces to ~66-82% per champion (level-dependent)
- Solo lane XP advantage: ~30-40% more XP than duo lane per champion
- Champion kill XP: 42-990 XP (based on enemy champion level), +20% bonus at levels 1-6, +10% at levels 7-8

---

## Item System

### Item Tiers
- **Basic items**: 300-500 gold (components)
- **Epic items**: 800-1600 gold (intermediate)
- **Legendary items**: 2600-3600 gold (full items)

### Common Legendary Items with Costs

#### AD / Marksman Items
| Item | Total Cost | Key Stats |
|------|-----------|-----------|
| Infinity Edge | 3400 | 70 AD, 25% crit, passive: crits deal 40% bonus damage (requires 60% crit) |
| Bloodthirster | 3400 | 55 AD, 20% crit, 15% lifesteal |
| Lord Dominik's Regards | 3000 | 40 AD, 25% crit, 35% armor penetration |
| Phantom Dancer | 2800 | 30 AD, 25% crit, 30% AS |
| Blade of the Ruined King | 3200 | 40 AD, 25% AS, passive: 9% current HP on-hit |
| Kraken Slayer | 3200 | 50 AD, 25% crit, 35% AS |
| Yun Tal Wildarrows | 3000 | 55 AD, 25% crit |
| Rapid Firecannon | 2800 | 30 AD, 25% crit, 30% AS |
| Mortal Reminder | 2800 | 35 AD, 25% crit, grievous wounds |

#### AP / Mage Items
| Item | Total Cost | Key Stats |
|------|-----------|-----------|
| Rabadon's Deathcap | 3600 | 130 AP, passive: +35% total AP |
| Zhonya's Hourglass | 2900 | 80 AP, 45 armor, active: 2.5s stasis |
| Void Staff | 2800 | 65 AP, 40% magic penetration |
| Luden's Companion | 2900 | 90 AP, 10 magic pen, 600 mana |
| Shadowflame | 2800 | 100 AP, 12 magic pen |
| Banshee's Veil | 2600 | 80 AP, 45 MR, spell shield |
| Lich Bane | 2700 | 85 AP, 8% MS |
| Cryptbloom | 2850 | 70 AP, 30% magic pen |

#### Tank / Fighter Items
| Item | Total Cost | Key Stats |
|------|-----------|-----------|
| Guardian Angel | 3200 | 45 AD, 40 armor, passive: revive on death |
| Sterak's Gage | 3200 | 50 AD, 400 HP |
| Dead Man's Plate | 2900 | 300 HP, 45 armor |
| Force of Nature | 2900 | 350 HP, 60 MR |
| Thornmail | 2700 | 350 HP, 70 armor, grievous wounds |
| Randuin's Omen | 2700 | 400 HP, 60 armor |
| Warmog's Armor | 3000 | 800 HP, regen passive |
| Jak'Sho, The Protean | 3200 | 300 HP, 50 armor, 50 MR |

#### Boots
| Boot | Cost | Key Stats |
|------|------|-----------|
| Boots (Tier 1) | 300 | 25 MS |
| Berserker's Greaves | 1100 | 45 MS, 30% AS |
| Sorcerer's Shoes | 1100 | 45 MS, 18 magic pen |
| Plated Steelcaps | 1100 | 45 MS, 25 armor |
| Mercury's Treads | 1100 | 45 MS, 25 MR, 30% tenacity |
| Ionian Boots of Lucidity | 900 | 45 MS, 15 ability haste |
| Boots of Swiftness | 900 | 60 MS |
| Tier 3 Boot Upgrade | +750 | 50 MS, enhanced stats (requires 2 legendary items) |

---

## Minion Waves

### Wave Composition
- Waves spawn every **30 seconds** starting at 1:05
- **Normal wave**: 3 melee minions + 3 caster minions
- **Cannon wave**: 3 melee + 3 caster + 1 siege minion
- Cannon wave frequency: every 3rd wave (early), every 2nd wave (after 15:05), every wave (after 25:05)

### Minion Stats (Base)

| Minion Type | HP | AD | Armor | MR | MS |
|------------|----|----|-------|-----|-----|
| Melee | 480 | 12 | 0 | 0 | 325 |
| Caster | 320 | 22.5 | 0 | 0 | 325 |
| Siege/Cannon | 900 | 39 | 0 | 0 | 325 |
| Super | 2200 | 190 | 100 | 100 | 325 |

- Minion stats scale every 90 seconds
- Super minions spawn in a lane when the corresponding inhibitor is destroyed

---

## Dragon Types and Soul System

### Elemental Drakes (6 types + Elder)

| Drake | Slayer Buff (per stack, max 4) | Soul Buff (4 stacks) |
|-------|-------------------------------|---------------------|
| **Infernal** | +4% AD and AP per stack | Damaging attacks/abilities trigger AoE explosion dealing bonus adaptive damage (3s CD) |
| **Ocean** | +2% missing HP/mana regen per 5s per stack | Dealing damage heals for 150 (+26% bAD)(+17% AP)(+7% bonus HP) and restores mana over 4s |
| **Mountain** | +6% armor and MR per stack | Gain a shield for 200 (+18% bAD)(+13.5% AP)(+13.5% bonus HP) after 5s out of combat |
| **Cloud** | +3.5% MS (out of combat) per stack | +10% MS, +50% MS for 3s after casting ultimate |
| **Hextech** | +5 ability haste and +5% AS per stack | Chain lightning on attacks/abilities dealing true damage and slowing |
| **Chemtech** | +5% tenacity and healing/shielding power per stack | Below 50% HP: gain increasing damage, up to 10% at critical HP; on takedown heal for % missing HP |

### Elder Dragon
- Spawns after one team claims Dragon Soul (or at 35:00)
- **Aspect of the Dragon**: attacks burn enemies for 75-225 true damage over 2.25s (based on game time)
- **Execute threshold**: enemies below 20% HP are instantly killed by Elder-empowered damage
- Duration: 150 seconds
- Respawn: 6:00

### Rift Terrain Transformation
- After the second drake is slain, the Rift transforms based on the dominant element:
  - Infernal: brush removed, walls destroyed, new paths
  - Ocean: new brush, honeyfruit spawns
  - Mountain: new rock walls, narrow paths
  - Cloud: speed zones in jungle
  - Hextech: teleportation gates appear
  - Chemtech: camouflage zones in jungle

---

## Baron Nashor

### Spawn and Stats
- **Spawns**: 20:00
- **Respawn timer**: 6:00
- **HP**: 12500 (+180 per minute)
- **AD**: high and scales with game time

### Hand of Baron Buff
- **Duration**: 180 seconds (3 minutes)
- **Champion buffs**: up to 48 AD and 80 AP (scales with game time)
- **Minion empowerment** (when buff holder is near):
  - All minions: speed set to 92.5% of nearby champion speed (cap 500 MS)
  - All minions: basic attacks deal AoE splash damage
  - Melee minions: gain increased size, +75% MS toward champions, +75 attack range
  - Caster minions: +50% damage reduction from champions
  - Siege/Super minions: +600 attack range, attacks deal AoE in a large area
  - All non-super minions: take 75% reduced damage from AoE
- Empowerment deactivates when no buffed champion is within 1500 units

---

## Tower / Inhibitor Mechanics

### Turret Types

| Turret | HP | Armor/MR | Plates | Notes |
|--------|-----|---------|--------|-------|
| Outer turret | 5000 (+1000 per plate) | 40/40 | 5 | Plates fall at 14:00 |
| Inner turret | 3600 | 55/55 | 0 | |
| Inhibitor turret | 3300 | 55/55 | 0 | |
| Nexus turret (x2) | 2700 | 70/70 | 0 | |

### Turret Damage
- Base shot damage: ~180 (outer) to ~285 (nexus turrets)
- Ramps up 40% per consecutive shot on same champion (Warming Up)
- Turrets prioritize: minions > pets > champions (re-targets to champions attacking allied champions)

### Turret Plating
- 5 plates per outer turret, each plate = 1000 HP
- Destroying a plate: **125 local gold** to nearby champions
- Max gold per turret from plates: 625 gold
- Plates grant Bulwark (damage reduction) for 20s when broken, stacking
- Fortification: top/mid outer turrets take 50% reduced damage for first 5:00

### Inhibitors
- **HP**: 3000
- **Respawn time**: 5:00
- Destroying inhibitor: enemy lane spawns super minions until inhibitor respawns
- No direct gold reward

---

## Death Timer Formula

### Base Respawn Wait (BRW) by Champion Level

| Level | BRW (seconds) |
|-------|---------------|
| 1 | 10 |
| 2 | 10 |
| 3 | 12 |
| 4 | 12 |
| 5 | 14 |
| 6 | 16 |
| 7 | 20 |
| 8 | 25 |
| 9 | 28 |
| 10 | 32.5 |
| 11 | 35 |
| 12 | 37.5 |
| 13 | 40 |
| 14 | 42.5 |
| 15 | 45 |
| 16 | 47.5 |
| 17 | 50 |
| 18 | 52.5 |

### Time Scaling Factor (TIF)
- Before 15:00: TIF = 0%
- After 15:00: TIF increases by 2% every 30 seconds
- TIF cap: 50% (reached at ~55:00)

### Formula
```
Death Timer = BRW + (BRW x TIF)
```

### Example Death Timers
| Scenario | Timer |
|----------|-------|
| Level 1, early game | 10s |
| Level 6, 10:00 | 16s |
| Level 11, 20:00 | 38.5s |
| Level 16, 35:00 | ~62s |
| Level 18, 40:00 | ~70s |
| Level 18, 50:00+ | ~79s (max) |

---

## Typical Stats by Role (at 30 minutes, approximate)

| Metric | Top | Jungle | Mid | ADC | Support |
|--------|-----|--------|-----|-----|---------|
| CS | 220-260 | 160-200 | 220-260 | 240-280 | 30-60 |
| Gold | 9000-11000 | 8000-10000 | 9500-11500 | 10000-12000 | 5500-7500 |
| Level | 13-15 | 12-14 | 13-15 | 13-14 | 10-12 |
| Items completed | 2-3 | 2-3 | 2-3 | 3 | 1-2 |
| Kills+Assists | 5-10 | 6-12 | 5-12 | 5-10 | 5-15 |
| KDA | 2.0-4.0 | 2.5-4.5 | 2.5-5.0 | 3.0-6.0 | 2.0-5.0 |

---

## Match Timeline (Typical 30-minute game)

| Time | Event |
|------|-------|
| 0:00 | Game starts, buy starting items (500 gold budget) |
| 0:15-1:00 | Leash jungle or ward |
| 1:05 | Minion waves spawn |
| 1:30 | Jungle camps spawn |
| 3:00-3:30 | Jungler finishes first clear, looks for gank or scuttle |
| 5:00 | First dragon spawns, Void Grubs spawn |
| 6:00-8:00 | First ganks, potential First Blood |
| 8:00-10:00 | First turret plates contested |
| 14:00 | Plates fall, Rift Herald spawns, laning phase ends |
| 14:00-20:00 | Mid-game rotations, dragon stacking, tower trading |
| 20:00 | Baron Nashor spawns |
| 20:00-25:00 | Baron dance, vision control, 3rd/4th dragon fights |
| 25:00-30:00 | Baron attempts, dragon soul fight, inhibitor pushes |
| 30:00+ | Elder Dragon spawns (if soul taken), game-ending plays |
| 35:00-45:00 | Late game: single fight often determines outcome |

---

## Interference-Relevant Complexity

This domain is rich for narrative interference because:
- **Gold values change over time** (minion scaling, bounty accumulation)
- **Multiple overlapping systems** (gold, XP, items, objectives)
- **Contextual value shifts** (a kill worth 300 early vs 1000 with shutdown)
- **Role-dependent expectations** (support gold vs ADC gold norms differ by 2x)
- **Temporal dependencies** (plates only before 14:00, Baron only after 20:00)
- **Stacking mechanics** (dragon buffs 1-4, bounty tiers 0-8+)
