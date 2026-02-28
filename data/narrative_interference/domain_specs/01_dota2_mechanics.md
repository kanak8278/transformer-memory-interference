Now I have enough data to compile a comprehensive response. Here is everything organized:

---

# Dota 2 Game Mechanics Reference for Simulator Building

## 1. Hero Pool

**Total heroes: 127** (as of patch 7.40, December 2025, with Largo as the latest addition).

**Picking restriction**: Each hero can only be picked once per match (unique draft -- no duplicates across both teams).

**Attribute distribution**: 36 Strength, 35 Agility, 34 Intelligence, 22 Universal.

**Complete hero list** (127 heroes, alphabetical):

Abaddon, Alchemist, Ancient Apparition, Anti-Mage, Arc Warden, Axe, Bane, Batrider, Beastmaster, Bloodseeker, Bounty Hunter, Brewmaster, Bristleback, Broodmother, Centaur Warrunner, Chaos Knight, Chen, Clinkz, Clockwerk, Crystal Maiden, Dark Seer, Dark Willow, Dawnbreaker, Dazzle, Death Prophet, Disruptor, Doom, Dragon Knight, Drow Ranger, Earth Spirit, Earthshaker, Elder Titan, Ember Spirit, Enchantress, Enigma, Faceless Void, Grimstroke, Gyrocopter, Hoodwink, Huskar, Invoker, Io, Jakiro, Juggernaut, Keeper of the Light, Kez, Kunkka, Largo, Legion Commander, Leshrac, Lich, Lifestealer, Lina, Lion, Lone Druid, Luna, Lycan, Magnus, Marci, Mars, Medusa, Meepo, Mirana, Monkey King, Morphling, Muerta, Naga Siren, Nature's Prophet, Necrophos, Night Stalker, Nyx Assassin, Ogre Magi, Omniknight, Oracle, Outworld Destroyer, Pangolier, Phantom Assassin, Phantom Lancer, Phoenix, Primal Beast, Puck, Pudge, Pugna, Queen of Pain, Razor, Riki, Ringmaster, Rubick, Sand King, Shadow Demon, Shadow Fiend, Shadow Shaman, Silencer, Skywrath Mage, Slardar, Slark, Snapfire, Sniper, Spectre, Spirit Breaker, Storm Spirit, Sven, Techies, Templar Assassin, Terrorblade, Tidehunter, Timbersaw, Tinker, Tiny, Treant Protector, Troll Warlord, Tusk, Underlord, Undying, Ursa, Vengeful Spirit, Venomancer, Viper, Visage, Void Spirit, Warlock, Weaver, Windranger, Winter Wyvern, Witch Doctor, Wraith King, Zeus.

*Note: This list is compiled from multiple sources. Verify against the official Dota 2 website or Liquipedia for the exact current roster, as minor additions/removals may have occurred in sub-patches.*

---

## 2. Game Phases

### Drafting/Picking Phase
**Ranked All Pick:**
- Ban phase: 15 seconds for each player to vote on a hero to ban. Half of voted heroes are randomly selected to be banned.
- Pick phase: Alternating picks between Radiant and Dire. ~30 seconds per pick. Each player has bonus time.

**Captain's Mode (competitive):**
- Ban/Pick order: First-pick team gets 3-2-2 ban order; second-pick gets 4-1-2. Pick order is 1-3-1 for both teams.
- Timer: 15 seconds for first ban phase, 30 seconds for subsequent bans and picks. Each captain has 130 seconds reserve time.

### In-Game Phases

| Phase | Time Window | Description |
|-------|------------|-------------|
| **Strategy/Pre-horn** | -1:30 to 0:00 | Buy starting items, move to lanes |
| **Laning Phase** | 0:00 to ~10:00-12:00 | Last-hitting, denying, trading in lanes |
| **Mid Game** | ~12:00 to ~26:00 | Tower pushes, Roshan, teamfights, rotations |
| **Late Game** | ~26:00+ | High-ground pushes, buyback fights, Roshan contention |

---

## 3. Gold Mechanics

### Starting Gold
**600 gold** per hero (reduced from 625 in a prior patch).

### Passive Gold (Periodic Gold)
- 1 reliable gold every 0.67 seconds starting at 0:00
- Approximately **90 gold/min** at game start
- Scales over time:

| Game Time | Gold Per Minute |
|-----------|----------------|
| 0:00 | ~100 |
| 5:00 | ~106 |
| 22:00 | ~112 |
| 40:00 | ~120 |
| 62:00+ | ~128 (cap) |

### Gold Types
- **Reliable gold**: Passive gold, bounty runes, hero kills. Cannot be lost on death.
- **Unreliable gold**: Creep kills, building kills, Hand of Midas, etc. Lost on death.

### Gold from Last Hits (Lane Creeps at game start, scaling +1 gold per melee / +3 gold per ranged every 7:30)

| Creep Type | Base Gold Bounty | Base XP Bounty |
|------------|-----------------|----------------|
| Melee Creep | 38-48 (avg ~40) | 57 |
| Ranged Creep | 43-53 (avg ~48) | 69 |
| Siege Creep | 66-80 (avg ~74) | 88 |
| Flagbearer Creep | 38-44 + 34-39 bonus (total ~75-83) | 57 |

### Neutral Creep Camp Gold (approximate total per camp)

| Camp Tier | Average Gold |
|-----------|-------------|
| Small Camp | ~59 |
| Medium Camp | ~74 |
| Large Camp | ~101 |
| Ancient Camp | ~164 |

Stacking penalty: The hero who stacked receives 15-20% reduced gold/XP from stacked creeps.

### Gold from Hero Kills

**Kill bounty formula:**
```
Kill Gold = 125 + (Killed Hero Level x 8) + Kill Streak Bonus
```

**Kill streak bonus values** (by consecutive kills of the dying hero):

| Streak | Bonus Gold |
|--------|-----------|
| 3 kills | 60 |
| 4 kills | 100 |
| 5 kills | 150 |
| 6 kills | 210 |
| 7 kills | 280 |
| 8 kills | 360 |
| 9 kills | 450 |
| 10+ kills | 550 |

**Assist gold formula:**
```
f = 15 + ((50 + (VictimNetWorth x 0.037)) / NumberOfHeroes)
AssistGold = (VictimTeamNetWorth / KillerTeamNetWorth) x f
```
All heroes within 1500 range who assisted (damaged or debuffed the victim) share assist gold equally.

### Gold Loss on Death
```
Gold Lost = NetWorth / 40
```
Only unreliable gold is lost. Reliable gold is never lost.

### Buyback Cost
```
Buyback Cost = 200 + (NetWorth x 0.13)
```
- Buyback cooldown: **8 minutes**
- Buyback penalty: Next death respawn timer increased by **+25 seconds**

### Tower Gold Bounty

| Tower Tier | Team Bounty (each player) | Last Hit Bonus |
|-----------|--------------------------|----------------|
| Tier 1 | 100 | 100-200 |
| Tier 2 | 120 | 100-200 |
| Tier 3 | 140 | 100-200 |
| Tier 4 | 160 | 100-200 |

Denied towers give half the team bounty to the enemy.

### Barracks Gold Bounty
- Melee Barracks: 155 team bounty
- Ranged Barracks: 90 team bounty

---

## 4. XP Mechanics

### XP from Creeps
- **Melee creep**: 57 XP
- **Ranged creep**: 69 XP
- **Siege creep**: 88 XP
- **Flagbearer creep**: 57 XP
- **Denied creep**: Enemy gets **50%** of the XP bounty

### XP Range
**1500 units** -- all allied heroes within 1500 range of a dying enemy unit share XP equally.

### XP from Hero Kills

**Kill XP formula:**
```
Kill XP = (100 + 0.13 x DeadHeroTotalXP) / n + StreakXP
```
Where `n` = number of heroes in range.

**Assist XP formula:**
```
Assist XP = (40 + 0.14 x DeadHeroTotalXP) / n + StreakValue
```

**AoE bonus factors** (by number of heroes in range):

| Heroes in Range | Kill Bonus Factor | Assist Bonus Factor |
|----------------|-------------------|---------------------|
| 1 | 0.124 | 0.30 |
| 2 | 0.124 | 0.30 |
| 3 | 0.108 | 0.20 |
| 4 | 0.081 | 0.15 |
| 5 | 0.065 | 0.12 |

**Streak XP formula:**
```
Streak XP = (1.25 x StreakLength^2 - 2.5 x StreakLength + 10) x DeadHeroLevel
```

### XP Required Per Level (Level 1-30)

| Level | XP to Next Level | Cumulative XP |
|-------|-----------------|---------------|
| 1 | 0 (start) | 0 |
| 2 | 240 | 240 |
| 3 | 360 | 600 |
| 4 | 480 | 1,080 |
| 5 | 600 | 1,680 |
| 6 | 620 | 2,300 |
| 7 | 660 | 2,960 |
| 8 | 740 | 3,700 |
| 9 | 890 | 4,590 |
| 10 | 930 | 5,520 |
| 11 | 970 | 6,490 |
| 12 | 1,010 | 7,500 |
| 13 | 1,050 | 8,550 |
| 14 | 1,100 | 9,650 |
| 15 | 1,150 | 10,800 |
| 16 | 1,200 | 12,000 |
| 17 | 1,250 | 13,250 |
| 18 | 1,350 | 14,600 |
| 19 | 1,500 | 16,100 |
| 20 | 1,590 | 17,690 |
| 21 | 1,600 | 19,290 |
| 22 | 1,850 | 21,140 |
| 23 | 2,100 | 23,240 |
| 24 | 2,350 | 25,590 |
| 25 | 2,600 | 28,190 |
| 26 | 3,500 | 31,690 |
| 27 | 4,500 | 36,190 |
| 28 | 5,500 | 41,690 |
| 29 | 6,500 | 48,190 |
| 30 | 7,500 | 55,690 |

*Note: Levels 7-8 and 14-18 are approximate interpolations from partial wiki data. Verify levels 7, 8, and 14-18 against the official wiki for exact values. The confirmed values are: levels 2-6 (240/360/480/600/620), levels 9-13 (890/930/970/1010/1050), levels 19-25 (1500/1590/1600/1850/2100/2350/2600), levels 26-30 (3500/4500/5500/6500/7500).*

---

## 5. Item System

Items can be sold at **50% of their purchase price**. Many active items have cooldowns. Items can be disassembled in some cases.

### Starting Items (0-600 gold)

| Item | Cost |
|------|------|
| Tango | 90 |
| Healing Salve | 100 |
| Clarity | 50 |
| Faerie Fire | 70 |
| Iron Branch | 50 |
| Circlet | 155 |
| Slippers of Agility | 140 |
| Gauntlets of Strength | 140 |
| Mantle of Intelligence | 140 |
| Band of Elvenskin | 450 |
| Quelling Blade | 100 |
| Stout Shield / Ring of Protection | 175 |
| Magic Stick | 200 |
| Boots of Speed | 500 |
| Wind Lace | 250 |
| Orb of Venom | 275 |

### Early Game Items (500-2000 gold)

| Item | Cost |
|------|------|
| Magic Wand | 450 |
| Bracer | 505 |
| Wraith Band | 505 |
| Null Talisman | 505 |
| Urn of Shadows | 880 |
| Medallion of Courage | 1025 |
| Power Treads | 1400 |
| Phase Boots | 1480 |
| Arcane Boots | 1300 |
| Hand of Midas | 2200 |
| Vanguard | 1825 |
| Hood of Defiance | 1500 |
| Drum of Endurance | 1700 |
| Vladmir's Offering | 2450 |

### Core Items (2000-5000 gold)

| Item | Cost |
|------|------|
| Blink Dagger | 2250 |
| Force Staff | 2200 |
| Eul's Scepter | 2725 |
| Shadow Blade | 3000 |
| Desolator | 3500 |
| Maelstrom | 2700 |
| Diffusal Blade | 2500 |
| Orchid Malevolence | 3475 |
| Sange and Yasha | 4100 |
| Kaya and Sange | 4100 |
| Yasha and Kaya | 4100 |
| Black King Bar (BKB) | 4050 |
| Battle Fury | 3900 |
| Aghanim's Scepter | 4200 |
| Manta Style | 4650 |
| Pipe of Insight | 3475 |
| Crimson Guard | 3600 |
| Lotus Orb | 3850 |

### Luxury Items (5000+ gold)

| Item | Cost |
|------|------|
| Radiance | 4700 |
| Daedalus | 5100 |
| Butterfly | 5450 |
| Eye of Skadi | 5300 |
| Satanic | 5050 |
| Heart of Tarrasque | 5000 |
| Assault Cuirass | 5250 |
| Monkey King Bar | 4975 |
| Scythe of Vyse (Sheepstick) | 5675 |
| Refresher Orb | 5200 |
| Divine Rapier | 5950 |
| Abyssal Blade | 6250 |
| Overwhelming Blink | 6800 |
| Swift Blink | 6800 |
| Arcane Blink | 6800 |

*Note: Costs may shift by 50-200 gold between patches. Verify against the current 7.40c patch notes.*

---

## 6. Creep/Lane Mechanics

### Spawn Timing
- First wave spawns at **0:00** (game horn)
- Subsequent waves every **30 seconds**
- Flagbearer creeps start at **2:00** (5th wave), then every 2nd wave
- Siege creeps start at **5:00** (11th wave), then every 5 minutes (every 10th wave)

### Wave Composition Over Time

| Time | Melee | Ranged | Siege (every 5 min) |
|------|-------|--------|---------------------|
| 0:00 | 3 | 1 | 1 (at 5:00+) |
| 15:00 | 4 | 1 | 1 |
| 30:00 | 5 | 1 | 1 |
| 35:00 | 5 | 1 | 2 |
| 40:00 | 5 | 2 | 2 |

### Creep Stat Upgrades
Every **7 minutes 30 seconds**, lane creeps gain permanent upgrades:
- Melee: +12 HP, +1 damage, +1 gold bounty
- Ranged: +12 HP, +2 damage, +3 gold bounty, +8 XP bounty

### Creep Equilibrium
Creeps meet at the midpoint of each lane. Equilibrium shifts when:
- One side has more creeps (from pulling, deaths, extra spawns)
- Denying your own creeps reduces enemy push
- Pulling neutral camps into the lane disrupts enemy creep wave
- Tower damage accelerates wave clearing

---

## 7. Tower/Building Mechanics

### Towers Per Lane: 3 (Tier 1, Tier 2, Tier 3) + 2 Tier 4 towers guarding the Ancient

**Total structures per team:**
- 3 lanes x 3 towers = 9 lane towers
- 2 Tier 4 towers (Ancient guards)
- 6 Barracks (3 Melee + 3 Ranged, one pair per lane)
- 1 Ancient
- 2 Shrines (varies by patch)

### Tower Stats

| Tier | Health | Armor | Attack Damage | Team Bounty |
|------|--------|-------|---------------|-------------|
| T1 | 1800 | 12 | ~100-120 | 100 gold/player |
| T2 | 1900 | ~15 | 220-240 | 120 gold/player |
| T3 | 2000 | ~18 | 170-180 | 140 gold/player |
| T4 | 2100 | 21 | 170-180 | 160 gold/player |

Towers gain **backdoor protection** (rapid HP regen) when no allied creeps are nearby.

### Barracks
- Destroying melee barracks: enemy lane spawns **super melee creeps** in that lane
- Destroying ranged barracks: enemy lane spawns **super ranged creeps** in that lane
- Destroying all 6 barracks: spawns **mega creeps** in all lanes (extremely powerful, usually game-ending)

### Ancient
- The Ancient is invulnerable until both Tier 4 towers are destroyed
- Cannot be denied
- Destroying the Ancient wins the game

---

## 8. Roshan

### Spawn & Respawn
- Initial spawn: Available from game start at 0:00
- Respawn timer after death: **8-11 minutes** (random within that window)

### Drops (depends on which pit -- Dire vs Radiant)

| Kill # | Dire Pit Drop | Radiant Pit Drop |
|--------|--------------|------------------|
| 1st | Aegis of the Immortal | Aegis of the Immortal |
| 2nd | Aegis + Cheese | Aegis + Roshan's Banner |
| 3rd | Aegis + Cheese + Aghanim's Blessing | Aegis + Banner + Refresher Shard |
| 4th+ | Same as 3rd | Same as 3rd |

### Drop Details
- **Aegis**: Holder respawns after 5 seconds with full HP/mana upon death. Expires after 5 minutes if unused.
- **Cheese**: Instantly restores 2500 HP and 1500 mana. Consumable.
- **Refresher Shard**: Single-use, resets all ability and item cooldowns.
- **Aghanim's Blessing**: Consumed, permanently upgrades hero abilities. Adds 5800 net worth.
- **Roshan's Banner**: Placed to buff allied creeps in a lane.

### Roshan Stats
Roshan grows stronger over time (gains HP, damage, armor as the game progresses).

---

## 9. Death and Respawn

### Respawn Time Formula
```
Respawn Time = 5 + (3.8 x Hero Level)
```
This gives 100 seconds at level 25.

| Level | Respawn Time (seconds) |
|-------|----------------------|
| 1 | ~9 |
| 5 | 24 |
| 10 | 43 |
| 15 | 62 |
| 20 | 81 |
| 25 | 100 |
| 30 | 119 |

### Buyback
- **Cost**: 200 + (NetWorth x 0.13)
- **Cooldown**: 8 minutes
- **Penalty**: +25 seconds added to next death's respawn timer

---

## 10. Match Timing

### Average Match Length
- **Public matches**: 35-42 minutes average (current patch ~41 minutes)
- **Pro matches**: 35-45 minutes
- **90% of games**: Between 23.5 and 59.5 minutes

### Key Timestamps

| Time | Event/Significance |
|------|-------------------|
| 0:00 | Game horn, first creep wave |
| 0:00 | Bounty runes spawn (then every 3 min) |
| 2:00 | Power runes spawn mid (then every 2 min) |
| 2:00 | Flagbearer creeps start |
| 5:00 | First siege creep wave |
| 5:00 | Neutral Tormentor available |
| 7:30 | First creep upgrade |
| ~10:00-12:00 | Laning phase typically ends |
| ~12:00-15:00 | Mid-game begins, first Roshan attempt for strong teams |
| 15:00 | +1 melee creep per wave |
| 20:00 | Mid-game power spike window (key items completed) |
| ~20:00-25:00 | Second Roshan typically contested |
| 30:00 | +1 melee creep per wave |
| 35:00 | +1 siege creep per wave |
| 40:00 | +1 ranged creep per wave |
| 45:00+ | Late game, buyback becomes critical |

### Power Spike Timing
- **Level 6** (~5-7 min): Ultimate abilities unlocked
- **Level 12** (~12-15 min): Second ultimate level
- **Level 18** (~20-25 min): Max ultimate level
- **First major item** (~12-18 min for cores): BKB, Blink, Battle Fury
- **Two core items** (~22-30 min): Full fighting capability

---

## 11. Team Composition & Roles

### Positions (Farm Priority)

| Position | Role | Lane | Farm Priority |
|----------|------|------|---------------|
| Pos 1 | Hard Carry (Safelane) | Safe Lane (bot Radiant / top Dire) | Highest |
| Pos 2 | Midlaner | Mid Lane | Second |
| Pos 3 | Offlaner | Off Lane (top Radiant / bot Dire) | Third |
| Pos 4 | Soft Support (Roamer) | Off Lane / Roaming | Fourth |
| Pos 5 | Hard Support | Safe Lane (with Pos 1) | Lowest |

### Farm Priority
- Pos 1 gets safe lane farm, jungle priority, and lane creeps after laning
- Pos 2 gets solo mid XP advantage, transitions to ganking or farming
- Pos 3 survives offlane, builds aura/utility items, initiates fights
- Pos 4 stacks camps, roams for kills, places offensive wards
- Pos 5 protects Pos 1 in lane, buys wards/detection, sacrifices farm entirely

---

## 12. Stats Per Role (Realistic Benchmarks)

### GPM by Role (average pub game, ~35-40 min game)

| Role | Low | Average | High (Good Game) |
|------|-----|---------|------------------|
| Pos 1 (Carry) | 500 | 600-650 | 700-800+ |
| Pos 2 (Mid) | 450 | 550-600 | 650-750+ |
| Pos 3 (Offlane) | 350 | 400-450 | 500-550 |
| Pos 4 (Soft Sup) | 250 | 300-350 | 400-450 |
| Pos 5 (Hard Sup) | 200 | 250-300 | 350-400 |

### XPM by Role

| Role | Low | Average | High |
|------|-----|---------|------|
| Pos 1 (Carry) | 550 | 650-700 | 800+ |
| Pos 2 (Mid) | 600 | 700-750 | 850+ |
| Pos 3 (Offlane) | 450 | 550-600 | 700+ |
| Pos 4 (Soft Sup) | 350 | 400-450 | 550+ |
| Pos 5 (Hard Sup) | 300 | 350-400 | 500+ |

### KDA by Role (typical averages)

| Role | Kills | Deaths | Assists | KDA Ratio |
|------|-------|--------|---------|-----------|
| Pos 1 (Carry) | 8-12 | 4-6 | 8-12 | ~3.5-4.5 |
| Pos 2 (Mid) | 8-12 | 5-7 | 10-14 | ~3.0-4.0 |
| Pos 3 (Offlane) | 4-8 | 6-8 | 12-18 | ~2.5-3.5 |
| Pos 4 (Soft Sup) | 3-6 | 7-9 | 15-20 | ~2.0-3.0 |
| Pos 5 (Hard Sup) | 1-4 | 8-11 | 15-22 | ~1.5-2.5 |

### Last Hits by Role at Different Timestamps

| Timestamp | Pos 1 | Pos 2 | Pos 3 | Pos 4 | Pos 5 |
|-----------|-------|-------|-------|-------|-------|
| 10 min | 55-70 | 50-65 | 30-45 | 5-15 | 0-5 |
| 20 min | 130-170 | 120-150 | 70-100 | 20-40 | 5-15 |
| 30 min | 220-280 | 200-250 | 120-160 | 30-60 | 10-25 |
| 40 min | 300-380 | 270-340 | 160-220 | 40-80 | 15-35 |

---

## Caveats for Your Simulator

1. **Many values scale over time** (creep bounties, Roshan stats, passive gold). Your simulator needs a time-based scaling system.
2. **XP table levels 7-8 and 14-18 are interpolated** from partial data. I was unable to scrape the full wiki table due to tool restrictions. You should verify these specific values against the [Dota 2 Wiki Experience page](https://dota2.fandom.com/wiki/Experience) or [Liquipedia Experience page](https://liquipedia.net/dota2/Experience).
3. **Item costs shift between patches** (sometimes by 50-200 gold). The values above are approximate for patch 7.40c.
4. **Gold and XP formulas have been simplified**. The actual game has additional edge cases (comeback mechanics, AoE gold distribution based on net worth differences, etc.).
5. **Neutral item drops** (free items from jungle at timed intervals) are not covered here but are significant for simulation accuracy.

---

## Sources

- [Dota 2 Wiki - Gold](https://dota2.fandom.com/wiki/Gold)
- [Dota 2 Wiki - Experience](https://dota2.fandom.com/wiki/Experience)
- [Dota 2 Wiki - Lane Creeps](https://dota2.fandom.com/wiki/Lane_Creeps)
- [Dota 2 Wiki - Buildings](https://dota2.fandom.com/wiki/Buildings)
- [Dota 2 Wiki - Death](https://dota2.fandom.com/wiki/Death)
- [Dota 2 Wiki - Roshan](https://dota2.fandom.com/wiki/Roshan)
- [Liquipedia - Gold](https://liquipedia.net/dota2/Gold)
- [Liquipedia - Experience](https://liquipedia.net/dota2/Experience)
- [Liquipedia - Buildings](https://liquipedia.net/dota2/Buildings)
- [Liquipedia - Hero List](https://liquipedia.net/dota2/Portal:Heroes)
- [Liquipedia - Game Modes](https://liquipedia.net/dota2/Game_Modes)
- [Official Dota 2 Heroes Page](https://www.dota2.com/heroes)
- [DotaBuff - Items Most Used](https://www.dotabuff.com/items)
- [Dota Coach - Game Length](https://dotacoach.gg/en/game-length)
- [How Many Heroes in Dota 2 - addrom.com](https://addrom.com/how-many-heroes-in-dota-2-all-characters-2026/)
- [Dota 2 Roles Explained 2025](https://pickem-mongolia.com/news/dota2-roles-2025/)
- [Skycoach - Heroes Tier List 7.40c](https://skycoach.gg/blog/dota-2/articles/heroes-tier-list)
- [GitHub - matyuschenko/dota2 Hero List](https://github.com/matyuschenko/dota2)