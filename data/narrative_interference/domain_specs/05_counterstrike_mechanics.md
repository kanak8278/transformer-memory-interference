# Counter-Strike 2 (CS2) Complete Game Mechanics Reference

## 1. Match Structure

### MR12 Format (Max Rounds 12)
- **Total regulation rounds**: 24 maximum (12 per half)
- **Rounds to win**: 13 (first team to 13 wins)
- **Side swap**: After 12 rounds, teams switch sides (T becomes CT and vice versa)
- **Sides**: Terrorists (T) vs Counter-Terrorists (CT)
- **Round time limit**: 1 minute 55 seconds (1:55)
- **Freeze time**: 15 seconds at the start of each round (buy phase)
- **Buy time**: 20 seconds from round start
- **Warmup**: Before match starts, players practice until all connect

### Overtime Rules
- **Format**: MR3 (6 rounds per overtime period)
- **Starting money**: $10,000 for all players
- **Structure**: 3 rounds on one side, then swap, 3 rounds on other side
- **Win condition**: First team to win 4 overtime rounds (4-2 or better)
- **Tie at 3-3**: Another overtime period begins (unlimited OT cycles)
- **Money resets**: Economy resets at start of each overtime period and at side swap within OT

### Match Duration
- **Average match**: ~30-40 minutes (regulation)
- **With overtime**: Can extend to 50+ minutes
- **Per round**: ~2 minutes average (including freeze time, action, and post-round)

---

## 2. Economy System -- EXACT Numbers

### Starting Money
- **Pistol round (Round 1 and Round 13)**: $800
- **All other rounds**: Carry over from previous round + round reward

### Max Money Cap
- **$16,000** -- cannot accumulate more than this

### Round Win Rewards
| Condition | Reward |
|-----------|--------|
| Round win (elimination) | $3,250 |
| Round win (bomb explosion, T side) | $3,500 |
| Round win (bomb defuse, CT side) | $3,500 |
| Round win (time runs out, CT side) | $3,250 |

### Round Loss Rewards (Loss Bonus Ladder)
| Consecutive Losses | Reward |
|--------------------|--------|
| 1st loss | $1,400 |
| 2nd consecutive loss | $1,900 |
| 3rd consecutive loss | $2,400 |
| 4th consecutive loss | $2,900 |
| 5th+ consecutive loss (max) | $3,400 |

- Loss bonus increases by $500 per consecutive loss
- **Resets to $1,400** after winning a round
- **Pistol round loss**: $1,900 (starts at 2nd tier)

### Special Bonuses
| Action | Reward |
|--------|--------|
| Bomb plant (individual planter) | $300 |
| Team bomb plant (T side loses after plant) | +$800 to all T players |
| Defuse (individual defuser) | $300 |

### Kill Rewards by Weapon Category
| Weapon Category | Kill Reward |
|-----------------|-------------|
| Knife | $1,500 |
| Pistols (most) | $300 |
| CZ75-Auto | $100 |
| SMGs (most) | $600 |
| P90 | $300 |
| Shotguns | $900 |
| Rifles | $300 |
| AWP | $100 |
| Auto-snipers (SCAR-20, G3SG1) | $300 |
| SSG 08 | $300 |
| Machine guns (Negev, M249) | $300 |
| Grenades (HE, Molotov/Incendiary) | $300 |
| Zeus x27 | $0 |

---

## 3. Weapon System -- Complete List

### Pistols

| Weapon | Price | Side | Kill Reward | Base Damage | Armor Pen | Magazine | Notes |
|--------|-------|------|-------------|-------------|-----------|----------|-------|
| Glock-18 | $200 (free for T) | T | $300 | 30 | 47% | 20/120 | Burst fire mode, default T pistol |
| USP-S | $200 (free for CT) | CT | $300 | 35 | 50.5% | 12/24 | Silenced, default CT pistol option |
| P2000 | $200 (free for CT) | CT | $300 | 35 | 50.5% | 13/52 | Default CT pistol option, larger mag than USP-S |
| P250 | $300 | Both | $300 | 38 | 64% | 13/26 | Budget upgrade, can 1-tap at close range no helmet |
| Five-SeveN | $500 | CT | $300 | 32 | 91.15% | 20/100 | High armor pen, CT only |
| Tec-9 | $500 | T | $300 | 33 | 90.15% | 18/90 | High armor pen, T only |
| CZ75-Auto | $500 | Both | $100 | 33 | 77.65% | 12/12 | Full-auto pistol, low kill reward, limited ammo |
| Dual Berettas | $300 | Both | $300 | 38 | 52% | 30/120 | Two pistols, large magazine |
| Desert Eagle | $700 | Both | $300 | 63 | 93.2% | 7/35 | One-shot headshot through helmet, HS multiplier 3.9x |
| R8 Revolver | $600 | Both | $300 | 86 | 93.2% | 8/8 | Delayed trigger (right-click for fan fire), very high damage |

### SMGs

| Weapon | Price | Side | Kill Reward | Base Damage | Armor Pen | Magazine | Fire Rate (RPM) |
|--------|-------|------|-------------|-------------|-----------|----------|-----------------|
| MAC-10 | $1,050 | T | $600 | 29 | 57.5% | 30/100 | 800 |
| MP9 | $1,250 | CT | $600 | 26 | 60% | 30/120 | 857 |
| MP7 | $1,500 | Both | $600 | 29 | 62.5% | 30/120 | 750 |
| MP5-SD | $1,500 | Both | $600 | 27 | 62.5% | 30/120 | 750, silenced |
| UMP-45 | $1,200 | Both | $600 | 35 | 65% | 25/100 | 667 |
| PP-Bizon | $1,400 | Both | $600 | 27 | 57.5% | 64/120 | 750, largest SMG mag |
| P90 | $2,350 | Both | $300 | 26 | 69% | 50/100 | 857, reduced kill reward |

### Rifles

| Weapon | Price | Side | Kill Reward | Base Damage | Armor Pen | Magazine | Notes |
|--------|-------|------|-------------|-------------|-----------|----------|-------|
| Galil AR | $1,800 | T | $300 | 30 | 77.5% | 35/90 | Budget T rifle |
| FAMAS | $2,050 | CT | $300 | 30 | 70% | 25/90 | Budget CT rifle, burst mode |
| AK-47 | $2,700 | T | $300 | 36 | 77.5% | 30/90 | One-shot headshot through helmet, 600 RPM |
| M4A4 | $3,100 | CT | $300 | 33 | 70% | 30/120 | Cannot 1-tap helmet, 4x HS multiplier |
| M4A1-S | $2,900 | CT | $300 | 38 | 70% | 20/80 | Silenced, HS multiplier 3.475x, smaller mag |
| SG 553 | $3,000 | T | $300 | 30 | 100% | 30/90 | Scoped, 100% armor penetration |
| AUG | $3,300 | CT | $300 | 28 | 90% | 30/90 | Scoped CT rifle |

### Sniper Rifles

| Weapon | Price | Side | Kill Reward | Base Damage | Armor Pen | Magazine | Notes |
|--------|-------|------|-------------|-------------|-----------|----------|-------|
| SSG 08 | $1,700 | Both | $300 | 88 | 85% | 10/90 | "Scout", one-shot headshot, cheap |
| AWP | $4,750 | Both | $100 | 115 | 97.5% | 10/30 | One-shot kill (chest+head), low kill reward |
| SCAR-20 | $5,000 | CT | $300 | 80 | 82.5% | 20/90 | Auto-sniper, CT side |
| G3SG1 | $5,000 | T | $300 | 80 | 82.5% | 20/90 | Auto-sniper, T side |

### Heavy Weapons

#### Shotguns

| Weapon | Price | Side | Kill Reward | Base Damage (per pellet) | Pellets | Armor Pen | Magazine |
|--------|-------|------|-------------|--------------------------|---------|-----------|----------|
| Nova | $1,050 | Both | $900 | 26 | 9 | 50% | 8/32 |
| XM1014 | $2,000 | Both | $900 | 20 | 6 | 80% | 7/32 |
| MAG-7 | $1,300 | CT | $900 | 30 | 9 | 75% | 5/32 |
| Sawed-Off | $1,100 | T | $900 | 32 | 9 | 75% | 7/32 |

#### Machine Guns

| Weapon | Price | Side | Kill Reward | Base Damage | Armor Pen | Magazine | Notes |
|--------|-------|------|-------------|-------------|-----------|----------|-------|
| M249 | $5,200 | Both | $300 | 32 | 80% | 100/200 | Expensive, rarely bought |
| Negev | $1,700 | Both | $300 | 35 | 75% | 150/200 | Cheap, massive spray, wind-up mechanic |

### Equipment

| Item | Price | Side | Notes |
|------|-------|------|-------|
| Kevlar Vest | $650 | Both | Reduces body damage, 100 armor points |
| Kevlar + Helmet | $1,000 | Both | Vest + head protection; $350 upgrade if vest already owned |
| Defuse Kit | $400 | CT only | Halves defuse time (10s to 5s) |
| Zeus x27 | $200 | Both | One-use taser, instant kill at close range, $0 kill reward |

---

## 4. Grenade System

### Prices and Carry Limits

| Grenade | Price | Carry Limit | Side |
|---------|-------|-------------|------|
| Flashbang | $200 | 2 | Both |
| Smoke Grenade | $300 | 1 | Both |
| HE Grenade | $300 | 1 | Both |
| Molotov | $400 | 1 | T only |
| Incendiary Grenade | $600 | 1 | CT only |
| Decoy Grenade | $50 | 1 | Both |

### Total grenade carry: 4 grenades maximum per player
- Example loadout: 2 flashbangs + 1 smoke + 1 molotov = 4 grenades (full)
- Molotov and Incendiary count as the same slot (side-specific variants)

### Grenade Damage Values

| Grenade | Max Damage (unarmored) | Max Damage (armored) | Radius | Duration |
|---------|------------------------|----------------------|--------|----------|
| HE Grenade | ~98 | ~57 | ~3.5m lethal radius | Instant |
| Molotov/Incendiary | ~40/sec | ~36/sec | Spreads on ground | ~7 seconds |
| Flashbang | 0 (blinds) | 0 (blinds) | ~2 second full blind at close range | Flash duration varies by angle/distance |
| Smoke Grenade | 0 | 0 | ~288 unit radius | ~18 seconds |
| Decoy Grenade | ~1-2 (on explosion) | ~1 | N/A | ~15 seconds then self-destructs |

### Key Grenade Mechanics
- Molotov/Incendiary fire can be extinguished by a smoke grenade
- Flashbang effectiveness depends on: distance, angle (looking at vs away), and obstacles
- Smoke grenades are volumetric in CS2 (3D cloud, not flat sprite)
- HE grenade damage falls off with distance from center
- Decoy mimics the sound of the player's most expensive weapon

---

## 5. Health and Armor System

### Health
- **Max HP**: 100
- **No natural regeneration** -- damage is permanent for the round
- **Headshot multiplier**: 4.0x for most weapons (exceptions: M4A1-S = 3.475x, Desert Eagle = 3.9x)

### Armor
- **Max armor value**: 100
- **Kevlar vest ($650)**: Protects chest, stomach, and arms
- **Kevlar + Helmet ($1,000)**: Also protects head

### Armor Damage Reduction
- Armor reduces damage based on the weapon's **armor penetration** percentage
- Damage formula: `final_damage = base_damage * armor_penetration_ratio`
- Remaining damage absorbed by armor (armor loses durability proportional to absorbed damage)
- **Helmet protection**: Prevents one-shot headshot kills from most weapons
  - AK-47 CAN one-shot headshot through helmet
  - M4A4/M4A1-S CANNOT one-shot headshot through helmet
  - AWP one-shots regardless of armor
  - Desert Eagle CAN one-shot headshot through helmet at close/medium range
  - SSG 08 CANNOT one-shot headshot through helmet

### Practical Damage Examples (armored head)
| Weapon | Headshot Damage (armored) | One-shot kill? |
|--------|---------------------------|----------------|
| AK-47 | ~111 | Yes |
| M4A4 | ~92 | No |
| M4A1-S | ~92 | No |
| AWP | ~448 | Yes |
| Desert Eagle | ~233 | Yes (close range) |
| Glock-18 | ~56 | No |
| USP-S | ~70 | No |
| P250 | ~97 | No (but yes without helmet) |

### Armor Penetration Key Values
| Weapon | Armor Penetration |
|--------|-------------------|
| SG 553 | 100% |
| AWP | 97.5% |
| Desert Eagle | 93.2% |
| Five-SeveN | 91.15% |
| Tec-9 | 90.15% |
| AUG | 90% |
| SSG 08 | 85% |
| SCAR-20 / G3SG1 | 82.5% |
| XM1014 | 80% |
| M249 | 80% |
| AK-47 / Galil AR | 77.5% |
| CZ75-Auto | 77.65% |
| Negev | 75% |
| MAG-7 / Sawed-Off | 75% |
| M4A4 / M4A1-S / FAMAS | 70% |
| P90 | 69% |
| UMP-45 | 65% |
| MP7 / MP5-SD | 62.5% |
| MP9 | 60% |
| MAC-10 / PP-Bizon | 57.5% |
| P250 / Dual Berettas | ~52-64% |
| USP-S / P2000 | 50.5% |
| Nova | 50% |
| Glock-18 | 47% |

---

## 6. Maps

### Active Duty Map Pool (as of early 2026)

| Map | Setting | Bombsites | CT/T Sided | Notes |
|-----|---------|-----------|------------|-------|
| Dust2 | Middle Eastern desert | A, B | Slightly T-sided | Iconic, simple layout |
| Mirage | Moroccan cityscape | A, B | Slightly CT-sided | Mid control is critical |
| Inferno | Italian village | A, B | CT-sided | Banana control key for B |
| Nuke | Nuclear facility | A, B (stacked vertically) | CT-sided | Unique vertical layout |
| Overpass | German park/canal | A, B | Slightly CT-sided | Complex rotations |
| Ancient | Mayan ruins | A, B | Balanced | Newer competitive map |
| Anubis | Egyptian temple | A, B | Balanced | Returned to pool Jan 2026 |

### Dust2 Callouts (Complete)

**T Side / Approach:**
- T Spawn -- Terrorist starting area
- T Plat -- Elevated area left of T Spawn
- T Ramp -- Path from T Spawn toward B Tunnels
- Outside Tunnels -- Open area between T Spawn and Tunnel entrance
- Upper Tunnels (Upper B / Upper Dark) -- Upper section of B tunnels
- Lower Tunnels (Lower B) -- Lower section connecting to B site
- Mid Doors (T Mid / Suicide) -- Opening from T Spawn toward mid, highly exposed
- Xbox -- Large box in mid near catwalk access

**Mid:**
- Mid -- Central corridor connecting T and CT sides
- Mid Doors (Double Doors) -- Metal doors CTs can AWP through
- Catwalk (Cat / Short A) -- Elevated path from mid to A site
- A Short -- Upper portion of catwalk entering A site

**A Site:**
- A Long (Long / Long Doors) -- Long corridor from T Spawn to A site
- Long Doors -- Double doors at the start of Long A
- Blue / Long Corner -- Corner at end of Long A
- Pit -- Sunken area at end of Long A, common AWP position
- A Cross -- Crossing point between Long and A site
- A Ramp -- Ramp leading up to A site
- A Site -- Bombsite A
- Goose -- Corner behind A site (near the goose graffiti)
- A Platform (Plat / Elevator) -- Elevated position on A site
- CT Ramp -- Ramp from CT spawn to A site
- A Car -- Vehicle near A site
- Short Stairs -- Steps connecting short to A site
- Ninja -- Hidden corner on A site behind boxes

**B Site:**
- B Tunnels -- Tunnels leading to B site
- B Doors -- Entrance from tunnels to B site
- Upper B -- Upper tunnel area
- B Site -- Bombsite B
- B Plat (B Platform) -- Elevated platform on B site
- Back Plat -- Far corner behind B platform
- B Car -- Vehicle on B site
- Big Box -- Large box near tunnel entrance
- Double Stack -- Stacked boxes on B site
- B Closet -- Small alcove on B site
- B Window -- Window overlooking B site
- B Back Site -- Rear of B site
- Fence -- Wall near B site entrance
- Dog (Close) -- Close left corner entering B from tunnels

**CT Side:**
- CT Spawn -- Counter-Terrorist starting area
- CT Mid -- Mid area near CT spawn
- B Doors (CT B Doors) -- Double doors between CT mid and B site
- A Short (CT Short) -- Path from CT spawn to A short

### Mirage Callouts (Complete)

**T Side:**
- T Spawn -- Terrorist starting area
- T Ramp (A Ramp) -- Ramp from T Spawn toward A site
- Palace -- Building above A Ramp with windows overlooking A site
- Top Mid -- Elevated area where Ts first see mid
- Mid Boxes -- Stack of boxes at Top Mid for cover

**Mid:**
- Mid -- Central area of the map
- Window (Sniper's Nest) -- CT position overlooking mid
- Connector -- Passage from mid to A site behind CT spawn
- Underpass -- Tunnel below mid connecting to B Short area
- Ladder Room -- Room with ladder connecting mid to B Short
- Cat (Catwalk) -- Narrow path from mid to B Short
- Chair -- Position in mid near window

**A Site:**
- A Site -- Bombsite A
- A Default -- Standard plant spot
- A Ramp -- Approach from T Spawn
- Palace -- Upper approach to A
- Stairs -- Steps on A site
- Sandwich (Between Boxes) -- Gap between stacked boxes on A
- Tetris -- L-shaped boxes on A site
- Triple Box (Triple) -- Three stacked boxes, common CT position
- Firebox -- Box near connector side of A
- Ticket Booth (Ticket) -- Booth structure on A site
- CT (CT Spawn side of A) -- Back of A site toward CT Spawn
- Jungle -- Vegetated area connecting CT spawn to A connector
- Balcony -- Wooden balcony above A site
- Shadows -- Area under the balcony
- Pillars -- Pillar structures near Palace and A site
- Ninja -- Hidden spot on A site
- Under Palace -- Lower entrance from Palace to A

**B Site:**
- B Site -- Bombsite B
- B Apartments (B Apps) -- Building complex leading to B site
- B Short -- Short approach to B from mid/underpass
- Van -- Vehicle on B site
- Bench -- Bench at back of B site
- Market (Door/Window) -- Room behind B site, CT rotation route
- Market Door -- Door from Market into B
- Market Window -- Window from Market overlooking B
- Kitchen -- Room adjacent to Market
- Arches -- Arched entrance from Kitchen/Market to B site
- B Default -- Standard plant position
- Back B (Back Site) -- Rear wall area of B site
- Pillar -- Pillar on B site

**CT Side:**
- CT Spawn -- Counter-Terrorist starting area
- Truck -- Vehicle in CT Spawn area
- Bench (CT Bench) -- Near CT spawn
- Shop -- Building near CT spawn
- Scaffolding -- Structure near B apartments

---

## 7. Round Types and Buy Strategies

### Pistol Round (Rounds 1 and 13)
- **Budget**: $800
- **T side typical buys**:
  - Glock (free) + Kevlar ($650) + remaining on utility
  - Glock (free) + Smoke ($300) + Flash ($200) + Flash ($200) + $100 remaining
  - P250 ($300) + Light utility
- **CT side typical buys**:
  - USP-S (free) + Kevlar ($650) + remaining on utility
  - USP-S (free) + Defuse Kit ($400) + Flash ($200) + Flash ($200)
  - P250 ($300) + Kevlar ($650) = exactly $950 (leaves nothing)

### Eco Round (Save Round)
- **Budget**: Usually $1,400-$2,400 (but saving for next round)
- **Goal**: Spend $0-$500 max, save money for next round's full buy
- **Typical buys**: Default pistol only, maybe a P250 ($300) or a few flashbangs
- **When**: After losing pistol round (round 2 or 14), or when team economy is broken

### Force Buy
- **Budget**: $2,000-$3,500 (not enough for full buy but buying anyway)
- **Goal**: Gamble on winning with inferior equipment
- **Typical buys**:
  - Tec-9/Five-SeveN ($500) + Kevlar ($650) + utility
  - Galil/FAMAS ($1,800-$2,050) + minimal or no armor
  - Deagle ($700) + Kevlar ($650) + some utility
- **When**: Score is critical (match point), after bomb plant giving extra $800, or second-round force after pistol loss

### Half Buy
- **Budget**: $3,500-$4,500
- **Goal**: Moderate equipment while maintaining some economy
- **Typical buys**:
  - SMG (MAC-10/MP9, $1,050-$1,250) + Kevlar+Helmet ($1,000) + some utility
  - Galil/FAMAS + Kevlar + limited utility
- **When**: Transitional rounds, team partially recovered economically

### Full Buy (Gun Round)
- **Budget**: $4,500+ (ideally $5,000-$6,500)
- **T side full buy** (~$4,750-$5,500):
  - AK-47 ($2,700) + Kevlar+Helmet ($1,000) + Smoke ($300) + Flash ($200) + Molotov ($400) = $4,600
  - Add Flash ($200) = $4,800
- **CT side full buy** (~$5,500-$6,800):
  - M4A4 ($3,100) + Kevlar+Helmet ($1,000) + Defuse Kit ($400) + Smoke ($300) + Flash ($200) + Incendiary ($600) = $5,600
  - Add Flash ($200) = $5,800
- **AWP player full buy**:
  - AWP ($4,750) + Kevlar+Helmet ($1,000) + Smoke ($300) + Flash ($200) = $6,250

---

## 8. Bomb Mechanics

### Bomb Plant
- **Plant time**: 3.2 seconds (must hold use key)
- **Plant locations**: Only within designated bombsite areas (A or B)
- **Only T side** can plant
- **Planter reward**: $300 personal bonus
- **Team reward on loss after plant**: +$800 to all T players

### Bomb Timer
- **Explosion countdown**: 40 seconds after plant
- **C4 beeping**: Increases in frequency as timer counts down
- **At ~10 seconds**: Beeping becomes very rapid

### Bomb Defuse
- **Without defuse kit**: 10 seconds
- **With defuse kit ($400)**: 5 seconds
- **Defuse must be uninterrupted** -- taking damage does not interrupt, but dying does
- **Latest defuse start (no kit)**: Must begin by 30 seconds remaining
- **Latest defuse start (with kit)**: Must begin by 35 seconds remaining
- **"Ninja defuse"**: Defusing while enemies are unaware, often in smoke

### Post-Plant Positions (Strategic)
- **Terrorists**: Hold angles covering the bomb from positions that force CTs to expose themselves
- **Counter-Terrorists**: Must retake the site, clear positions, then defuse under time pressure
- **Common T post-plant**: Off-site positions watching from distance, using utility to delay
- **Key consideration**: 40-second timer means CTs have limited time to retake AND defuse

---

## 9. Typical Stats by Skill Level

### Premier Rating / Rank Tiers
| Tier | Premier Rating | Old Rank Equivalent |
|------|---------------|---------------------|
| Low | 0-4,999 | Silver 1 - Silver Elite Master |
| Below Average | 5,000-9,999 | Gold Nova 1 - Gold Nova Master |
| Average | 10,000-14,999 | Master Guardian 1 - Distinguished Master Guardian |
| Above Average | 15,000-19,999 | Legendary Eagle - Supreme Master First Class |
| High | 20,000-24,999 | Global Elite level |
| Elite | 25,000-30,000+ | Semi-pro / FPL level |
| Professional | N/A (HLTV rated) | Tier 1-3 pro teams |

### Average Stats by Skill Level

| Stat | Low Rank | Average | Above Average | High Rank | Professional |
|------|----------|---------|---------------|-----------|-------------|
| ADR | 50-65 | 65-75 | 75-85 | 85-95 | 80-100+ |
| Headshot % | 25-35% | 35-45% | 45-52% | 50-58% | 48-65% |
| KAST % | 55-62% | 62-68% | 68-74% | 74-80% | 70-82% |
| Rating 2.0 | 0.70-0.90 | 0.90-1.00 | 1.00-1.10 | 1.10-1.25 | 1.00-1.30 |
| K/D Ratio | 0.60-0.85 | 0.85-1.00 | 1.00-1.15 | 1.10-1.30 | 1.00-1.40 |
| First Kill % | 8-12% | 12-16% | 16-20% | 18-24% | 15-25% |
| Clutch Win % | 5-10% | 10-15% | 15-22% | 20-30% | 18-35% |

### HLTV Rating 2.0 Components
- **KPR** (Kills Per Round)
- **DPR** (Deaths Per Round)
- **APR** (Assists Per Round)
- **ADR** (Average Damage per Round)
- **KAST%** (% of rounds with Kill, Assist, Survived, or Traded)
- **Impact** (measures multi-kills, opening kills, clutches)
- Formula weights these components; 1.00 is average for professional play

---

## 10. Match Timeline

### Pre-Round (Freeze Time): 0-15 seconds
- Players frozen in spawn
- Buy phase: purchase weapons, armor, grenades
- IGL calls strategy for the round
- Team discusses positions and executes

### Early Round: 0-30 seconds after freeze
- **T side**: Take map control, push for information, set up utility
- **CT side**: Hold positions, gather information, watch for rushes
- **Key actions**: AWP picks, early aggression, smoking choke points
- **Utility usage**: Early smokes to block sightlines, flashes for information

### Mid Round: 30-75 seconds
- **T side**: Execute strategy, fake sites, split pushes, default play (spreading out)
- **CT side**: Hold angles, rotate on information, stack sites if read is correct
- **Key actions**: Map control battles, utility trading, picks
- **T "default"**: Spread across map, look for picks, gather information before committing

### Late Round: 75-115 seconds (last 40 seconds)
- **T side**: Must commit to a site, execute with remaining utility
- **CT side**: Rotate based on information, play retake if needed
- **Key pressure**: Time running out forces T aggression
- **If bomb not planted by ~1:15**: Ts are in trouble

### Post-Plant: 0-40 seconds (bomb timer)
- **T side**: Play time, hold positions, use remaining utility to delay defuse
- **CT side**: Must retake site, clear enemies, and defuse within 40 seconds
- **Retake**: CTs push site together using coordinated utility
- **Trades**: Each side tries to get favorable exchanges

### Round Types by Pace
- **Rush**: T side pushes a site immediately (0-15 seconds), overwhelming with speed
- **Execute**: Coordinated utility + push at 30-60 second mark
- **Default**: Slow play, spread map control, hit when opportunity arises
- **Retake**: CTs give up site, regroup, push back together with utility

---

## 11. Team Roles

### AWPer
- **Primary weapon**: AWP ($4,750)
- **Function**: Long-range picks, holding angles, map control
- **Typical stats**: Lower HS% (30-40%), high impact, high ADR
- **Economy impact**: Most expensive player on the team
- **Famous examples**: s1mple, ZywOo, dev1ce
- **Expected stats**: ~0.70-0.85 KPR, Rating 1.10-1.30, lower KAST (dies holding angles)
- **Key skill**: Flick shots, holding angles, repositioning

### Entry Fragger
- **Primary weapon**: AK-47 / M4A4
- **Function**: First player into sites, creates space, gets opening kills or dies trying
- **Typical stats**: Highest first-kill %, high deaths, high ADR
- **Expected stats**: ~20-25% opening duel rate, Rating 1.00-1.15, high impact
- **Famous examples**: dupreeh, YEKINDAR, Magisk
- **Key skill**: Fast aim, movement, peeking, trading with support

### In-Game Leader (IGL)
- **Primary weapon**: AK-47 / M4A4 (sometimes utility-focused)
- **Function**: Calls strategies, reads opponent, manages economy, mid-round adjustments
- **Typical stats**: Often lowest stats on team (distracted by calling)
- **Expected stats**: Rating 0.90-1.05, lower ADR (80-90), compensates with KAST%
- **Famous examples**: gla1ve, karrigan, FalleN, Aleksib
- **Key skill**: Game sense, communication, adaptability, reading opponents

### Lurker
- **Primary weapon**: AK-47 / M4A4
- **Function**: Plays off the team, flanks, catches rotators, gathers information
- **Typical stats**: High clutch rate, fewer opening kills, good survival rate
- **Expected stats**: Rating 1.00-1.15, high KAST%, moderate ADR
- **Famous examples**: flusha, ropz, Twistzz
- **Key skill**: Timing, patience, game sense, 1vX clutch ability

### Support Player
- **Primary weapon**: AK-47 / M4A4 (sometimes cheaper to afford full utility)
- **Function**: Throws utility for team, sets up teammates, trades kills
- **Typical stats**: Highest assist numbers, high KAST%, lower personal stats
- **Expected stats**: Rating 0.95-1.10, high utility damage, most flashes thrown
- **Famous examples**: Xyp9x, SANJI, Perfecto
- **Key skill**: Grenade lineups, positioning for trades, communication

---

## 12. Economy Patterns

### Typical 15-Round Half (T Side Example)

| Round | Type | Scenario | Approx Team Money | Buy |
|-------|------|----------|-------------------|-----|
| 1 | Pistol | Start with $800 each | $800 | Glock + utility |
| 2 (Win R1) | Force/SMG | Won pistol, ~$3,500 each | $3,500 | SMG + Kevlar+Helmet + utility |
| 2 (Lose R1) | Eco | Lost pistol, ~$1,900 each | $1,900 | Save (maybe P250) or force Tec-9 + Kevlar |
| 3 (Won R1+R2) | Full Buy | ~$6,000+ each | $6,000+ | AK + full utility |
| 3 (Lost R1, eco R2) | Force/Full | ~$3,800-$4,500 | $4,000 | Galil/AK + Kevlar (light utility) |
| 3 (Lost R1+R2) | Full Buy | ~$4,300 (loss bonus $2,400 + saved) | $4,300 | AK + Kevlar + some utility |
| 4-6 | Gun Rounds | Full buy vs full buy | $5,000-$6,000 | AK/AWP + full equipment |
| 7 (after loss streak) | Eco | Economy broken | $1,400-$3,400 | Save for next round |
| 8 | Full Buy | Saved + loss bonus | $5,000+ | Full buy attempt |
| 9-12 | Mixed | Depends on results | Varies | Alternating gun rounds and resets |

### Economy Decision Rules
1. **Full buy threshold**: ~$4,700+ for T side, ~$5,500+ for CT side
2. **Eco if**: Team average money < $2,000 AND cannot afford rifles
3. **Force buy if**: Match point, or bomb planted (extra $800), or 2+ players can buy rifles
4. **Save (eco) if**: Only 1-2 rounds until team can full buy together
5. **Anti-eco**: After winning vs eco, buy SMGs for $600 kill reward bonus

### Key Economy Concepts
- **Team buying**: All 5 players should buy at the same level (all eco or all buy)
- **Dropping weapons**: Rich players buy weapons for poor teammates
- **Max money warning**: At $16,000 cap, must spend or waste potential earnings
- **Reset**: Winning a round resets loss bonus to $1,400 -- devastating if team then loses
- **Second-round force after plant**: Common T strategy -- if you plant bomb in pistol round but lose, the $800 plant bonus means you can force buy round 2
- **CT economy is harder**: CT weapons cost more (M4 vs AK), incendiary costs $600 vs $400 molotov, defuse kits needed

### Typical Economy Flow Patterns
- **3-0 start after pistol win**: Common, gives huge economic advantage
- **Pistol loss -> eco -> buy round 3**: Standard reset pattern
- **Double eco**: Rare, only when economy is deeply broken
- **"Broke" threshold**: When team has <$1,500 average and needs 2+ rounds of saving

---

## Appendix: Quick Reference Numbers

| Parameter | Value |
|-----------|-------|
| Starting money (pistol round) | $800 |
| Max money | $16,000 |
| Round time | 1:55 |
| Freeze time | 15 seconds |
| Buy time | 20 seconds |
| Bomb plant time | 3.2 seconds |
| Bomb explosion timer | 40 seconds |
| Defuse time (no kit) | 10 seconds |
| Defuse time (with kit) | 5 seconds |
| Max HP | 100 |
| Max Armor | 100 |
| Rounds to win (regulation) | 13 |
| Total regulation rounds | 24 |
| Side swap | After round 12 |
| Overtime format | MR3, $10,000 start |
| Overtime rounds to win | 4 (out of 6) |
| Headshot multiplier (standard) | 4.0x |
| Max grenades carried | 4 |
| Max flashbangs carried | 2 |
| Players per team | 5 |
