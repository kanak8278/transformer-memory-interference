# Formula 1 Racing -- Domain Mechanics Reference

## Overview

Formula 1 is the pinnacle of open-wheel single-seater motor racing. The 2024-2025 seasons feature **10 teams** with **20 cars** (2 per team), racing approximately 20-24 Grand Prix per season. Races run ~305 km (minimum 305 km or 2 hours, whichever comes first). The 2025 season is the last under the current technical regulations before major 2026 changes.

---

## 2024 Season Teams and Drivers

| Team | Driver 1 | Driver 2 | Power Unit |
|------|----------|----------|------------|
| Red Bull Racing | Max Verstappen | Sergio Perez | Honda RBPT |
| Ferrari | Charles Leclerc | Carlos Sainz Jr. | Ferrari |
| Mercedes | Lewis Hamilton | George Russell | Mercedes |
| McLaren | Lando Norris | Oscar Piastri | Mercedes |
| Aston Martin | Fernando Alonso | Lance Stroll | Mercedes |
| Alpine | Pierre Gasly | Esteban Ocon | Renault |
| RB (VCARB) | Yuki Tsunoda | Daniel Ricciardo / Liam Lawson | Honda RBPT |
| Haas | Nico Hulkenberg | Kevin Magnussen | Ferrari |
| Williams | Alex Albon | Logan Sargeant / Franco Colapinto | Mercedes |
| Kick Sauber | Valtteri Bottas | Zhou Guanyu | Ferrari |

### 2024 Championship Results
- **Drivers' Champion**: Max Verstappen (4th consecutive title)
- **Constructors' Champion**: McLaren

---

## 2025 Season Teams and Drivers

| Team | Driver 1 | Driver 2 | Power Unit |
|------|----------|----------|------------|
| Red Bull Racing | Max Verstappen | Liam Lawson | Honda RBPT |
| Ferrari | Charles Leclerc | Lewis Hamilton | Ferrari |
| Mercedes | George Russell | Kimi Antonelli | Mercedes |
| McLaren | Lando Norris | Oscar Piastri | Mercedes |
| Aston Martin | Fernando Alonso | Lance Stroll | Mercedes |
| Alpine | Pierre Gasly | Franco Colapinto | Renault |
| Haas | Esteban Ocon | Oliver Bearman | Ferrari |
| Racing Bulls (RB) | Yuki Tsunoda | Isack Hadjar | Honda RBPT |
| Williams | Alex Albon | Carlos Sainz Jr. | Mercedes |
| Kick Sauber | Nico Hulkenberg | Gabriel Bortoleto | Ferrari |

### Key 2025 Driver Changes
- Lewis Hamilton: Mercedes -> Ferrari (after 12 seasons at Mercedes)
- Carlos Sainz Jr.: Ferrari -> Williams
- Kimi Antonelli: F2 -> Mercedes (replacing Hamilton, youngest Mercedes driver)
- Liam Lawson: RB -> Red Bull Racing (replacing Perez)
- Oliver Bearman, Isack Hadjar, Gabriel Bortoleto: rookie entries

---

## Tire Compounds

### Dry Compounds (Pirelli, 2025)

| Compound | Code | Color | Characteristics | Typical Life (laps) | Performance Delta |
|----------|------|-------|----------------|---------------------|-------------------|
| C1 | Hard (hardest available) | White | Maximum durability, least grip | 40-50+ | Baseline |
| C2 | Hard/Medium | White | High durability | 35-45 | ~0.5-0.8s faster than C1 |
| C3 | Medium | Yellow | Balanced grip/durability | 25-35 | ~0.5-0.8s faster than C2 |
| C4 | Medium/Soft | Yellow | Higher grip, moderate durability | 20-30 | ~0.5-0.8s faster than C3 |
| C5 | Soft | Red | High grip, short life | 15-22 | ~0.5-0.8s faster than C4 |
| C6 | Soft (softest) | Red | Maximum grip, very short life | 10-18 | ~0.3-0.5s faster than C5 |

**Notes**:
- C6 was introduced for 2025 on low-degradation circuits (e.g., Monaco, Singapore)
- At each race, Pirelli selects 3 of the 6 compounds, labeling them Hard/Medium/Soft
- Typical compound allocation examples:
  - High degradation (Barcelona, Silverstone): C1, C2, C3
  - Medium degradation (Monza, Suzuka): C2, C3, C4
  - Low degradation (Monaco, Singapore): C3, C4, C5 or C4, C5, C6

### Wet Compounds

| Compound | Color | Usage | Water Displacement |
|----------|-------|-------|-------------------|
| Intermediate | Green | Light rain, drying track, damp conditions | ~25 liters/sec at speed |
| Full Wet | Blue | Heavy rain, standing water | ~65 liters/sec at speed |

### Tire Rules
- **Mandatory pit stop**: must use at least 2 different dry compounds during a dry race
- **Tire allocation per weekend**: 13 sets of dry tires (split across compounds), 3 sets intermediate, 2 sets wet
- **Tire blanket temperature**: 70 degrees C (front and rear)
- **Sprint races**: no mandatory tire change required

---

## Pit Stop Mechanics

### Time Breakdown

| Component | Duration | Notes |
|-----------|----------|-------|
| Pit lane speed limit | 80 km/h (race) | 60 km/h at Monaco, Melbourne, Singapore |
| Pit lane transit (entry + exit) | ~18-22 seconds | Varies by circuit pit lane length |
| Stationary time (tire change) | 2.0-3.5 seconds | Top teams target sub-2.5s |
| Total pit stop time loss | ~20-25 seconds | Compared to a lap at racing speed |

### Fastest Pit Stops
- World record (stationary time): ~1.80 seconds (Red Bull)
- Typical top-team stop: 2.0-2.5 seconds
- Typical midfield stop: 2.5-3.5 seconds
- Slow stop (issue): 4-8+ seconds

### Pit Crew
- **20+ mechanics** per car
- 3 per wheel (gun operator, wheel-off, wheel-on)
- 2 jack operators (front + rear)
- 1 front wing adjuster (if needed)
- Stabilizer operators

### Pit Stop Strategy
- **Typical race strategies**: 1-stop, 2-stop, sometimes 3-stop
- **Undercut**: pitting before rival to gain time on fresh tires vs. degraded tires
  - Typical undercut advantage: 1-3 seconds gained
- **Overcut**: staying out longer, exploiting a clear track and tire sweet spot
  - Works best when new tires take laps to warm up
- **Safety car stops**: free pit stop opportunity (pit lane time loss drops to ~10-12s vs 20-25s)

---

## DRS (Drag Reduction System)

### How It Works
- Rear wing flap opens by ~50mm, reducing drag
- Speed boost: +10-12 km/h (6.2-7.5 mph) on straights
- Top-speed gain varies by circuit: +15-20 km/h on long straights

### Activation Rules
- **Detection zone**: timing loop on track measures gap between cars
- **Activation zone**: begins 10-50 meters after detection zone on a straight
- **Gap requirement**: following car must be within **1.0 second** of car ahead at detection point
- **Manual activation**: driver pushes button on steering wheel after receiving electronic clearance
- **Automatic deactivation**: on braking or manual button press
- **Disabled conditions**: first 2 laps of race, 2 laps after Safety Car restart, wet conditions (Race Director decision)
- **Number of DRS zones per circuit**: typically 1-3 (most tracks have 2)

### 2025 Context
- 2025 is the **final season with DRS** -- it will be replaced by active aerodynamics in 2026

---

## ERS (Energy Recovery System) / Battery

### Current System (2014-2025 Regulations)

| Component | Specification |
|-----------|---------------|
| **MGU-K** (Motor Generator Unit - Kinetic) | Recovers energy from braking; deploys to rear axle |
| MGU-K max deployment power | 120 kW (~161 hp) |
| MGU-K max energy deployment per lap | 4 MJ |
| MGU-K max energy recovery per lap | 2 MJ |
| **MGU-H** (Motor Generator Unit - Heat) | Recovers waste heat from turbocharger exhaust |
| MGU-H energy recovery | Unlimited per lap (no cap) |
| **Energy Store (ES/Battery)** | Lithium-ion, stores energy from both MGUs |
| ES maximum energy | 4 MJ capacity |
| ES minimum weight | 20 kg |
| **Total Power Unit output** | ~1000 hp combined (ICE ~850 hp + ERS ~150 hp) |

### ERS Deployment Strategy
- **Qualifying**: maximum deployment for single-lap pace
- **Race**: managed over entire lap; more deployment on straights, less in corners
- **Lift-and-coast**: technique to conserve energy/fuel by lifting throttle early before braking zones
- **Harvesting penalty**: recovering energy creates drag, costs ~0.1-0.3s per lap

### 2026 Changes (Preview)
- MGU-H **eliminated**
- MGU-K power increased to **350 kW** (~470 hp)
- Battery capacity remains 4 MJ but deployment increased to 4 MJ bursts, unlimited bursts per lap
- ~50% ICE power + ~50% electric power = ~1000 hp total

---

## Fuel Load

### Regulations
- **Maximum fuel allowance**: 110 kg per race
- **Fuel flow limit**: 100 kg/hour maximum
- **Minimum fuel sample**: 1.0 liter must remain at end of race (for FIA testing)

### Performance Impact
- **Weight penalty**: ~0.03-0.04 seconds per kg per lap
- **Full tank penalty**: ~3.3-4.4 seconds per lap slower than empty car
- **Fuel burn rate**: ~1.5-2.0 kg per lap (circuit dependent)
  - Low fuel circuits (Monaco): ~1.3 kg/lap
  - High fuel circuits (Spa): ~2.2 kg/lap

### Strategy
- **Short-fueling**: carrying 5-15 kg less than maximum; faster per lap but requires lift-and-coast later
- **Fuel saving techniques**: lift-and-coast, engine mapping adjustments, coasting into corners
- Typical race fuel load: 95-110 kg depending on circuit and strategy

---

## Penalty Types

### In-Race Penalties

| Penalty | Severity | Application | Common Offenses |
|---------|----------|-------------|-----------------|
| **5-second time penalty** | Minor | Added to race time or served at pit stop | Causing a collision, track limits (repeated), unsafe rejoin |
| **10-second time penalty** | Moderate | Added to race time or served at pit stop | More serious collision, significant advantage gained |
| **Drive-through penalty** | Significant | Must drive through pit lane at pit speed (no stopping) | Pit lane speeding, unsafe release |
| **10-second stop-and-go** | Severe | Must stop in pit box for 10s, crew cannot touch car | Serious safety infringement, ignoring blue flags |
| **Black flag (DSQ)** | Maximum | Disqualified from race | Technical infringement, ignoring penalties, dangerous driving |
| **Black-and-white flag** | Warning | Shown once as final warning | Unsportsmanlike behavior |
| **Black-and-orange flag** | Mandatory | Must pit for repairs | Car damage deemed dangerous |

### Post-Race Penalties
- **Time penalties**: 5s, 10s, 20s, 30s added post-race
- **Grid penalties**: 3, 5, 10 places, or back of grid for next race
- **Disqualification**: results removed entirely
- **Reprimand**: official warning (3 reprimands in a season = 10-place grid penalty)

### Penalty Points (Super License)
- Drivers accumulate penalty points for infractions (1-3 per incident)
- **12 penalty points in 12 months = 1 race ban**
- Points expire after 12 months

### Component Penalties
- Exceeding PU allocation: grid penalties
  - Each driver allowed per season: 4 ICEs, 4 turbochargers, 4 MGU-Hs, 4 MGU-Ks, 2 Energy Stores, 2 Control Electronics, 8 exhaust systems
  - First excess component: 10-place grid penalty
  - Each subsequent excess: 5-place grid penalty
  - Back-of-grid penalty if accumulated penalties exceed grid positions

---

## Safety Car / Virtual Safety Car

### Full Safety Car (SC)

| Aspect | Details |
|--------|---------|
| Deployed | Significant incident requiring marshals on track |
| Speed | ~100-140 km/h (varies by circuit) |
| Field behavior | All cars line up behind Safety Car in race order |
| Gap closing | Field bunches together, gaps eliminated |
| Pit stops | Allowed (significantly reduced time loss: ~10-12s vs ~20-25s) |
| Typical duration | ~4-6 laps |
| Restart | Safety Car peels into pit; leader controls restart |
| Lapped cars | May be allowed to unlap themselves (wave-by) |
| Overtaking | Forbidden until Safety Car Line / start-finish line |

### Virtual Safety Car (VSC)

| Aspect | Details |
|--------|---------|
| Deployed | Minor incident, stranded car near marshal post |
| Speed reduction | All cars must slow by ~35% (sector target times enforced) |
| Field behavior | Cars maintain gaps (no bunching) |
| Pit stops | Allowed but reduced advantage compared to SC |
| Typical duration | 1-3 laps |
| Restart | Race Director announces VSC ending with ~10-15 second warning |

### Red Flag
- Race stopped completely; cars return to pit lane
- Deployed for: major crashes, extreme weather, track obstruction
- Standing restart or rolling restart when cleared
- Tire changes and repairs allowed under red flag (unlike SC/VSC)

---

## Qualifying Format

### Standard Qualifying (Saturday)

| Session | Duration | Cars | Elimination |
|---------|----------|------|-------------|
| **Q1** | 18 minutes | 20 cars | Bottom 5 eliminated (P16-P20) |
| **Q2** | 15 minutes | 15 cars | Bottom 5 eliminated (P11-P15) |
| **Q3** | 12 minutes | 10 cars | Sets grid P1-P10 |

### Key Rules
- **107% rule**: must lap within 107% of fastest Q1 time to be allowed to start race
- Multiple runs per session allowed
- Red flag: session clock stopped, restarted
- Cars may do 1-2 runs per session on new tire sets
- Track evolution: later runs often faster as rubber builds up

---

## Sprint Format (2025)

### Sprint Weekend Schedule (6 events per season)

| Day | Session | Duration |
|-----|---------|----------|
| Friday | Free Practice 1 | 60 minutes |
| Friday | Sprint Qualifying (Shootout) | SQ1: 12 min, SQ2: 10 min, SQ3: 8 min |
| Saturday | Sprint Race | ~100 km (~30 minutes, ~1/3 race distance) |
| Saturday | Grand Prix Qualifying | Q1: 18 min, Q2: 15 min, Q3: 12 min |
| Sunday | Grand Prix | Full race (~305 km) |

### Sprint Rules
- No mandatory tire change required
- Parc ferme applies from Sprint Qualifying onward
- Sprint distance: ~100 km (about 17-30 laps depending on circuit)

### 2025 Sprint Venues
China, Miami, Belgium, United States (Austin), Brazil, Qatar

---

## Points System (2025)

### Grand Prix Points (Top 10)

| Position | Points |
|----------|--------|
| 1st | 25 |
| 2nd | 18 |
| 3rd | 15 |
| 4th | 12 |
| 5th | 10 |
| 6th | 8 |
| 7th | 6 |
| 8th | 4 |
| 9th | 2 |
| 10th | 1 |

### Sprint Points (Top 8)

| Position | Points |
|----------|--------|
| 1st | 8 |
| 2nd | 7 |
| 3rd | 6 |
| 4th | 5 |
| 5th | 4 |
| 6th | 3 |
| 7th | 2 |
| 8th | 1 |

### Scoring Rules (2025 Changes)
- **Fastest lap point REMOVED** for 2025 (was 1 point in 2019-2024)
- **Shortened race rule**: if race is stopped and cannot restart:
  - <25% distance: no points
  - 25-50% distance: half points (rounded down)
  - 50-75% distance: full points
  - 75%+ distance: full points
- **Constructors' Championship**: sum of both drivers' points
- **Classification requirement**: must complete 90% of winner's distance to be classified

### Maximum Points Per Season (2025)
- Per race weekend (no sprint): 25 points
- Per sprint weekend: 25 + 8 = 33 points
- Theoretical season maximum (24 races, 6 sprints): 24 x 25 + 6 x 8 = 600 + 48 = 648 points

---

## Typical Lap Times and Circuit Data

### Representative Circuits (2024-2025 approximate)

| Circuit | Country | Lap Distance | Typical Qualifying Lap | Race Lap Range | Top Speed |
|---------|---------|-------------|----------------------|----------------|-----------|
| Monaco | Monaco | 3.337 km | ~1:10-1:12 | 1:12-1:16 | ~280 km/h |
| Monza | Italy | 5.793 km | ~1:19-1:21 | 1:22-1:27 | ~360+ km/h |
| Silverstone | UK | 5.891 km | ~1:25-1:27 | 1:27-1:32 | ~340 km/h |
| Spa-Francorchamps | Belgium | 7.004 km | ~1:42-1:44 | 1:45-1:50 | ~345 km/h |
| Suzuka | Japan | 5.807 km | ~1:28-1:30 | 1:31-1:36 | ~325 km/h |
| Bahrain | Bahrain | 5.412 km | ~1:28-1:30 | 1:31-1:36 | ~325 km/h |
| Jeddah | Saudi Arabia | 6.174 km | ~1:27-1:29 | 1:30-1:34 | ~335 km/h |
| Melbourne | Australia | 5.278 km | ~1:16-1:18 | 1:19-1:23 | ~325 km/h |
| Barcelona | Spain | 4.657 km | ~1:12-1:14 | 1:15-1:19 | ~330 km/h |
| Singapore | Singapore | 4.940 km | ~1:29-1:31 | 1:33-1:38 | ~310 km/h |
| Las Vegas | USA | 6.201 km | ~1:32-1:34 | 1:36-1:40 | ~345 km/h |

### Sector Timing
- Each lap divided into **3 sectors** (S1, S2, S3)
- Sector times displayed in real-time:
  - **Purple**: fastest overall (all-time session best)
  - **Green**: personal best
  - **Yellow**: slower than personal best
- Delta time: displayed to driver showing gap to reference lap (+/- seconds)

---

## Undercut / Overcut Strategy

### Undercut
- **Definition**: pitting before your rival to gain an advantage
- **Mechanism**: fresh tires provide a significant pace boost (~1-3s per lap for first 2-3 laps); the rival on old tires cannot match this pace
- **Works best when**: tire degradation is high, pit lane loss is low, traffic is clear
- **Risk**: rejoining in traffic, tires may overheat on out-lap
- **Typical gain**: 1-3 seconds if executed well

### Overcut
- **Definition**: staying out longer while rival pits
- **Mechanism**: clear air (no car ahead) allows faster laps; new tires may take 1-2 laps to reach optimal temperature
- **Works best when**: tire warm-up is slow (cold conditions), clear track, low degradation
- **Risk**: tires may fall off cliff, safety car could negate advantage
- **Typical gain**: 0.5-2 seconds if executed well

### Strategy Variables
- **Tire cliff**: sudden and dramatic loss of grip after extended use
- **Graining**: tire surface tears, reducing grip temporarily (often recovers)
- **Blistering**: overheating causes rubber to separate internally (permanent damage)
- **Tire management**: driving below limit to extend tire life; can cost 0.3-1.0s per lap

---

## Car Performance Specifications (2024-2025 Regulations)

| Specification | Value |
|---------------|-------|
| Minimum car weight (with driver) | 798 kg |
| Engine | 1.6L V6 turbo hybrid |
| Max RPM | 15,000 |
| Total power output | ~1000 hp (750 kW) |
| 0-100 km/h | ~2.6 seconds |
| 0-200 km/h | ~4.5 seconds |
| 0-300 km/h | ~10.5 seconds |
| Maximum speed | ~350-370 km/h (circuit dependent) |
| Maximum braking deceleration | ~6g |
| Maximum lateral g-force | ~5-6g |
| Maximum downforce | ~3.5x car weight at high speed |
| Fuel capacity | 110 kg maximum |
| Tire diameter | 720mm (18-inch wheels, introduced 2022) |

---

## Interference-Relevant Complexity

This domain is rich for narrative interference because:
- **20 drivers across 10 teams** with frequent driver swaps between seasons
- **Tire compound numbering** (C1-C6) vs. race labeling (Hard/Medium/Soft) creates aliasing
- **Points vary by event type** (sprint vs GP vs shortened race)
- **ERS specifications** involve precise energy values (MJ) and power limits (kW)
- **Fuel load** affects lap time through a continuous function (~0.035s per kg per lap)
- **Penalty types** have different severities and application rules
- **Strategy terms** (undercut/overcut) involve multi-step causal reasoning
- **Qualifying format** eliminates different numbers of cars per session (5/5/0)
- **2024 vs 2025 driver lineups** create temporal interference potential (who drove for which team when)
