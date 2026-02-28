# PUBG (PlayerUnknown's Battlegrounds) -- Complete Game Mechanics Reference

This document provides exhaustive game mechanics data for building a PUBG game state simulator.
All values are for PUBG: Battlegrounds PC version unless otherwise noted.

---

## 1. Player Setup

### Match Sizes
- **Solo**: 100 players, no teams
- **Duo**: 100 players, 50 teams of 2
- **Squad**: 100 players, 25 teams of 4
- **Ranked Mode**: 64 players (no bots, no red zone, faster early circles)

### Plane Path Mechanics
- All 100 players spawn inside a C-130 cargo plane.
- The plane follows a **randomized linear flight path** across the map each match. The path is determined at match start and does not change.
- The plane flies at an altitude of approximately **1,500 meters** above sea level.
- The plane speed is approximately **360 km/h**.
- Players can jump at any point along the flight path by pressing the jump key.
- Players who do not jump are automatically ejected at the end of the flight path.
- The plane crosses the full map in roughly **45-60 seconds** (for 8x8 km maps).

### Parachuting Mechanics
- After jumping, players enter **free fall**.
- Free fall speeds:
  - Neutral (no input): accelerates to **180 km/h** downward, 0 lateral movement
  - Diving straight down (forward key + looking down): **234 km/h** downward, 0 lateral
  - Gliding (forward key + looking up): **126 km/h** vertical, **122 km/h** lateral
- **Automatic parachute deployment**: at **300 meters above ground level** (AGL, not sea level -- deploys earlier over mountains)
- Manual early deployment is possible by pressing the deploy key.
- Parachute descent speeds:
  - Neutral: **54 km/h** down, 0 lateral
  - Forward: **64 km/h** down, **47 km/h** lateral
- Maximum horizontal reach from jump point: approximately **2.0-2.5 km** with optimal gliding.
- Time from jump to landing: approximately **60-90 seconds** depending on technique and altitude.

---

## 2. Map Details

### Map Roster (as of 2025)

| Map | Size | Players | Theme | Status |
|-----|------|---------|-------|--------|
| Erangel | 8x8 km | 100 | Eastern European island (temperate) | Active |
| Miramar | 8x8 km | 100 | Mexican desert | Active |
| Sanhok | 4x4 km | 100 | Southeast Asian tropical | Active |
| Vikendi | 6x6 km | 100 | Adriatic island (snow/thaw) | Active |
| Taego | 8x8 km | 100 | South Korean countryside | Active |
| Deston | 8x8 km | 100 | American city/swamp | Active |
| Karakin | 2x2 km | 64 | North African arid | Rotational |
| Paramo | 3x3 km | 64 | South American volcanic highlands | Rotational |
| Haven | 1x1 km | 32 | Industrial urban (night) | Rotational |
| Rondo | 8x8 km | 100 | Mediterranean coastal | Active (newer) |

### Erangel -- All Major Named Locations

Erangel is the original and most iconic map. The following is a comprehensive list of named/major points of interest:

**Northern Region:**
- Zharki (northwest coastal town)
- Severny (north-central town with church)
- Stalber (northeast elevated ruins/lookout)
- Kameshki (northeast coastal town)
- Shooting Range (north-central, military structures)

**Central-North Region:**
- Georgopol (major port city, west coast, split into North and South by river; includes container yard and crate stacks)
- Georgopol Hospital (on hill overlooking Georgopol)
- Yasnaya Polyana (large city, northeast; one of biggest loot zones)
- Rozhok (central small town, divided by road)
- School (south of Rozhok, multi-story building, high-tier loot hot drop)
- Shelter (underground bunker, central-north)

**Central Region:**
- Pochinki (central town, dense buildings, notorious hot drop)
- Farm (south of Pochinki, open fields with scattered compounds)
- Ruins (central, stone ruins)
- Water Town (partially submerged town near Rozhok)
- Apartments (east of Pochinki, high-rise buildings)

**Western Region:**
- Gatka (west-central, small town with radio tower)
- Mansion (west, large standalone compound)
- Crater (west, distinctive terrain feature)
- Spawn Island (far northwest, small island off coast -- plane starts near here but rarely accessible)

**Eastern Region:**
- Lipovka (east, small town)
- Prison (east, large compound)
- Mylta (east-central, small town)
- Mylta Power (east coast, nuclear power plant -- high-tier loot)
- Ferry Pier (east coast, dock area)

**Southern Region (Main Island):**
- Primorsk (southwest coast, town with church and warehouses)
- Quarry (south, open pit mine)

**Southern Island (Military Island):**
- Sosnovka Military Base (large military complex -- highest tier loot, barracks, hangars, watchtowers)
- Novorepnoye (southeast coast, port town with container yard)
- North and South Bridges (two bridges connecting main island to military island -- notorious choke points)

**Other:**
- Various unnamed compounds, clusters, and warehouses scattered across the map
- Radio towers, bunkers, and observation posts

### Miramar Notable Locations
Los Leones, El Pozo, Pecado, Hacienda del Patron, San Martin, Chumacera, Monte Nuevo, Water Treatment, Puerto Paraiso, La Cobreria, Impala, Los Higos, Tierra Bronca, Campo Militar, Ladrillera, Torre Ahumada, Alcantara, Cruz del Valle, Minas Generales, El Azahar, Graveyard, Power Grid, Prison

### Sanhok Notable Locations
Paradise Resort, Bootcamp, Camp Alpha, Camp Bravo, Camp Charlie, Ruins, Pai Nan, Ha Tinh, Ban Tai, Sahmee, Tat Mok, Mongnai, Khao, Lakawi, Cave, Quarry, Docks, Kampong

### Vikendi Notable Locations
Castle, Cosmodrome, Dinopark, Dobro Mesto, Goroka, Hot Springs, Lumber Yard, Movatra, Peshkova, Pilnec, Trevno, Tovar, Vihar, Volnova, Winery, Zabava

---

## 3. Health System

### Base Health
- **Maximum HP: 100** (displayed as a bar; internally the game uses 100 health points)
- Health does not regenerate naturally. Must use consumables.
- At 0 HP, the player is either **knocked** (DBNO in team modes) or **killed** (solo mode).

### Healing Items

| Item | Heal Amount | Cast Time | Notes |
|------|-------------|-----------|-------|
| Bandage | 10 HP (over ~4s after cast) | 4 seconds | Heals only up to 75 HP. Stackable (carry up to 20). |
| First Aid Kit | Heals TO 75 HP (not +75) | 6 seconds | Cannot be used above 75 HP. Instantly sets HP to 75. |
| Med Kit | Heals TO 100 HP | 8 seconds | Fully restores health. Rare. |

- **Bandage**: Most common. Heals 10 HP over time after the 4s cast. Caps at 75 HP. To go from ~10 HP to 75 HP you need approximately 7 bandages and ~50 seconds.
- **First Aid Kit**: Sets health to 75 instantly after 6s cast. Most efficient common heal.
- **Med Kit**: Full heal to 100. 8s cast. Found rarely in world loot, more common in care packages.

### Boost Items (Adrenaline/Energy System)

PUBG has a **boost meter** (0-100%) displayed as a bar above the health bar. Boost provides:
- **Passive healing over time** (the higher the boost, the faster the heal)
- **Movement speed bonus** at high boost levels

| Boost Level | Effect |
|-------------|--------|
| 0-20% | No effect |
| 20-60% | Heals ~1% HP per 8 seconds |
| 60-90% | Heals ~1% HP per 5 seconds |
| 90-100% | Heals ~1% HP per 3 seconds + **movement speed boost (~6.2%)** |

| Boost Item | Boost Increase | Cast Time | Total HP Healed (over duration) |
|------------|----------------|-----------|--------------------------------|
| Energy Drink | +40% boost | 4 seconds | ~23 HP over ~2 minutes |
| Painkiller | +60% boost | 7.5 seconds | ~40 HP over ~3 minutes |
| Adrenaline Syringe | +100% boost (fills completely) | 10 seconds | ~86 HP over ~5 minutes + speed boost |

- **Energy Drink**: Common. Quick cast. Good for topping off.
- **Painkiller**: Uncommon. Strong sustained healing.
- **Adrenaline Syringe**: Rare (care package / rare world spawn). Best boost item -- fills meter completely.

### Damage Over Time (Blue Zone)
- See Section 8 for exact blue zone damage values per phase.
- Blue zone damage is applied every second.
- Blue zone damage **does not** have any bleed/DOT after leaving the zone -- damage stops immediately when you enter the safe zone.

---

## 4. Armor System

### Helmets

| Tier | Name | Damage Reduction (Head) | Durability (HP) |
|------|------|------------------------|------------------|
| Level 1 | Motorcycle Helmet | 30% | 80 |
| Level 2 | Military Helmet | 40% | 150 |
| Level 3 | Spetsnaz Helmet | 55% | 230 |

### Vests

| Tier | Name | Damage Reduction (Body) | Durability (HP) |
|------|------|------------------------|------------------|
| Level 1 | Police Vest | 30% | 200 |
| Level 2 | Police Vest (Level 2) | 40% | 220 |
| Level 3 | Military Vest | 55% | 250 |

### How Armor Works
- Damage reduction percentage is **constant regardless of remaining durability**. A Level 3 vest with 1 durability remaining still reduces 55% of body damage.
- When durability reaches 0, the armor is **destroyed** and removed.
- **After destruction**, vests still provide a residual **20% damage reduction** until fully gone (disputed -- some sources say 0% after break).
- Durability is consumed based on the **damage absorbed**. If a bullet deals 40 damage to the body and a Level 2 vest (40% reduction) is worn:
  - Player takes: 40 * 0.60 = 24 HP damage
  - Vest absorbs: 40 * 0.40 = 16 durability damage
- **Head armor (helmets)** and **body armor (vests)** are independent systems.
- Limb shots (arms, legs) are NOT protected by any armor. Limb damage multiplier is typically 0.5x-0.75x of base damage.

### Head Damage Multiplier
- **Headshot multiplier: 2.3x** for most weapons (varies slightly by weapon type)
- A Kar98k (79 base damage) headshot = 79 * 2.3 = 181.7 raw damage
  - vs Level 2 helmet: 181.7 * 0.60 = 109 damage (one-shot kill from 100 HP)
  - vs Level 3 helmet: 181.7 * 0.45 = 81.8 damage (survivable with full HP)

---

## 5. Weapon System

### Weapon Categories and Complete Weapon List

#### Assault Rifles (AR)
Uses: 5.56mm or 7.62mm ammo. Versatile mid-range. Full-auto, burst, or single fire.

| Weapon | Damage | Ammo | Fire Rate (RPM) | Mag Size (default/extended) | Notes |
|--------|--------|------|-----------------|---------------------------|-------|
| AKM | 47 | 7.62mm | 600 | 30/40 | High damage, high recoil |
| M416 | 40 | 5.56mm | 750 | 30/40 | Most versatile, most attachments |
| SCAR-L | 42 | 5.56mm | 620 | 30/40 | Balanced, stable |
| M16A4 | 43 | 5.56mm | 750 (burst) | 30/40 | Burst/single only, no full auto |
| Beryl M762 | 44 | 7.62mm | 700 | 30/40 | High damage + high fire rate, heavy recoil |
| QBZ | 42 | 5.56mm | 750 | 30/40 | Sanhok exclusive |
| G36C | 41 | 5.56mm | 750 | 30/40 | Vikendi exclusive (replaces SCAR-L) |
| AUG A3 | 41 | 5.56mm | 680 | 30/40 | Crate weapon. Very stable. |
| Groza | 47 | 7.62mm | 750 | 30/40 | Crate weapon. Highest AR DPS. |
| Mk47 Mutant | 49 | 7.62mm | 600 | 20/30 | Burst/single only. High damage per shot. |
| ACE32 | 43 | 7.62mm | 680 | 30/40 | Versatile 7.62 AR |
| K2 | 41 | 5.56mm | 750 | 30/40 | Taego exclusive |
| FAMAS | 42 | 5.56mm | 900 (burst) | 25/35 | Deston exclusive, burst fire |

#### Designated Marksman Rifles (DMR)
Semi-automatic. Mid-to-long range.

| Weapon | Damage | Ammo | Fire Rate (RPM) | Mag Size | Notes |
|--------|--------|------|-----------------|----------|-------|
| Mini 14 | 46 | 5.56mm | 300 | 20/30 | High velocity, low drop |
| SKS | 53 | 7.62mm | 300 | 10/20 | Takes many attachments |
| SLR | 58 | 7.62mm | 300 | 10/20 | High damage, high recoil |
| QBU | 48 | 5.56mm | 300 | 10/20 | Sanhok exclusive, built-in bipod |
| VSS Vintorez | 43 | 9mm | 330 | 10/20 | Built-in suppressor + 4x scope. Low velocity. |
| Mk12 | 50 | 5.56mm | 300 | 20/30 | Versatile DMR |
| Mk14 EBR | 61 | 7.62mm | 340 | 10/20 | Crate weapon. Can go full-auto. Highest DMR damage. |
| Dragunov | 60 | 7.62mm | 300 | 10/20 | Replaces SLR on some maps |

#### Sniper Rifles (SR)
Bolt-action (except Lynx). One shot potential.

| Weapon | Damage | Ammo | Mag Size | Notes |
|--------|--------|------|----------|-------|
| Kar98k | 79 | 7.62mm | 5 (internal) | World spawn. One-shots Level 2 helmet. |
| M24 | 79 | 7.62mm | 5/7 | World spawn. Similar to Kar98k. |
| Mosin-Nagant | 79 | 7.62mm | 5 (internal) | Replaces Kar98k on some maps |
| Win94 | 66 | .45 ACP | 8 (internal) | Miramar exclusive. Iron sight / 2.7x built-in. |
| AWM | 105 | .300 Magnum | 5/7 | Crate only. One-shots ANY helmet. Highest damage in game. |
| Lynx AMR | 180 | .50 BMG | 5 | Crate only. Anti-materiel rifle. Can damage vehicles. |

#### Submachine Guns (SMG)
Close range. High fire rate. Low recoil.

| Weapon | Damage | Ammo | Fire Rate (RPM) | Mag Size | Notes |
|--------|--------|------|-----------------|----------|-------|
| UMP45 | 41 | .45 ACP | 650 | 25/35 | Stable, good range for SMG |
| Micro UZI | 26 | 9mm | 1200 | 25/35 | Extremely high fire rate |
| Vector | 31 | 9mm | 1100 | 19/33 | High fire rate, fast TTK up close |
| Tommy Gun | 40 | .45 ACP | 700 | 30/50 | No scope attachment. Large drum mag option. |
| MP5K | 33 | 9mm | 900 | 30/40 | Vikendi exclusive |
| PP-19 Bizon | 35 | 9mm | 700 | 53 | Huge default mag, no extended mag |
| P90 | 36 | 5.7mm | 900 | 50 | Crate weapon. Built-in scope rail. |
| MP9 | 30 | 9mm | 1100 | 25/32 | Deston exclusive. Very fast fire rate. |
| JS9 | 36 | 9mm | 900 | 30/40 | Newer SMG |

#### Light Machine Guns (LMG)

| Weapon | Damage | Ammo | Fire Rate (RPM) | Mag Size | Notes |
|--------|--------|------|-----------------|----------|-------|
| DP-28 | 52 | 7.62mm | 550 | 47 | World spawn. Built-in bipod. Pan magazine. |
| M249 | 41 | 5.56mm | 750 | 75/150 | Crate weapon (moved to world spawn in some updates) |
| MG3 | 42 | 7.62mm | 660/990 | 75 | Crate weapon. Selectable fire rate. |

#### Shotguns

| Weapon | Damage (per pellet) | Pellets | Total Damage | Ammo | Mag Size | Notes |
|--------|---------------------|---------|-------------|------|----------|-------|
| S686 | 26 | 9 | 234 | 12 gauge | 2 | Double barrel. Two quick shots. |
| S1897 | 26 | 9 | 234 | 12 gauge | 5 | Pump action. |
| S12K | 22 | 9 | 198 | 12 gauge | 5/8 | Semi-auto. Takes AR attachments. |
| DBS | 26 | 9 | 234 | 12 gauge | 14 | Crate weapon. Double barrel, pump reload. |
| Sawed-Off | 22 | 9 | 198 | 12 gauge | 2 | Pistol slot. Miramar exclusive. |

#### Pistols

| Weapon | Damage | Ammo | Mag Size | Notes |
|--------|--------|------|----------|-------|
| P92 | 35 | 9mm | 15/20 | Default spawn pistol |
| P1911 | 41 | .45 ACP | 7/12 | Higher damage per shot |
| P18C | 23 | 9mm | 17/25 | Full-auto capable |
| R45 | 55 | .45 ACP | 6 | Revolver. Miramar exclusive. |
| R1895 | 64 | 7.62mm | 7 | Nagant revolver. Very slow reload. |
| Skorpion | 22 | 9mm | 20/40 | Full-auto machine pistol |
| Deagle | 62 | .45 ACP | 7/10 | High damage semi-auto |
| Flare Gun | N/A | Flare | 1 | Calls care package (inside zone) or BRDM (outside zone) |

#### Melee Weapons

| Weapon | Damage | Notes |
|--------|--------|-------|
| Pan | 80 | Also blocks bullets when on back (acts as armor) |
| Crowbar | 60 | |
| Machete | 60 | |
| Sickle | 60 | |

#### Throwables

| Weapon | Damage | Radius | Fuse/Timer | Notes |
|--------|--------|--------|-----------|-------|
| Frag Grenade | 100+ (at center) | ~5m lethal, ~10m damage | 5 seconds | Can cook (hold to reduce fuse). Deals 20% less to prone players. |
| Molotov Cocktail | ~10 DPS (fire) | ~4m spread | Instant on impact | Creates fire area. Burns for ~7 seconds. Spreads on surfaces. |
| Smoke Grenade | 0 | ~10m smoke cloud | 3s fuse + ~15s duration | Visual concealment. Blocks thermal scope. |
| Stun Grenade | 0 | ~20m effective range | 2.5s fuse | Blinds for 0.5-5.5 seconds depending on distance and facing. |
| C4 / Sticky Bomb | 500+ | ~25m effective radius | 16 seconds (auto detonation) | Penetrates cover. Destroys vehicles. Cannot be detonated early. |
| Blue Zone Grenade | 0 | Creates mini blue zone | Tactical | Taego exclusive. Creates small damaging zone. |

### Ammo Types

| Ammo | Used By | Weapons |
|------|---------|---------|
| 5.56mm | ARs, DMRs, LMGs | M416, SCAR-L, M16A4, QBZ, G36C, AUG, K2, FAMAS, Mini14, QBU, Mk12, M249 |
| 7.62mm | ARs, DMRs, SRs, LMGs, Pistol | AKM, Beryl, Mk47, ACE32, Groza, SKS, SLR, Mk14, Dragunov, DP-28, MG3, Kar98k, M24, Mosin, R1895 |
| .45 ACP | SMGs, Pistols | UMP45, Tommy Gun, P1911, R45, Deagle, Win94 |
| 9mm | SMGs, DMR, Pistols | Vector, Micro UZI, MP5K, PP-19, MP9, JS9, VSS, P92, P18C, Skorpion |
| .300 Magnum | SR | AWM only |
| .50 BMG | SR | Lynx AMR only |
| 5.7mm | SMG | P90 only |
| 12 Gauge | Shotguns | S686, S1897, S12K, DBS, Sawed-Off |
| Flare | Signal | Flare Gun only |
| Bolts | Crossbow | Crossbow only |

### Attachment Types

**Muzzles:**
- Compensator (reduces vertical + horizontal recoil significantly)
- Flash Hider (reduces muzzle flash, minor recoil reduction)
- Suppressor (reduces sound signature and muzzle flash, minor recoil reduction)
- Barrel Extender (extends effective damage range -- mainly shotguns)
- Choke (shotgun only -- tightens pellet spread)
- Duckbill (shotgun only -- widens horizontal spread, tightens vertical)

**Grips (AR/DMR/SMG):**
- Vertical Foregrip (reduces vertical recoil)
- Angled Foregrip (reduces horizontal recoil, faster ADS)
- Half Grip (reduces recoil + recovery, decreases stability)
- Light Grip (increases stability, faster recoil recovery)
- Thumb Grip (faster ADS, slight recoil reduction)
- Laser Sight (improves hip-fire accuracy -- replaces grip slot on some weapons)

**Scopes/Sights:**
- Red Dot Sight (1x)
- Holographic Sight (1x)
- Canted Sight (secondary sight, 1x, quick-switch)
- 2x Aimpoint Scope
- 3x Backpack Scope
- 4x ACOG Scope
- 6x Scope (variable 3x-6x)
- 8x CQBSS Scope (variable 4x-8x)
- 15x PM II Scope (crate only, variable 8x-15x)
- Thermal 4x Scope (Taego exclusive -- detects heat signatures through smoke)

**Magazines:**
- Quickdraw Magazine (faster reload)
- Extended Magazine (increased capacity)
- Extended Quickdraw Magazine (both -- faster reload + more capacity)

**Stocks:**
- Tactical Stock (M416, Vector -- reduces recoil, faster ADS)
- Cheek Pad (SRs, DMRs -- reduces recoil, faster scope-in)
- Folded Stock (UZI -- reduces recoil)
- Stock for Mk47 / Tommy Gun (reduces recoil, faster ADS)

---

## 6. Loot System

### World Loot Spawning
- Loot spawns at **predetermined spawn points** inside buildings, on rooftops, and near structures.
- Loot is generated at match start and is **static** (does not respawn after pickup).
- Every building/room has spawn points with randomized loot tables.
- Open fields generally have no loot; loot is concentrated in buildings and structures.

### Loot Tiers by Location

| Tier | Locations (Erangel examples) | Typical Loot |
|------|------------------------------|-------------|
| High | Military Base, Mylta Power, Novorepnoye, School, Georgopol Crates | Level 2-3 gear, ARs, DMRs, SRs, scopes, meds |
| Medium | Pochinki, Yasnaya Polyana, Rozhok, Primorsk | Level 1-2 gear, ARs, SMGs, shotguns, some scopes |
| Low | Small compounds, isolated buildings, farms | Pistols, shotguns, Level 1 gear, bandages |

### Care Package (Air Drops)
- A **C-130 plane** flies over the map at random intervals and drops a care package.
- The crate descends on a **single large parachute** with a visible flashing white light.
- On landing, a **red smoke flare** activates for ~2 minutes, visible from a distance.
- Care packages typically begin dropping after the **first circle** starts to close.
- **Drop frequency**: approximately every 3-5 minutes after the first few circles. Multiple drops can occur per match (typically 5-10 total).

**Care Package Exclusive Weapons:**
- AWM (.300 Magnum sniper)
- Groza (7.62 AR) -- now also world spawn on some maps
- Mk14 EBR (7.62 DMR, full-auto capable)
- AUG A3 (5.56 AR)
- M249 (5.56 LMG) -- now also world spawn on some maps
- MG3 (7.62 LMG)
- DBS (12 gauge shotgun)
- P90 (5.7mm SMG)
- Lynx AMR (.50 BMG anti-materiel rifle)

**Care Package Contents (typical):**
- 1 crate-exclusive weapon + matching ammo
- Level 3 armor piece (helmet or vest, sometimes both)
- Sometimes a Level 3 backpack
- Consumables (med kits, adrenaline syringe)
- Sometimes a ghillie suit

### Flare Gun Drops
- Flare gun found as rare world spawn.
- **Fired inside safe zone**: calls a **super air drop** (3 parachutes, larger crate, contains ~2x normal care package loot).
- **Fired outside safe zone**: calls a **BRDM-2** amphibious armored vehicle.
- Only works after first blue zone phase begins.
- Maximum active flare drops per match is limited (typically 3-4 max).

---

## 7. Vehicle System

### Land Vehicles

| Vehicle | Seats | Top Speed (km/h) | Fuel Capacity (L) | HP | Maps | Notes |
|---------|-------|-------------------|-------------------|-----|------|-------|
| UAZ (Open Top) | 4 | 116 | 95 | ~1820 | Erangel, Miramar | 4WD, good off-road |
| UAZ (Closed Top) | 4 | 116 | 95 | ~1820 | Erangel | Same stats, enclosed cabin |
| Dacia 1300 | 4 | 133 | 95 | ~1820 | Erangel | Fast on roads |
| Buggy | 2 | 100 | 95 | ~1500 | Erangel, Miramar | Light, nimble, no cover |
| Motorcycle | 2 | 152 | 50 | ~1000 | Erangel, Miramar, Sanhok | Fastest vehicle. Can flip. |
| Motorcycle w/ Sidecar | 3 | 130 | 50 | ~1000 | Erangel | 3rd seat in sidecar |
| Mirado | 4 | 152 | 95 | ~1820 | Miramar | Muscle car. Very fast on roads. |
| Pickup Truck | 4 | 110 | 95 | ~1820 | Miramar | Open bed = exposed passengers |
| Rony (Pickup) | 4 | 110 | 95 | ~1500 | Sanhok | Sanhok equivalent of pickup |
| Tukshai | 3 | 75 | 60 | ~1000 | Sanhok | Three-wheeler. Slow. |
| Scooter | 2 | 90 | 50 | ~800 | Sanhok | Light, slow |
| Snowmobile | 2 | 110 | 50 | ~1000 | Vikendi | For snow terrain |
| Zima | 4 | 100 | 95 | ~1500 | Vikendi | Vikendi-specific car |
| Pony Coupe | 4 | 130 | 95 | ~1500 | Taego | AWD coupe |
| Coupe RB | 2 | 150 | 95 | ~1500 | Miramar | Sports car. Very fast. |
| Mountain Bike | 1 | ~40 | N/A (no fuel) | N/A | Various | Silent. No fuel needed. Cannot be destroyed. |
| BRDM-2 | 4 | 100 (land), 22 (water) | 95 | ~2500 | All (flare drop) | Armored. Amphibious. Bullet resistant. |
| Pillar Car | 4 | 130 | 95 | ~1820 | Deston | |

### Water Vehicles

| Vehicle | Seats | Top Speed (km/h) | Notes |
|---------|-------|-------------------|-------|
| PG-117 (Boat) | 5 | 90 | Large boat. No cover. |
| Aquarail (Jet Ski) | 2 | 90 | More maneuverable than boat |
| Airboat | 5 | ~70 | Deston exclusive, can traverse shallow water and land |

### Vehicle Damage Mechanics
- Vehicles take damage from: bullets, explosions, collisions, falling, blue zone, red zone.
- **Running over a player**: instant kill (at sufficient speed, roughly >50 km/h).
- **Player-to-vehicle collision**: player takes damage based on vehicle speed. At 70+ km/h, typically lethal.
- **Vehicle explosion**: when vehicle HP reaches 0, it explodes after a short delay (~5 seconds with fire), dealing lethal damage to all occupants and nearby players (~100m lethal radius is very small, ~3-5m).
- **Tire damage**: vehicles (except BRDM) have shootable tires. Losing tires reduces speed and handling. Motorcycles become unrideable with a flat tire.
- **Fuel**: vehicles consume fuel while driving. Running out stops the vehicle. Gas cans (refill ~100 fuel units, ~60% of tank) can be found as world loot.

---

## 8. Zone/Circle Mechanics

### Erangel (8x8 km) -- Standard Battle Royale Zone Phases

The play zone ("white circle") shrinks over multiple phases. The "blue zone" is the area outside the safe zone that deals damage.

| Phase | Wait Time (s) | Shrink Time (s) | Blue Zone Damage/sec | Circle Radius (approx) | Notes |
|-------|---------------|-----------------|---------------------|----------------------|-------|
| Pre-game | 120 | -- | 0 | Full map | First circle appears on map 2 min after plane |
| Phase 1 | 300 (5 min) | 300 (5 min) | 0.4% / sec | ~4.5 km diameter | Very survivable outside |
| Phase 2 | 200 (3:20) | 180 (3 min) | 0.6% / sec | ~3.0 km diameter | Still easy to outheal |
| Phase 3 | 150 (2:30) | 150 (2:30) | 0.8% / sec | ~1.8 km diameter | Getting dangerous |
| Phase 4 | 120 (2 min) | 120 (2 min) | 1.0% / sec | ~1.0 km diameter | Hard to survive outside |
| Phase 5 | 90 (1:30) | 90 (1:30) | 3.0% / sec | ~0.5 km diameter | Cannot outheal |
| Phase 6 | 60 (1 min) | 60 (1 min) | 5.0% / sec | ~250m diameter | Rapidly lethal |
| Phase 7 | 45 (45s) | 45 (45s) | 7.0% / sec | ~100m diameter | Near instant death |
| Phase 8 | 30 (30s) | 30 (30s) | 11.0% / sec | ~50m diameter | |
| Phase 9 (Final) | 180 (3 min) | 15 | 11.0% / sec | 0 (closes completely) | Everyone outside dies |

**Note**: These values have been adjusted multiple times through patches. The above represents approximate current values for standard 8x8 maps. Smaller maps (4x4, 2x2) have faster timings and higher early-zone damage.

### Zone Movement Rules
- The **center of each new circle** is randomly placed within the current white zone, biased slightly toward areas with terrain features.
- The circle NEVER moves to be entirely at the edge -- it always overlaps with the current zone.
- Circle center shift distance decreases as circles get smaller.
- Blue zone moves at a constant speed per phase (calculated from shrink time and distance to travel).

### Sanhok (4x4 km) -- Faster Timing
- Faster circles overall. Phase 1 wait ~120s, Phase 1 shrink ~120s.
- Higher early-zone damage compared to 8x8 maps.
- Approximately 8 phases total.

### Red Zone Mechanics
- **Red zone** is a randomly placed circular area (~300m diameter) marked on the map in red.
- Bombs are dropped within the red zone for approximately **30 seconds**.
- **Damage**: Direct bomb hit is an **instant kill** (100+ damage).
- Bombs fall at random positions within the zone -- getting hit is somewhat unlikely.
- **Protection**: Being inside any building with a roof provides complete protection. Being under solid structures (arches, bridges) also protects.
- **Frequency**: Red zones appear randomly every ~2-5 minutes throughout the match, starting after the first circle.
- Red zones only appear inside the current white (safe) zone.
- **Not present** in ranked mode.
- Red zone primarily serves as an audio distraction (loud explosions mask footsteps and vehicle sounds).

---

## 9. Kill/Knock Mechanics

### DBNO (Down But Not Out) -- Team Modes Only
- In **duo** and **squad** modes, when a player's HP reaches 0 and at least one teammate is alive (not knocked), the player enters **DBNO state** instead of dying.
- In **solo** mode, reaching 0 HP = instant death. No DBNO.

### DBNO State Details
- **DBNO HP**: knocked player has a separate DBNO health pool (starts at 100 DBNO HP).
- **Bleed-out rate**: DBNO HP drains over time. Bleed-out time decreases with each successive knock in a match:
  - 1st knock: **~90 seconds** bleed-out
  - 2nd knock: **~33 seconds**
  - 3rd knock: **~20 seconds**
  - 4th knock: **~14 seconds**
  - 5th knock: **~11 seconds**
  - 6th knock: **~9 seconds**
  - 7th+ knock: **instant death** (0 seconds)
- Knocked players can **crawl** slowly but cannot use weapons or items.
- Knocked players can be **carried** by teammates (picks them up and moves faster than crawling, but carrier cannot shoot).
- **DBNO in water**: drowning + bleed-out stack. Death within ~10 seconds in water while DBNO.

### Reviving
- Revive time: **10 seconds** (standard). Some sources note it was changed to ~10s from earlier values. Patch 15.2 introduced a 3-second "quick revive" for emergency situations (limited uses).
- Reviving restores the knocked player to **~30 HP** (very low -- needs immediate healing).
- Reviving can be interrupted by taking damage or releasing the revive key.
- Only one person can revive a knocked teammate at a time.

### Instant Death Conditions (No DBNO)
- All teammates are already knocked or dead.
- Killed by red zone bomb.
- Vehicle explosion.
- Fall damage from extreme height.
- Already been knocked too many times in the match (7th+ knock).
- Blue zone kills a DBNO player.

### Kill Credit
- **Kill credit** goes to the player who dealt the most recent damage to the victim (the "finisher").
- If a knocked player bleeds out, kill credit goes to the player who knocked them.
- If a knocked player is finished by a different player, that finisher gets the kill.
- **Assist credit** goes to other players who dealt damage to the victim within a recent time window (~30-60 seconds).

---

## 10. Scoring/Ranking

### Placement Scoring (Casual/Normal Mode)
- Placement is based on survival order. Last player/team standing wins.
- XP and BP (battle points) rewards scale with placement:
  - Win (1st place): highest rewards
  - Top 10: significant bonus
  - Top 25: moderate bonus
  - Early elimination: minimal rewards
- Kill XP/BP: each kill grants additional XP and BP.

### Ranked Mode Scoring
Ranked mode uses **Rank Points (RP)** to determine tier/division.

**Rank Tiers (ascending):**
1. Bronze (5 divisions: V through I)
2. Silver (5 divisions)
3. Gold (5 divisions)
4. Platinum (5 divisions)
5. Diamond (5 divisions)
6. Master (no divisions -- leaderboard based)

**RP Calculation Formula (simplified):**
- RP = Placement Points + Kill Points - Entry Cost
- **Placement points** increase significantly for higher placements (top 10 is where major gains happen).
- **Kill points**: each kill = a flat amount of RP (varies by rank tier).
- **Entry cost**: increases at higher ranks, so you must perform well to gain RP.
- Top 3 placement: large RP boost.
- Win: maximum placement bonus.
- Early death with 0 kills at high ranks: lose RP.

### Key Stats Tracked
- Wins / Win Rate (%)
- Kills / K/D Ratio
- Average Damage
- Average Survival Time
- Top 10 Rate
- Headshot Kill %
- Longest Kill Distance

---

## 11. Realistic Match Timeline

### Typical 30-Minute Squad Match on Erangel (8x8 km)

| Time | Phase | Players Alive (approx) | Activity |
|------|-------|----------------------|----------|
| 0:00 | Lobby/Plane | 100 | Plane begins crossing map |
| 0:00-1:00 | Jumping/Parachuting | 100 | Players jump from plane, free fall, deploy chutes |
| 1:00-2:00 | Landing/Early Loot | ~95 | Players land and begin looting nearest buildings |
| 2:00 | First circle appears | ~90 | Blue zone boundary shown on map. 5 min wait. |
| 2:00-5:00 | **Early Game (Looting Phase)** | 90 -> 70 | Hot drop fights resolve. ~20-30 players die in first 3 min. Survivors loot aggressively. |
| 5:00-7:00 | Phase 1 blue zone closes | ~65 | Players at edges begin rotating. Vehicle fights. Some caught by zone. |
| 7:00-10:00 | **Mid-Game Transition** | 65 -> 50 | Phase 2 wait. Players consolidate positions. Compound-holding begins. |
| 10:00-13:00 | Phase 2 closes | ~45 | More rotation fights. Bridge camps on Erangel. Care packages contested. |
| 13:00-16:00 | Phase 3 | ~35 | Circle getting tight. Contact increases. Teams start engaging more. |
| 16:00-19:00 | **Mid-Late Game** | 35 -> 25 | Phase 4. Teams jockeying for position. Smoke grenades used heavily. |
| 19:00-22:00 | Phase 5 | ~20 | Zone is very small. Nearly every team in contact. Heavy fighting. |
| 22:00-25:00 | **Late Game** | 20 -> 10 | Phase 6-7. Intense firefights. Teams getting wiped. |
| 25:00-28:00 | Phase 7-8 | ~5-8 | Final circles. Everyone fighting. Grenades, smokes, aggression. |
| 28:00-32:00 | **Endgame** | 5 -> 1 | Final circle closes to nothing. Last team standing wins. |

### Kill Distribution by Phase
- **~30-35% of all kills** occur in the first 5 minutes (hot drop chaos).
- **~15-20%** during mid-game rotations (phases 2-3).
- **~25-30%** during late-game phases (phases 4-6).
- **~15-20%** in the final circles (phases 7-9).

### Typical Player Count Milestones
- 1 minute in: ~95 alive (a few die on landing or instant hot-drop kills)
- 3 minutes in: ~70 alive (hot drops resolved)
- 5 minutes in: ~60-65 alive (first zone closing)
- 10 minutes in: ~45-50 alive
- 15 minutes in: ~30-35 alive
- 20 minutes in: ~20 alive
- 25 minutes in: ~8-10 alive
- 30 minutes in: ~2-4 alive (final fight)

---

## 12. Typical Stats (Player Performance Benchmarks)

### K/D Ratio (Kills per Death)

| Tier | K/D | Description |
|------|-----|-------------|
| Bottom 25% | 0.2-0.5 | New/casual players |
| Below Average | 0.5-0.8 | Learning players |
| Average | 0.8-1.2 | Median PUBG player |
| Above Average | 1.2-2.0 | Solid player |
| Good | 2.0-3.5 | Strong player |
| Excellent | 3.5-6.0 | Top-tier player |
| Pro/Streamer | 6.0+ | Competitive/professional level |

Note: Mathematically, the average K/D across all players is slightly below 1.0 (since some kills go to the zone/suicide). Bots inflate K/D in casual matches.

### Average Damage Per Match

| Tier | Damage | Description |
|------|--------|-------------|
| Bottom 25% | 50-100 | Few shots landed |
| Below Average | 100-200 | Limited engagements |
| Average | 200-350 | Typical casual player |
| Above Average | 350-500 | Regular fighters |
| Good | 500-800 | Aggressive, skilled |
| Excellent | 800+ | Hot droppers, high-kill games |

### Average Survival Time

| Tier | Time | Description |
|------|------|-------------|
| Very Short | 3-5 min | Dies on hot drop |
| Short | 5-10 min | Early game death |
| Average | 12-18 min | Typical game |
| Long | 18-25 min | Makes it to late game |
| Winner-tier | 25-32 min | Frequently reaches endgame |

### Win Rate

| Tier | Win % | Description |
|------|-------|-------------|
| Expected Average | 1% (solo), 4% (squad) | Mathematically expected random win rate |
| Typical Average | 2-5% | Most players |
| Good | 5-10% | Consistent player |
| Very Good | 10-20% | Strong player |
| Exceptional | 20%+ | Pro/competitive |

### Other Useful Stats
- **Average kills per match** (all players): ~0.9 (since 99 players die per solo match, 100 players / 99 deaths = ~0.99 max average)
- **Average kills for winners**: ~5-8 kills in solo, ~8-15 team total in squads
- **Headshot percentage**: average ~15-20%, good players ~25-35%
- **Longest common kill distances**: most kills under 100m, long-range (300m+) is uncommon

---

## Appendix: Key Constants for Simulator

```
MAX_PLAYERS = 100
MAX_PLAYERS_RANKED = 64
MAX_HP = 100
MAX_BOOST = 100

# Healing caps
BANDAGE_HEAL = 10
BANDAGE_CAP = 75
FIRST_AID_CAP = 75
MED_KIT_CAP = 100

# Cast times (seconds)
BANDAGE_CAST = 4
FIRST_AID_CAST = 6
MED_KIT_CAST = 8
ENERGY_DRINK_CAST = 4
PAINKILLER_CAST = 7.5
ADRENALINE_CAST = 10

# Armor damage reduction
ARMOR_REDUCTION = {1: 0.30, 2: 0.40, 3: 0.55}
HELMET_DURABILITY = {1: 80, 2: 150, 3: 230}
VEST_DURABILITY = {1: 200, 2: 220, 3: 250}

# Headshot multiplier (most weapons)
HEADSHOT_MULTIPLIER = 2.3

# Limb damage multiplier
LIMB_MULTIPLIER = 0.5  # arms
LEG_MULTIPLIER = 0.75  # legs (some weapons vary)

# Zone damage per second by phase (Erangel 8x8)
ZONE_DAMAGE = {
    1: 0.4,
    2: 0.6,
    3: 0.8,
    4: 1.0,
    5: 3.0,
    6: 5.0,
    7: 7.0,
    8: 11.0,
    9: 11.0,
}

# Zone timing (seconds) [wait, shrink]
ZONE_TIMING = {
    0: [120, 0],     # pre-game (circle appears)
    1: [300, 300],
    2: [200, 180],
    3: [150, 150],
    4: [120, 120],
    5: [90, 90],
    6: [60, 60],
    7: [45, 45],
    8: [30, 30],
    9: [180, 15],    # final circle closes to nothing
}

# DBNO bleed-out times (seconds) by knock count
DBNO_BLEEDOUT = {1: 90, 2: 33, 3: 20, 4: 14, 5: 11, 6: 9, 7: 0}
REVIVE_TIME = 10  # seconds

# Vehicle speeds (km/h)
VEHICLE_SPEED = {
    "motorcycle": 152,
    "mirado": 152,
    "coupe_rb": 150,
    "dacia": 133,
    "motorcycle_sidecar": 130,
    "pony_coupe": 130,
    "uaz": 116,
    "pickup": 110,
    "snowmobile": 110,
    "buggy": 100,
    "brdm_land": 100,
    "zima": 100,
    "scooter": 90,
    "boat": 90,
    "aquarail": 90,
    "tukshai": 75,
    "mountain_bike": 40,
    "brdm_water": 22,
}

# Parachute speeds (km/h)
FREEFALL_NEUTRAL = 180
FREEFALL_DIVE = 234
FREEFALL_GLIDE_VERTICAL = 126
FREEFALL_GLIDE_LATERAL = 122
CHUTE_NEUTRAL = 54
CHUTE_FORWARD_VERTICAL = 64
CHUTE_FORWARD_LATERAL = 47
AUTO_DEPLOY_ALTITUDE_AGL = 300  # meters

# Match duration (typical)
MATCH_DURATION_MINUTES = 30  # approximate
PLANE_CROSS_TIME = 50  # seconds, approximate for 8x8
```

---

## Sources and Caveats

- PUBG Corp (Krafton) does not publish official weapon damage values. All damage numbers are community-sourced from datamining (battlegrounds.party), extensive testing (WackyJacky101), and wiki aggregation.
- Values may shift slightly with patches. The numbers above represent the best community consensus as of early 2025.
- PUBG Mobile has different values from PUBG PC/Console for many mechanics (damage, zone timing, etc.). This document covers **PC/Console** values.
- Zone timing and damage have been adjusted many times since launch. The values here are approximate and represent the general current state.

### Key Reference Sources
- PUBG Fandom Wiki (pubg.fandom.com)
- PUBG Wiki (pubg.wiki.gg)
- battlegrounds.party (datamined stats)
- WackyJacky101 testing (wackyjacky101.com)
- Steam Community weapon guides
- Official PUBG game info (pubg.com/en/game-info)
