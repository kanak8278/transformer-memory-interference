# Fortnite Battle Royale -- Complete Game Mechanics Reference

> Compiled for match-state simulator development.
> Values reflect the **standard (non-competitive) Battle Royale** rule set as of
> Chapter 6 / early 2026. Fortnite patches weapon and system values frequently;
> numbers marked with (*) are long-standing defaults that may shift between
> seasons. Where a value has changed across patches, the most commonly referenced
> "canonical" value is given.

---

## 1. Match Overview

| Parameter | Value |
|---|---|
| Players per match (Solo) | 100 |
| Players per match (Duos) | 100 (50 teams of 2) |
| Players per match (Trios) | 100 (~33 teams of 3) |
| Players per match (Squads) | 100 (25 teams of 4) |
| Typical match duration (full game) | 20--25 minutes |
| Typical match duration (early elimination) | 2--5 minutes |
| Pre-game lobby + Battle Bus transit | ~60 seconds |
| Average K/D ratio (population) | 1.00 |
| Median eliminations per match (casual player) | 1--3 |
| Good eliminations per match | 5--8 |
| High-skill / content-creator range | 10--20+ |

---

## 2. Health and Shield System

### 2.1 Base Values

| Stat | Value |
|---|---|
| Maximum Health (HP) | 100 |
| Maximum Shield | 100 |
| Total effective HP (Health + Shield) | 200 |
| Overshield (when active, e.g. Chapter 4+) | +50 on top of 200, for 250 total |
| Overshield regeneration delay | ~5 seconds after last damage taken |
| Overshield regen rate | ~1.5/s (slow passive regen) |

Damage is applied to **Overshield first**, then **Shield**, then **Health**.

### 2.2 Healing Items

| Item | Heals | Target | Use Time | Max Stack | Cap |
|---|---|---|---|---|---|
| Bandage | 15 HP | Health only | 4 s | 15 | Heals up to 75 HP only |
| Medkit | Full HP (to 100) | Health only | 10 s (*some patches 9s) | 3 | -- |
| Mini Shield Potion | 25 Shield | Shield only | 2 s | 10 | Only works up to 50 Shield |
| Shield Potion (large/"big pot") | 50 Shield | Shield only | 5 s | 3 | -- |
| Slurp Juice | 75 total (1 HP + 1 Shield /s for ~37.5s) | Both | 2 s | 2 | Heals HP first, then Shield |
| Chug Jug | Full HP + Full Shield (200 total) | Both | 15 s | 1 | -- |
| Chug Splash | 20 HP or Shield | Both (splash, AoE) | Instant (thrown) | 6 | Heals lowest bar first |
| Bandage Bazooka | 15 HP per shot | Health (ranged heal) | ~1 s per shot | 1 (takes 2 slots) | Unlimited ammo, cooldown-based |
| Campfire (placed) | 2 HP/s for 25 s (50 total) | Health | Instant placement | -- | Proximity-based, can be stoked |
| Flopper (fish) | 40 HP | Health | 1 s | 4 | -- |
| Shield Fish / Slurpfish | 40 Shield / 40 effective | Shield / Both | 1 s | 4 | -- |

*Note: The loot pool rotates per season. Not all items are available simultaneously. The above covers the "classic" item set that recurs most often.*

### 2.3 Down-But-Not-Out (DBNO) -- Team Modes

| Parameter | Value |
|---|---|
| DBNO Health (when knocked) | 100 HP, draining at ~2 HP/s |
| DBNO bleedout time | ~50 seconds |
| Revive time (teammate) | 10 seconds (*reduced in some seasons) |
| Revive health restored | 30 HP (no shield) |
| Successive DBNO bleedout speed | Faster each time knocked in same match |

---

## 3. Reboot System (Team Modes)

| Parameter | Value |
|---|---|
| Reboot Card drop | On full elimination (after DBNO bleedout or squad wipe) |
| Reboot Card expiry timer | 90 seconds (original); **270 seconds** (4 min 30s, Chapter 5+) |
| Reboot Van interaction time | 10 seconds |
| Reboot Van cooldown after use | ~120 seconds (reduced in some patches to ~60s) |
| Respawn loadout | Common pistol + 100 wood/stone/metal each |
| Respawn HP | 100 HP, 0 Shield |

---

## 4. Material System

### 4.1 Material Types and Caps

| Material | Color | Max Stack (Standard) | Max Stack (Competitive/Arena) |
|---|---|---|---|
| Wood | Brown | 999 | 500 |
| Stone (Brick) | Grey | 999 | 500 |
| Metal | Blue/Silver | 999 | 500 |

### 4.2 Harvesting (Farming)

**Each swing of the Harvesting Tool (Pickaxe):**
- Base harvest: variable per object (~5--15 materials per swing)
- **Weak-point (blue circle) hit**: 2x materials AND 2x damage to object
- Harvesting Tool damage to players: 20 per swing

**Approximate yields from common objects:**

| Object | Material | Approx. Total Yield | Approx. Swings to Destroy |
|---|---|---|---|
| Small tree (skinny) | Wood | 20--30 | 2--3 |
| Medium tree | Wood | 40--60 | 4--6 |
| Large tree (tall pine, palm) | Wood | 50--70 | 5--7 |
| Wooden pallet | Wood | 40--60 | 2--3 |
| Wooden furniture (chairs, tables) | Wood | 10--20 | 1--2 |
| Small rock | Stone | 8--15 | 2--3 |
| Medium boulder | Stone | 30--50 | 4--5 |
| Large rock formation | Stone | 40--60 | 5--7 |
| Brick wall (building) | Stone | 15--30 | 2--4 |
| Small metal object (fence, sign) | Metal | 5--15 | 1--2 |
| Car (sedan) | Metal | 16--25 | 3--5 |
| Truck / large vehicle | Metal | 25--40 | 5--7 |
| Metal shipping container | Metal | 30--50 | 5--7 |
| RV / trailer | Metal + Wood mix | 20--35 | 4--6 |

### 4.3 Build Cost

**Every building piece costs exactly 10 materials**, regardless of structure type or material type.

---

## 5. Building System

### 5.1 Structure Types

There are **four** buildable structure types:
1. **Wall** -- vertical barrier, 1x1 tile
2. **Floor** -- horizontal platform, 1x1 tile
3. **Ramp (Stairs)** -- angled surface, 1x1 tile
4. **Pyramid (Cone)** -- peaked roof structure, 1x1 tile

Each can be built from Wood, Stone, or Metal. Each costs **10** of the chosen material.

### 5.2 Structure Health Points

| Structure | Material | Starting HP (on placement) | Max HP (fully built) | Build Time to Max HP |
|---|---|---|---|---|
| **Wall** | Wood | 90 | 150 | ~4 s |
| **Wall** | Stone | 100 | 300 | ~12 s |
| **Wall** | Metal | 110 | 500 | ~25 s |
| **Ramp** | Wood | 84 | 140 | ~4 s |
| **Ramp** | Stone | 93 | 280 | ~12 s |
| **Ramp** | Metal | 101 | 460 | ~25 s |
| **Floor** | Wood | 84 | 140 | ~4 s |
| **Floor** | Stone | 93 | 280 | ~12 s |
| **Floor** | Metal | 101 | 460 | ~25 s |
| **Pyramid** | Wood | 84 | 140 | ~4 s |
| **Pyramid** | Stone | 93 | 280 | ~12 s |
| **Pyramid** | Metal | 101 | 460 | ~25 s |

**Key observations for simulator:**
- Structures spawn at roughly **60%** of max HP for wood, **33%** for stone, **22%** for metal.
- Wood builds fastest to max (best for quick fights), metal is strongest when fully built (best for turtling).
- HP builds linearly from starting to max over the build time.
- Editing a structure resets it to starting HP.

### 5.3 Editing

- Players can edit their OWN structures (not enemy builds).
- Editing is instant (edit -> confirm).
- Common edits: door in wall, half-wall, ramp rotation, window, arch.
- Editing temporarily opens the structure (vulnerability window).

---

## 6. Weapon System

### 6.1 Weapon Rarities

| Rarity | Color | Drop Likelihood | Damage Bonus vs Common |
|---|---|---|---|
| Common | Grey | Very High (floor loot) | Baseline |
| Uncommon | Green | High | ~+3--5% |
| Rare | Blue | Medium | ~+7--10% |
| Epic | Purple | Low | ~+12--17% |
| Legendary | Gold | Very Low (chests, supply drops) | ~+17--23% |
| Mythic | Gold (pulsing) | Unique (boss drops, 1 per match) | ~+25--30% |
| Exotic | Cyan/Teal | Purchasable from NPCs | Varies (unique effects) |

Each rarity tier generally adds **1--3 base damage**, improved reload speed, and sometimes larger magazine size compared to the tier below.

### 6.2 Weapon Categories and Representative Stats

*Values below are "canonical" / most-commonly-referenced across seasons. Actual per-season loot pools vary.*

#### Assault Rifles (AR)

| Weapon | Rarity Range | Body Damage | Headshot Mult. | Fire Rate (rounds/s) | Magazine | DPS (body) | Reload (s) |
|---|---|---|---|---|---|---|---|
| Standard AR (M16-type) | Common--Rare | 30 / 31 / 33 | 2.0x | 5.5 | 30 | 165--181 | 2.3--2.1 |
| SCAR (Assault Rifle) | Epic--Legendary | 33 / 35 | 2.0x | 5.5 | 30 | 181--192 | 2.1--2.0 |
| Heavy AR (AK) | Common--Legendary | 33 / 35 / 37 / 39 / 41 | 2.0x | 3.75 | 25 | 124--154 | 2.9--2.4 |
| Burst AR (FAMAS) | Common--Legendary | 27 / 29 / 30 / 32 / 33 | 2.0x | 1.75 (bursts) | 20--25 | ~141--173 | 2.9--2.3 |
| Tactical AR | Rare--Legendary | 22 / 23 / 24 | 1.75x | 7.0 | 30 | 154--168 | 2.4--2.2 |
| Mythic AR (boss variant) | Mythic | 36--37 | 2.0x | 5.5 | 30 | ~198--203 | 1.9 |

#### Shotguns

| Weapon | Rarity Range | Body Damage (all pellets) | Headshot Mult. | Fire Rate | Magazine | Pellets | Reload (s) |
|---|---|---|---|---|---|---|---|
| Pump Shotgun | Common--Legendary | 84 / 92 / 100 / 108 / 116 | 2.0x (*was 2.5x) | 0.7 | 5 | 10 | ~4.6--4.0 |
| Tactical Shotgun | Common--Legendary | 67 / 70 / 74 / 77 / 80 | 2.25x | 1.5 | 8 | 17 | ~6.3--5.7 |
| Heavy Shotgun | Epic--Legendary | 77 / 81 | 2.5x | ~1.0 | 7 | 10 | ~5.6 |
| Combat Shotgun | Rare--Legendary | 73 / 77 / 81 | 1.5x | 1.85 | 8 | -- | ~6.0--5.4 |
| Lever Action Shotgun | Common--Legendary | 93.6 / 97.2 / 100.8 / 104.4 / 108 | 1.75x | 1.05 | 6 | 12 | ~6.0 |
| Drum Shotgun | Common--Rare | 45 / 47 / 50 | 1.25x | 3.0 | 12 | 16 | ~5.0 |

**Shotgun notes:**
- Damage listed is MAX damage (all pellets hit) at close range.
- Damage falls off sharply with distance; effective range ~10--15 m.
- Pump can one-shot (headshot) with Epic+ rarity: 108 x 2.0 = 216 > 200 effective HP.
- Legendary Pump headshot: 116 x 2.0 = 232.

#### Submachine Guns (SMG)

| Weapon | Rarity Range | Body Damage | Headshot Mult. | Fire Rate | Magazine | DPS | Reload (s) |
|---|---|---|---|---|---|---|---|
| Submachine Gun (standard) | Common--Rare | 17 / 18 / 19 | 1.75x | 12.0 | 30 | 204--228 | 2.2--2.0 |
| Compact SMG (P90) | Epic--Legendary | 19 / 20 | 1.75x | 10.0 | 40 | 190--200 | 2.7--2.5 |
| Rapid Fire SMG | Common--Legendary | 13 / 14 / 14 / 15 / 15 | 1.75x | 15.0 | 20--26 | 195--225 | 1.3--1.1 |
| Tactical SMG | Rare--Legendary | 20 / 21 / 22 | 1.75x | 13.0 | 35 | 260--286 | 2.2--2.0 |
| Drum Gun | Uncommon--Rare | 22 / 23 | 1.5x | 9.0 | 50 | 198--207 | 3.3--3.0 |
| Suppressed SMG | Common--Epic | 18 / 19 / 20 / 21 | 2.0x | 9.0 | 30 | 162--189 | 2.2--1.9 |

#### Sniper Rifles

| Weapon | Rarity Range | Body Damage | Headshot Mult. | Fire Rate | Magazine | Reload (s) |
|---|---|---|---|---|---|---|
| Bolt-Action Sniper | Rare--Legendary | 105 / 110 / 116 | 2.5x | 0.33 | 1 | 2.5--2.0 |
| Heavy Sniper | Epic--Legendary | 132 / 138 | 2.5x | 0.33 | 1 | 4.2--3.9 |
| Hunting Rifle | Uncommon--Legendary | 75 / 80 / 85 / 89 / 94 | 2.5x | 1.0 | 1 | 1.5--1.1 |
| Semi-Auto Sniper | Uncommon--Epic | 72 / 75 / 78 | 2.5x | 1.65 | 10 | 2.8--2.4 |
| Storm Scout Sniper | Epic--Legendary | 81 / 85 | 2.0x | 1.33 | 6 | 2.9--2.7 |

**Sniper notes:**
- Bolt-Action / Heavy: single-shot, must chamber between shots.
- Legendary Bolt headshot: 116 x 2.5 = 290 (one-shot from any HP).
- Heavy Sniper body shot can break any wall in one shot (132/138 > all structure max HP except metal).
- Bullet drop and travel time apply to all snipers.

#### Pistols

| Weapon | Rarity Range | Body Damage | Headshot Mult. | Fire Rate | Magazine | Reload (s) |
|---|---|---|---|---|---|---|
| Pistol (standard) | Common--Rare | 24 / 25 / 26 | 2.0x | 6.75 | 16 | 1.5--1.3 |
| Revolver | Common--Rare | 54 / 57 / 60 | 2.0x | 1.0 | 6 | 2.4--2.2 |
| Hand Cannon (Deagle) | Epic--Legendary | 75 / 78 | 2.0x | 0.8 | 7 | 2.2--2.0 |
| Suppressed Pistol | Epic--Legendary | 28 / 30 | 2.0x | 6.75 | 16 | 1.5--1.3 |
| Dual Pistols | Epic--Legendary | 41 / 43 (x2 per burst) | 1.65x | 3.96 | 18 | 2.8--2.7 |
| Flint-Knock Pistol | Common--Uncommon | 86 / 90 | 2.0x | 0.33 | 1 | 3.2--3.0 |

#### Explosives / Launchers

| Weapon | Rarity Range | Damage | Radius | Fire Rate | Magazine | Structure Dmg |
|---|---|---|---|---|---|---|
| Rocket Launcher | Rare--Legendary | 110 / 116 / 121 | ~3 tiles | 0.75 | 1 | 1 hit destroys any build |
| Grenade Launcher | Rare--Legendary | 100 / 105 / 110 | ~2.5 tiles | 1.0 | 6 | High |
| Grenade (throwable) | Common | 100 | ~2 tiles | ~1 thrown/s | 10 (stack) | High |
| Firefly Jar | Common | 40 (+ burn 10/s for 5s) | ~1 tile | thrown | 6 | Sets builds on fire |
| Clingers (sticky) | Rare | 100 | ~1 tile | thrown | 3 | High |

#### Other / Utility Weapons

| Weapon | Rarity | Notes |
|---|---|---|
| Minigun | Epic--Legendary | 19--21 body dmg, unlimited ammo, fires 12/s, overheats after ~6s continuous |
| Harpoon Gun | Rare | 75 damage, pulls items/fish/players, 10 charges |
| Boom Bow | Legendary | 100 (splash) + 15 (direct), charged shot |
| Rail Gun | Epic--Legendary | 85/88, charges before firing, penetrates walls |
| Shockwave Grenade | -- | No damage, launches players through builds |
| Boogie Bomb | -- | Forces dance for 5 seconds, any damage breaks effect |

### 6.3 Ammo Types

| Ammo Type | Used By | Max Stack |
|---|---|---|
| Light Ammo (small bullets) | Pistols, SMGs, some ARs | 999 |
| Medium Ammo | Assault Rifles, some LMGs | 999 |
| Heavy Ammo | Snipers, Hand Cannon, some ARs | 999 |
| Shells (shotgun ammo) | Shotguns | 999 |
| Rockets | Rocket/Grenade Launcher | 12 (*was higher, capped in competitive) |

---

## 7. Loot System

### 7.1 Loot Sources

| Source | Contents | Notes |
|---|---|---|
| Floor Loot | 1 weapon or item | Random spawn points in buildings/POIs |
| Chest | 1 weapon + ammo + materials + 1 consumable/item | Glowing golden chest, fixed spawn points |
| Rare Chest | Better rarity distribution | Blue/purple glow |
| Supply Drop (airdrop) | 2 weapons (Epic+) + mats + healing | Drops mid-match from sky, visible balloon |
| Ammo Box | Ammo (various types) | Small green boxes in buildings |
| Eliminated player loot | All their inventory | Drops on ground as loot pile |
| Fishing spots | Fish (healing), weapons (rare) | Requires fishing rod |
| NPC vendors | Purchasable items/weapons | Gold bars currency |

### 7.2 Inventory Slots

| Slot Type | Count |
|---|---|
| Weapon/Item slots | 5 |
| Material slots | 3 (wood, stone, metal -- separate, not in weapon slots) |
| Trap slot (when traps are in game) | 1 (separate) |

Players can carry a maximum of **5 weapons/items** plus all 3 material types plus a trap.

---

## 8. Storm / Zone Mechanics

### 8.1 Storm Overview

The storm is a damage-dealing zone that shrinks the playable area over time, forcing player encounters. The playable area is called the **Safe Zone** (or "circle").

### 8.2 Storm Phase Table (Standard Battle Royale -- 9 Core Phases)

| Phase | Wait Time (before shrink) | Shrink Duration | Damage/second | Safe Zone Size (approx. % of map) |
|---|---|---|---|---|
| 1 (Pre-storm) | ~180 s (3 min) | ~150 s (2.5 min) | -- | 100% (no storm yet) |
| 2 (1st circle) | ~180 s (3 min) | ~150 s (2.5 min) | 1 /s | ~60--65% |
| 3 (2nd circle) | ~150 s (2.5 min) | ~120 s (2 min) | 1 /s | ~35--40% |
| 4 (3rd circle) | ~120 s (2 min) | ~90 s (1.5 min) | 2 /s | ~20--25% |
| 5 (4th circle) | ~60 s (1 min) | ~60 s (1 min) | 5 /s | ~10--12% |
| 6 (5th circle) | ~30 s | ~45 s | 8 /s | ~5% |
| 7 (6th circle) | ~0 s (immediate) | ~30 s | 10 /s | ~2% |
| 8 (7th circle) | ~0 s (immediate) | ~20 s | 10 /s | ~0.5% |
| 9 (Final / full close) | ~0 s | ~15 s | 10 /s | 0% (storm covers entire map) |

**Notes:**
- Phases 7--9 have no wait time; the storm begins shrinking immediately upon reaching the current circle.
- After phase 9, the storm deals 10/s everywhere -- game must end.
- Some seasons add phases 10--12 with continued 10/s damage.
- **Storm Surge** (competitive): if too many players are alive past a threshold, players below a damage-dealt threshold take periodic damage (25 HP every 5 seconds).

### 8.3 Storm Sickness (Anti-camping, Chapter 5+)

After absorbing a cumulative threshold of storm damage (~100--600 depending on mode), players gain **Storm Sickness**, which adds an extra 2--4 damage per tick on top of normal storm damage, making extended storm camping unviable.

---

## 9. Vehicles

### 9.1 Vehicle Types (Recurring Across Seasons)

| Vehicle | Type | Seats | HP | Fuel | Has Boost | Special |
|---|---|---|---|---|---|---|
| Sedan (OG car) | Land | 4 | ~800 | Yes (100 fuel) | No | -- |
| Sports Car | Land | 2 | ~600 | Yes | Yes (charged boost) | Fastest on-road |
| Pickup Truck | Land | 2 (+ bed riders) | ~1000 | Yes | No | Can carry items in bed |
| Semi Truck (Big Rig) | Land | 2 | ~1500 | Yes | No | Highest HP, slow |
| Taxi | Land | 4 | ~800 | Yes | No | -- |
| Motorboat | Water | 4 | ~800 | Yes | Yes (boost) | Has mounted missile launcher |
| Choppa (Helicopter) | Air | 4 | ~1500 | No fuel (infinite) | No | Slow, high altitude |
| Biplane (X-4 Stormwing) | Air | 5 (1 pilot + 4 on wings) | ~800 | No fuel | Yes (boost) | Wing-mounted gun |
| Shopping Cart | Land | 2 (1 rides, 1 pushes) | -- (no HP, indestructible) | No | Downhill momentum | No engine |
| ATK (All Terrain Kart) | Land | 4 | ~400 | No fuel | Yes (drift boost) | Fun vehicle, jump |
| Quadcrasher | Land | 2 | ~600 | No fuel | Yes (dash) | Destroys builds on impact |
| Baller (hamster ball) | Land | 1 | ~300 | No fuel | Grapple swing | Immune to storm (was patched) |
| Dirt Bike | Land | 2 | ~600 | Yes | Yes | Tricks give boost charge |
| Titan Tank (rare event) | Land | 2 | ~2500 | Yes | No | Cannon weapon, very slow |

**Vehicle notes:**
- Vehicles take storm damage when in storm.
- Shooting from vehicles: passengers can shoot; drivers cannot (except mounted weapons).
- Fuel depletes over time; gas cans (100 fuel) can refuel.
- Collision with builds destroys builds (high-speed impact).
- Vehicles can be destroyed; occupants take 25--50 damage on destruction.

---

## 10. Elimination and Scoring

### 10.1 Elimination Types

| Type | Description |
|---|---|
| Weapon Elimination | Kill with any weapon |
| Pickaxe Elimination | Kill with harvesting tool (20 dmg/swing) |
| Storm Elimination | Player dies to storm damage |
| Fall Damage Elimination | Fatal fall (>3 story height without glider) |
| Explosion Elimination | Rockets, grenades, gas cans |
| Vehicle Elimination | Run over by vehicle or vehicle explosion |
| Trap Elimination | Damage trap (when in game: 150 dmg) |
| Environmental | Propane tanks, barrels, fire |

### 10.2 Fall Damage

| Fall Height | Damage |
|---|---|
| < 3 stories (~10 m) | 0 |
| 3 stories | ~10 |
| 4 stories | ~20 |
| 5 stories | ~40 |
| 6+ stories | ~60--100+ |
| Terminal velocity / max fall | 100 (lethal if no shield) |

Fall damage scales roughly with height. Glider redeploy (when available) negates fall damage.

### 10.3 XP and Placement

| Placement (Solos) | Typical XP Range |
|---|---|
| Victory Royale (#1) | High (varies by season) |
| Top 5 | Moderate-High |
| Top 10 | Moderate |
| Top 25 | Low-Moderate |
| Top 50 | Low |
| Eliminated early | Minimal |

Each elimination also grants XP. Assists grant partial XP.

---

## 11. Typical Match Timeline (Solos)

| Time (approx.) | Phase | Typical Events | Players Alive |
|---|---|---|---|
| 0:00--1:00 | Pre-game + Bus | Lobby, Battle Bus path across map | 100 |
| 1:00--1:30 | Drop Phase | Players jump, glide to POI | 100 |
| 1:30--3:00 | Early Game | Landing, first loot, initial fights at POIs | 80--90 |
| 3:00--5:00 | Early-Mid | First storm warning, looting continues | 60--75 |
| 5:00--8:00 | Mid Game (Storm 1 shrinks) | Rotation begins, storm pushes players | 40--55 |
| 8:00--12:00 | Mid Game (Storms 2--3) | Engagements increase, rotations to zone | 25--40 |
| 12:00--16:00 | Late-Mid (Storms 3--4) | Consistent fighting, build battles | 15--25 |
| 16:00--19:00 | Late Game (Storms 5--6) | Small zone, high intensity, 3rd partying | 5--15 |
| 19:00--22:00 | End Game (Storms 7--9) | Final fights, heal-offs if zone closes | 2--5 |
| 22:00--25:00 | Victory | Last player standing | 1 |

---

## 12. Miscellaneous Mechanics

### 12.1 Movement

| Action | Speed / Value |
|---|---|
| Walk speed | ~5.5 m/s |
| Sprint speed | ~7.35 m/s |
| Crouch walk | ~3.0 m/s |
| Slide speed (on slope) | ~8--10 m/s |
| Mantling (climb over obstacles) | ~1 story max height |
| Tactical Sprint (Chapter 4+) | ~8.5 m/s, stamina-limited |
| Swimming | ~3.5 m/s |
| Glider deploy (from height) | ~15--20 m/s descent |

### 12.2 Damage Modifiers

| Modifier | Multiplier | Notes |
|---|---|---|
| Body shot | 1.0x | Baseline |
| Headshot | Weapon-dependent (1.25x--2.5x) | See weapon tables above |
| Limb shot | 0.75x (some weapons) | Not all weapons differentiate |
| Structure damage | Varies | Some weapons do extra struct dmg |

### 12.3 Inventory / Interaction

| Action | Time |
|---|---|
| Swap weapons | ~0.2--0.5 s (weapon dependent) |
| Open chest | ~1 s |
| Open door | Instant |
| Reload (varies by weapon) | 1.0--4.5 s (see weapon tables) |
| Pick up item | Instant |
| Drop item | Instant |

### 12.4 Map and POIs

| Parameter | Typical Value |
|---|---|
| Map size | ~2.5 km x 2.5 km (varies by chapter) |
| Named POIs (Points of Interest) | 12--20 per season |
| Unnamed landmarks | 30--50+ |
| Chest spawn rate per building | ~50--100% (varies by mode) |

---

## 13. Key Formulas for Simulator

### Effective Time to Kill (TTK)

```
TTK = target_effective_HP / weapon_DPS
```

Where `target_effective_HP` = Shield + Health (+ Overshield if applicable).

Example: Full HP target (200 HP) vs. Legendary SCAR (192 DPS) = ~1.04 seconds (theoretical, all shots hit).

### Storm Survival Time

```
survival_seconds = current_HP / storm_damage_per_second
```

Example: 100 HP in phase 5 storm (5/s) = 20 seconds.

### Material Efficiency

```
builds_possible = material_count / 10
```

999 wood = 99 builds maximum. 500 wood (competitive) = 50 builds.

### Structure Break Time

```
break_time = structure_current_HP / weapon_DPS_to_structures
```

---

## Sources

- [Fortnite Wiki -- Health (Battle Royale)](https://fortnite.fandom.com/wiki/Health_(Battle_Royale))
- [Fortnite Wiki -- The Storm](https://fortnite.fandom.com/wiki/The_Storm)
- [Fortnite Wiki -- Materials (Battle Royale)](https://fortnite.fandom.com/wiki/Materials_(Battle_Royale))
- [Fortnite Wiki -- Building](https://fortnite.fandom.com/wiki/Building)
- [Fortnite Wiki -- Weaponry (Battle Royale)](https://fortnite.fandom.com/wiki/Weaponry_(Battle_Royale))
- [Fortnite Wiki -- Vehicles](https://fortnite.fandom.com/wiki/Vehicles)
- [Fortnite Wiki -- Rebooting](https://fortnite.fandom.com/wiki/Rebooting)
- [Fortnite Archive Wiki -- The Storm (Battle Royale)](https://fortnite-archive.fandom.com/wiki/The_Storm_(Battle_Royale))
- [Fortnite Archive Wiki -- Building](https://fortnite-archive.fandom.com/wiki/Building)
- [Fortnite Archive Wiki -- Damage in Battle Royale](https://fortnite-archive.fandom.com/wiki/Damage_in_Battle_Royale)
- [DotEsports -- Full Stats of All Fortnite Weapons](https://dotesports.com/fortnite/news/full-stats-of-all-fortnite-battle-royale-weapons)
- [Fortnite.GG -- Weapons](https://fortnite.gg/weapons)
- [Gaming Tools -- Fortnite Storm Damage Table](https://gaming-tools.com/fortnite/the-storm/)
- [Gaming Tools -- Fortnite Health and Shield Guide](https://gaming-tools.com/fortnite/health-and-shield-device-guide/)
- [HotSpawn -- Fortnite Storm Mechanics](https://www.hotspawn.com/fortnite/guide/fortnite-storm-mechanics)
- [GameRant -- Fortnite Headshot Damage Stats](https://gamerant.com/fortnite-headshot-damage-stats/)
- [Kr4m -- Best Material for Building](https://kr4m.com/fortnite-material/)
- [Fort Fanatics -- Fortnite Materials Deep Dive](https://fortfanatics.com/fortnite-materials/)
- [PlayerAuctions -- Fortnite Weapon Guide 2026](https://www.playerauctions.com/fortnite-guide/weapons/fortnite-definitive-weapon-guide/)
- [The Gamer -- Fortnite Reboot Vans and Cards](https://www.thegamer.com/fortnite-reboot-vans-cards-explained/)
- [VideoGamer -- How Long Is a Fortnite Match](https://www.videogamer.com/guides/fortnite-how-long-is-match/)
- [Arena FPS -- Fortnite Shields/Armor System](https://arenafps.com/fortnite-battle-royale-shields-armor-system-mechanics/)
