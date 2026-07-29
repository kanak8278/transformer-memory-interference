# Minecraft Java Edition Survival Multiplayer Mechanics Reference

> Exhaustive reference for building a player state simulator.
> Based on Java Edition 1.21+ mechanics. Values sourced from Minecraft Wiki and community resources.

---

## 1. Health System

### Base Health
- Player has **20 HP** (10 hearts)
- Each heart = 2 HP
- Health displayed as heart icons on HUD
- Natural regeneration when hunger >= 18 (see Hunger section)

### Damage Types

| Damage Type | Details |
|---|---|
| **Melee (mob/player)** | Reduced by armor, enchantments |
| **Projectile** | Arrows, fireballs, shulker bullets — reduced by armor |
| **Fall** | 1 HP per block fallen beyond 3 blocks (fall from height H: damage = H - 3) |
| **Fire/Lava** | Fire: 1 HP/sec. Lava: 4 HP/half-sec in Overworld, 2 HP/half-sec in Nether (unarmored) |
| **Drowning** | 2 HP/sec after air supply (300 ticks / 15 sec) depletes |
| **Suffocation** | 1 HP/half-sec when head inside solid block |
| **Starvation** | 1 HP/4 sec when hunger = 0 (Easy: stops at 10 HP, Normal: 1 HP, Hard: kills) |
| **Void** | 4 HP/half-sec below Y=-64 |
| **Explosion (Creeper)** | Up to 49 HP (charged) at point-blank on Hard |
| **Explosion (TNT)** | Up to 65 HP at point-blank |
| **Wither effect** | 1 HP/2 sec (like poison but can kill) |
| **Poison** | Drains HP but cannot kill (stops at 1 HP) |
| **Instant Damage** | 6 HP (level I), 12 HP (level II) |
| **Lightning** | 5 HP |
| **Cactus** | 1 HP per contact tick |
| **Berry bush** | 1 HP per movement tick inside bush |
| **Magma block** | 1 HP per tick when standing on it (without sneaking) |
| **Freezing (Powder Snow)** | 1 HP/2 sec after 7 sec exposure |

### Armor Damage Reduction Formula (Java Edition)

The core formula:

```
damage_taken = damage * (1 - min(20, max(armor_defense / 5, armor_defense - (4 * damage) / (armor_toughness + 8))) / 25)
```

Where:
- `armor_defense` = total armor defense points (capped at 30 in attribute, but formula uses actual value)
- `armor_toughness` = total armor toughness (capped at 20)
- `damage` = incoming raw damage

**Key implications:**
- Each armor point gives up to 4% damage reduction
- Maximum natural damage reduction: 80% (full diamond/netherite)
- Toughness prevents armor effectiveness from dropping against high-damage attacks
- Without toughness, max damage reduction is reduced by 2 percentage points per HP of incoming damage

### Enchantment Protection Reduction (applied after armor)

```
damage_after_enchant = damage_after_armor * (1 - min(20, total_EPF) / 25)
```

Where `total_EPF` = sum of Enchantment Protection Factor across all armor pieces.
- Protection I-IV: EPF = level (per piece, max 4 each)
- Maximum total EPF = 20 (capped), giving 80% additional reduction
- Specialized protections (Fire, Blast, Projectile) give 2 * level EPF against their type

### Effective HP with Full Armor Sets

| Full Set | Armor Pts | Toughness | Approx. Damage Reduction (vs 5 dmg hit) |
|---|---|---|---|
| None | 0 | 0 | 0% |
| Leather | 7 | 0 | ~28% |
| Gold | 11 | 0 | ~44% |
| Chainmail | 12 | 0 | ~48% |
| Iron | 15 | 0 | ~60% |
| Diamond | 20 | 8 | ~80% |
| Netherite | 20 | 12 | ~80% |

---

## 2. Hunger System

### Core Variables

| Variable | Range | Initial Value | Description |
|---|---|---|---|
| `foodLevel` | 0–20 | 20 | Hunger bar (10 drumsticks) |
| `foodSaturationLevel` | 0–foodLevel | 5.0 (on spawn) | Hidden buffer before hunger drops |
| `foodExhaustionLevel` | 0–unlimited (resets at 4) | 0.0 | Accumulated exhaustion from actions |
| `foodTickTimer` | 0–80 | 0 | Timer for healing/starvation ticks |

### Exhaustion Mechanics

When `foodExhaustionLevel` >= 4.0:
1. Subtract 4.0 from exhaustion
2. If `foodSaturationLevel` > 0: subtract 1.0 from saturation
3. Else: subtract 1 from `foodLevel`

### Exhaustion Costs by Action

| Action | Exhaustion per unit |
|---|---|
| Swimming (per meter) | 0.01 |
| Walking (per meter) | 0.0 (free in 1.21) |
| Sneaking (per meter) | 0.0 (free in 1.21) |
| Sprinting (per meter) | 0.1 |
| Jumping | 0.05 |
| Sprint-jumping | 0.2 |
| Breaking a block | 0.005 |
| Attacking an entity | 0.1 |
| Taking damage that is absorbed by armor | 0.1 |
| Hunger effect (per tick) | 0.1 (level I) |

> Note: Exhaustion values were significantly reduced in 1.21 Tricky Trials update. Pre-1.21 values were much higher (e.g., sprinting was 0.1/m, attacking was 0.3).

### Healing from Saturation (Peaceful-style rapid heal)

When `foodLevel` = 20 AND `foodSaturationLevel` > 0:
- Heal 1 HP every 0.5 seconds (10 ticks)
- Costs 6.0 exhaustion per HP healed (drains saturation fast)

### Natural Regeneration

When `foodLevel` >= 18 AND `foodSaturationLevel` = 0 (or < threshold):
- Heal 1 HP every 4 seconds (80 ticks)
- Costs 6.0 exhaustion per HP healed

### Starvation

When `foodLevel` = 0:
- Take 1 HP damage every 4 seconds (80 ticks)
- Easy: stops at 10 HP (5 hearts)
- Normal: stops at 1 HP (half heart)
- Hard: can kill player

### Food Values (Key Foods)

| Food | Hunger Restored | Saturation Restored | Effective Quality |
|---|---|---|---|
| Golden Carrot | 6 | 14.4 | Best saturation in game |
| Cooked Steak | 8 | 12.8 | Best hunger + saturation combo |
| Cooked Porkchop | 8 | 12.8 | Same as steak |
| Cooked Mutton | 6 | 9.6 | Good mid-tier |
| Cooked Salmon | 6 | 9.6 | Same as mutton |
| Golden Apple | 4 | 9.6 | Also gives Absorption II (4 min), Regen II (5 sec) |
| Enchanted Golden Apple | 4 | 9.6 | Absorption IV (2 min), Regen II (20 sec), Resistance (5 min), Fire Resist (5 min) |
| Bread | 5 | 6.0 | Easy to farm |
| Cooked Chicken | 6 | 7.2 | Good early game |
| Baked Potato | 5 | 6.0 | Same as bread |
| Cooked Cod | 5 | 6.0 | Same as bread |
| Cooked Rabbit | 5 | 6.0 | Same as bread |
| Apple | 4 | 2.4 | Low saturation |
| Melon Slice | 2 | 1.2 | Very low |
| Raw Beef | 3 | 1.8 | Cook it |
| Rotten Flesh | 4 | 0.8 | 80% chance Hunger effect (30 sec) |
| Spider Eye | 2 | 3.2 | Gives Poison (4 sec) |
| Suspicious Stew | 6 | 7.2 | Effect varies by flower used |
| Dried Kelp | 1 | 0.6 | Fastest to eat (0.865 sec vs normal 1.61 sec) |
| Chorus Fruit | 4 | 2.4 | Teleports randomly |
| Cake (per slice) | 2 | 0.4 | 7 slices total, placed as block |
| Pumpkin Pie | 8 | 4.8 | High hunger, low saturation |
| Cookie | 2 | 0.4 | Low value |
| Beetroot Soup | 6 | 7.2 | Unstackable |
| Mushroom Stew | 6 | 7.2 | Unstackable |
| Rabbit Stew | 10 | 12.0 | Best hunger restore, unstackable |
| Honey Bottle | 6 | 1.2 | Clears poison |

---

## 3. XP (Experience) System

### XP Required Per Level

| Level Range | XP to Next Level | Formula (XP for level L from 0) |
|---|---|---|
| 0–16 | 2L + 7 | L^2 + 6L |
| 17–31 | 5L - 38 | 2.5L^2 - 40.5L + 360 |
| 32+ | 9L - 158 | 4.5L^2 - 162.5L + 2220 |

### Key XP Totals

| Level | Total XP from 0 | XP to reach from previous |
|---|---|---|
| 1 | 7 | 7 |
| 5 | 55 | 17 |
| 10 | 160 | 27 |
| 15 | 315 | 37 |
| 20 | 550 | 62 |
| 25 | 825 | 87 |
| 27 | 963 | 97 |
| 30 | 1,395 | 112 |
| 40 | 2,920 | 202 |
| 50 | 5,345 | 292 |

### Enchanting Table Costs

- **Slot 1 (top)**: 1 level + 1 lapis lazuli (weakest enchantment)
- **Slot 2 (middle)**: 2 levels + 2 lapis lazuli (medium enchantment)
- **Slot 3 (bottom)**: 3 levels + 3 lapis lazuli (strongest enchantment)

The actual enchantment level offered depends on bookshelves (max 15):
```
enchantment_level = randomInt(1, 8) + floor(bookshelves / 2) + randomInt(0, bookshelves)
```
- Slot 1: base_level / 3 (rounded down, min 1)
- Slot 2: base_level * 2/3 (rounded down, min 1)
- Slot 3: base_level (max level)
- With 15 bookshelves: slot 3 offers level 24–30 enchantments

### XP Sources

| Source | XP Dropped |
|---|---|
| Coal ore | 0–2 |
| Lapis ore | 2–5 |
| Redstone ore | 1–5 |
| Diamond ore | 3–7 |
| Emerald ore | 3–7 |
| Nether quartz ore | 2–5 |
| Monster (most) | 5 |
| Blaze | 10 |
| Ender Dragon (first) | 12,000 |
| Ender Dragon (subsequent) | 500 |
| Wither | 50 |
| Breeding animals | 1–7 |
| Trading with villager | 3–6 (varies) |
| Fishing | 1–6 |
| Smelting (per item) | 0.1–1.0 (varies by item) |
| Bottle o' Enchanting | 3–11 |

### Death XP Penalty
- Drop: `min(7 * level, 100)` XP points
- All remaining XP is lost
- Items drop on ground (5-minute despawn timer)

---

## 4. Armor Tiers

### Defense Points Per Piece

| Tier | Helmet | Chestplate | Leggings | Boots | **Total** | Toughness (per piece) | **Total Toughness** | Knockback Resist |
|---|---|---|---|---|---|---|---|---|
| Leather | 1 | 3 | 2 | 1 | **7** | 0 | **0** | 0 |
| Gold | 2 | 5 | 3 | 1 | **11** | 0 | **0** | 0 |
| Chainmail | 2 | 5 | 4 | 1 | **12** | 0 | **0** | 0 |
| Iron | 2 | 6 | 5 | 2 | **15** | 0 | **0** | 0 |
| Diamond | 3 | 8 | 6 | 3 | **20** | 2 | **8** | 0 |
| Netherite | 3 | 8 | 6 | 3 | **20** | 3 | **12** | 0.1 (per piece, 0.4 total) |

### Armor Durability

| Tier | Helmet | Chestplate | Leggings | Boots |
|---|---|---|---|---|
| Leather | 55 | 80 | 75 | 65 |
| Gold | 77 | 112 | 105 | 91 |
| Chainmail | 165 | 240 | 225 | 195 |
| Iron | 165 | 240 | 225 | 195 |
| Diamond | 363 | 528 | 495 | 429 |
| Netherite | 407 | 592 | 555 | 481 |

### Max Damage Reduction by Tier (approximate)

| Full Set | vs. 1 dmg | vs. 5 dmg | vs. 10 dmg | vs. 20 dmg |
|---|---|---|---|---|
| Leather (7) | 28% | 28% | 20% | 8% |
| Iron (15) | 60% | 60% | 52% | 36% |
| Diamond (20, T=8) | 80% | 80% | 76% | 68% |
| Netherite (20, T=12) | 80% | 80% | 78% | 72% |

---

## 5. Weapon Damage Values

### Swords (Java Edition)

| Tier | Attack Damage | Attack Speed | DPS | Durability |
|---|---|---|---|---|
| Wooden | 4 | 1.6 | 6.4 | 59 |
| Stone | 5 | 1.6 | 8.0 | 131 |
| Iron | 6 | 1.6 | 9.6 | 250 |
| Gold | 4 | 1.6 | 6.4 | 32 |
| Diamond | 7 | 1.6 | 11.2 | 1561 |
| Netherite | 8 | 1.6 | 12.8 | 2031 |

**Sweep attack**: Java Edition only. When standing still or walking (not sprinting), a fully charged sword attack hits nearby entities for 1 + (attack_damage * sweep_efficiency_level / (sweep_efficiency_level + 1)) damage in a sweep.

### Axes (Java Edition)

| Tier | Attack Damage | Attack Speed | DPS | Durability |
|---|---|---|---|---|
| Wooden | 7 | 0.8 | 5.6 | 59 |
| Stone | 9 | 0.8 | 7.2 | 131 |
| Iron | 9 | 0.9 | 8.1 | 250 |
| Gold | 7 | 1.0 | 7.0 | 32 |
| Diamond | 9 | 1.0 | 9.0 | 1561 |
| Netherite | 10 | 1.0 | 10.0 | 2031 |

**Shield disable**: Axes have a 100% chance to disable shields for 5 seconds when fully charged.

### Bows

| Draw Time | Damage | Notes |
|---|---|---|
| No charge (0 ticks) | 1 | |
| Medium charge (~0.3s) | 6 | |
| Full charge (1s, 20 ticks) | 6 | Base full draw |
| Critical (full charge + random) | 6–11 (avg ~9) | Particles on arrow = critical |

- **Arrow speed**: 3 blocks/tick at full charge
- **Max range**: ~120 blocks horizontal (angled ~45 deg)
- **Durability**: 384 uses
- **Infinity enchantment**: Unlimited arrows (still need 1 arrow in inventory)

### Crossbows

| Stat | Value |
|---|---|
| Loading time | 1.25 seconds (25 ticks) |
| Damage (arrow) | 6–11 (similar to bow) |
| Damage (firework rocket) | 5–6 base + up to 12 from stars (max ~18) |
| Durability | 465 uses (326 for fireworks) |
| Range | Slightly longer than bow |

- **Multishot**: Fires 3 arrows for cost of 1 (only 1 can be picked up)
- **Piercing**: Arrows pass through entities (level = number of extra entities)
- **Quick Charge**: Reduces load time by 0.25 sec per level (Quick Charge III = 0.5 sec load)

### Tridents

| Stat | Value |
|---|---|
| Melee damage | 9 |
| Melee attack speed | 1.1 |
| Melee DPS | 9.9 |
| Ranged damage | 8 |
| Durability | 250 |

- **Loyalty**: Returns after thrown (higher level = faster return)
- **Riptide**: Launches player in water/rain (cannot be thrown)
- **Channeling**: Summons lightning on hit during thunderstorm
- **Impaling**: +2.5 damage per level vs aquatic mobs (Java: vs mobs in water/rain)

### Other Damage Sources

| Weapon/Item | Damage |
|---|---|
| Fist (bare hand) | 1 |
| Mace (new 1.21) | 7 base, +wind charge bonus on fall |
| Snowball | 0 (knockback only, 3 dmg to Blazes) |
| Egg | 0 (knockback only) |
| Fishing rod | 0 (knockback only) |
| Shield (bash) | 0 (knockback only, blocks attacks) |

---

## 6. Tool Tiers and Durability

### Tool Durability (all tool types share same durability per tier)

| Tier | Durability | Mining Speed | Attack Damage (Pickaxe) |
|---|---|---|---|
| Wooden | 59 | 2.0x | 2 |
| Stone | 131 | 4.0x | 3 |
| Iron | 250 | 6.0x | 4 |
| Gold | 32 | 12.0x | 2 |
| Diamond | 1561 | 8.0x | 5 |
| Netherite | 2031 | 9.0x | 6 |

### Mining Level Requirements

| Tier | Can Mine |
|---|---|
| Wooden | Stone, coal ore, nether quartz |
| Stone | Iron ore, copper ore, lapis ore |
| Iron | Diamond ore, gold ore, emerald ore, redstone ore, obsidian |
| Diamond | Obsidian, ancient debris, crying obsidian |
| Netherite | Same as diamond (just faster + more durable) |

### Block Breaking Times (seconds, correct tool, no enchantments)

| Block | Wood | Stone | Iron | Diamond | Netherite |
|---|---|---|---|---|---|
| Stone | 1.15 | 0.6 | 0.4 | 0.3 | 0.25 |
| Obsidian | -- | -- | -- | 9.4 | 8.35 |
| Iron Ore | -- | 0.75 | 0.5 | 0.4 | 0.35 |
| Diamond Ore | -- | -- | 0.6 | 0.45 | 0.4 |
| Ancient Debris | -- | -- | -- | 22.5 | 20.0 |
| Dirt (shovel) | 0.4 | 0.2 | 0.15 | 0.1 | 0.1 |
| Log (axe) | 0.35 | 0.2 | 0.15 | 0.1 | 0.1 |

### Efficiency Enchantment Speed Boost

Mining speed multiplied by: `1 + level^2` (so Efficiency V = 1 + 25 = 26x multiplier on top of base tool speed).

---

## 7. Potion Effects

### Brewable Potions (Base Duration, Extended, Enhanced)

| Potion | Effect | Base Duration | Extended (Redstone) | Enhanced (Glowstone) |
|---|---|---|---|---|
| **Speed** | +20% movement speed | 3:00 | 8:00 | 1:30 (Speed II: +40%) |
| **Slowness** | -15% movement speed | 1:30 | 4:00 | 0:20 (Slowness IV: -60%) |
| **Strength** | +3 attack damage | 3:00 | 8:00 | 1:30 (Strength II: +6 dmg) |
| **Weakness** | -4 attack damage | 1:30 | 4:00 | N/A |
| **Instant Health** | Heals 4 HP | Instant | N/A | Instant (Heals 8 HP) |
| **Instant Damage** | Deals 6 HP | Instant | N/A | Instant (Deals 12 HP) |
| **Regeneration** | +1 HP/2.5 sec | 0:45 | 1:30 | 0:22 (Regen II: +1 HP/1.25 sec) |
| **Poison** | -1 HP/1.25 sec (min 1 HP) | 0:45 | 1:30 | 0:21 (Poison II: -1 HP/0.625 sec) |
| **Fire Resistance** | Immune to fire/lava damage | 3:00 | 8:00 | N/A |
| **Water Breathing** | No drowning, clearer vision | 3:00 | 8:00 | N/A |
| **Night Vision** | See in darkness as if max light | 3:00 | 8:00 | N/A |
| **Invisibility** | Player invisible (armor still shows) | 3:00 | 8:00 | N/A |
| **Leaping** | +50% jump height | 3:00 | 8:00 | 1:30 (Jump Boost II: +100%) |
| **Slow Falling** | Fall slowly, no fall damage | 1:30 | 4:00 | N/A |
| **Turtle Master** | Slowness IV + Resistance III | 0:20 | 0:40 | 0:20 (Slow VI + Resist IV) |
| **Swiftness** | Same as Speed | (see Speed) | (see Speed) | (see Speed) |

### Splash Potion Modifiers
- Same duration as drinkable potion
- Area of effect: 4-block radius
- Duration reduced based on distance from impact

### Lingering Potion Modifiers
- Duration = 1/4 of drinkable potion
- Creates area-of-effect cloud lasting 30 seconds
- Cloud shrinks as entities pass through it

### Non-Brewable Status Effects

| Effect | Source | Description |
|---|---|---|
| Absorption | Golden Apple, Enchanted Golden Apple, Totem | Extra hearts (yellow) above normal HP |
| Resistance | Enchanted Golden Apple, Beacon | Reduces all damage by 20% per level |
| Haste | Beacon, Conduit | Increases mining speed by 20% per level |
| Mining Fatigue | Elder Guardian | Reduces mining speed by 70% per level |
| Nausea | Pufferfish | Screen wobble/distortion |
| Blindness | Suspicious Stew (certain flowers) | Limits visibility to ~3 blocks |
| Hunger | Rotten Flesh, Husk, Pufferfish | Increases exhaustion by 0.1/tick |
| Wither | Wither, Wither Skeleton, Wither Rose | 1 HP/2 sec, can kill (unlike poison) |
| Glowing | Spectral Arrow | Outline visible through walls |
| Levitation | Shulker | Floats upward ~0.9 blocks/sec per level |
| Bad Luck | N/A (command only) | Reduces loot table quality |
| Hero of the Village | Completing a raid | Discounted villager trades |
| Conduit Power | Conduit | Water breathing + night vision + haste (underwater) |
| Darkness | Warden (sculk shrieker) | Periodic vision blackout |
| Bad Omen | Killing raid captain | Triggers raid when entering village |
| Trial Omen | Trial Spawner (1.21) | Triggers ominous trial spawner |
| Wind Charged | Breeze (1.21) | Creates wind burst on hit |
| Weaving | Breeze (1.21) | Spawns cobwebs on hit |
| Oozing | Breeze (1.21) | Spawns slimes on death |
| Infested | Breeze (1.21) | Spawns silverfish on hit |

---

## 8. Enchantment System

### Armor Enchantments

| Enchantment | Max Level | Effect | Applicable To |
|---|---|---|---|
| Protection | IV | -4% damage per level (all damage) | All armor |
| Fire Protection | IV | -8% fire damage per level, reduces burn time | All armor |
| Blast Protection | IV | -8% explosion damage per level, reduces knockback | All armor |
| Projectile Protection | IV | -8% projectile damage per level | All armor |
| Thorns | III | 15% chance per level to deal 1-4 damage to attacker | All armor (chestplate naturally) |
| Unbreaking | III | Chance to not consume durability: 100/(level+1)% | All armor |
| Mending | I | XP orbs repair item instead of going to level | All armor |
| Respiration | III | +15 sec underwater per level, reduces drowning damage | Helmet |
| Aqua Affinity | I | Normal mining speed underwater | Helmet |
| Depth Strider | III | +33% underwater movement speed per level | Boots |
| Frost Walker | II | Creates frosted ice on water surface | Boots |
| Feather Falling | IV | -12% fall damage per level (48% at IV) | Boots |
| Soul Speed | III | +20% speed per level on soul sand/soil (durability cost) | Boots |
| Swift Sneak | III | Faster sneaking speed | Leggings |
| Curse of Vanishing | I | Item destroyed on death | All armor |
| Curse of Binding | I | Cannot remove once equipped | All armor |

> Protection types are **mutually exclusive** (only one of Protection / Fire / Blast / Projectile per piece). EPF (Enchantment Protection Factor) cap = 20 across all pieces.

### Weapon Enchantments

| Enchantment | Max Level | Effect | Applicable To |
|---|---|---|---|
| Sharpness | V | +0.5 * level + 0.5 damage (V = +3 dmg) | Swords, Axes |
| Smite | V | +2.5 damage per level vs undead | Swords, Axes |
| Bane of Arthropods | V | +2.5 damage per level vs spiders/silverfish/bees/endermites | Swords, Axes |
| Knockback | II | Increased knockback | Swords |
| Fire Aspect | II | Sets target on fire (4 sec per level, I=3-4 dmg, II=7 dmg) | Swords |
| Looting | III | +1 max drop per level, improved rare drop chance | Swords |
| Sweeping Edge | III | Sweep damage = level/(level+1) of attack damage | Swords (Java only) |
| Unbreaking | III | Chance to not consume durability | Swords, Axes |
| Mending | I | XP repairs item | Swords, Axes |

> Sharpness, Smite, Bane of Arthropods are **mutually exclusive**.

### Bow Enchantments

| Enchantment | Max Level | Effect |
|---|---|---|
| Power | V | +25% damage per level (at V: +150%, so 15-25.5 dmg at full charge) |
| Punch | II | Increased knockback |
| Flame | I | Arrows set target on fire (5 HP fire damage) |
| Infinity | I | No arrow consumption (need 1 arrow in inventory) |
| Unbreaking | III | Durability preservation |
| Mending | I | XP repairs item |

> Infinity and Mending are **mutually exclusive**.

### Crossbow Enchantments

| Enchantment | Max Level | Effect |
|---|---|---|
| Quick Charge | III | -0.25 sec load time per level (III = 0.5 sec load) |
| Multishot | I | Fires 3 arrows (only 1 can be picked up) |
| Piercing | IV | Arrow passes through entities (level = extra entities hit) |
| Unbreaking | III | Durability preservation |
| Mending | I | XP repairs item |

> Multishot and Piercing are **mutually exclusive**.

### Tool Enchantments

| Enchantment | Max Level | Effect |
|---|---|---|
| Efficiency | V | Increases mining speed by level^2 + 1 |
| Fortune | III | Increases ore drops (III: average 2.2x for diamond) |
| Silk Touch | I | Mines block itself (e.g., diamond ore block) |
| Unbreaking | III | Durability preservation |
| Mending | I | XP repairs item |

> Fortune and Silk Touch are **mutually exclusive**.

### Trident Enchantments

| Enchantment | Max Level | Effect |
|---|---|---|
| Loyalty | III | Returns after thrown (faster at higher levels) |
| Riptide | III | Launches player when thrown in water/rain |
| Channeling | I | Summons lightning on hit during thunderstorm |
| Impaling | V | +2.5 damage per level vs aquatic mobs (Java: in water/rain) |
| Unbreaking | III | Durability preservation |
| Mending | I | XP repairs item |

> Loyalty/Channeling and Riptide are **mutually exclusive**.

### Anvil Mechanics

- Combining enchantments costs XP levels
- **Prior Work Penalty**: Each anvil use doubles the penalty: 2^n - 1 levels (where n = number of prior uses)
- **Too Expensive**: Operation blocked when cost >= 40 levels (in survival)
- Renaming: 1 level base cost
- Repair with material: 1 level base cost per unit + enchantment costs

---

## 9. Mob Types and Damage

### Common Hostile Mobs

| Mob | Health (HP) | Attack Damage (Normal) | Special |
|---|---|---|---|
| Zombie | 20 | 3 | Can pick up items, baby zombies are faster |
| Skeleton | 20 | Bow: 2-5 (varies with difficulty/range) | Strafes, burns in sunlight |
| Creeper | 20 | Explosion: ~25 (normal), ~49 (charged) | Explodes near player (1.5s fuse) |
| Spider | 16 | 2 | Climbs walls, hostile only at night/dark |
| Cave Spider | 12 | 2 + Poison (7-15 sec Normal/Hard) | Fits through 1x0.5 gaps |
| Enderman | 40 | 7 | Teleports, hostile only when looked at/attacked |
| Witch | 26 | Splash potions (Poison, Harming, Slowness) | Drinks healing/fire resist potions |
| Slime | Tiny: 1, Small: 4, Big: 16 | Tiny: 0, Small: 2, Big: 4 | Splits into smaller slimes |
| Phantom | 20 | 2 (Easy), 2 (Normal), 3 (Hard) | Spawns when player hasn't slept 3+ days |
| Drowned | 20 | 3 (melee), Trident: 8 | Can throw tridents |

### Nether Hostile Mobs

| Mob | Health (HP) | Attack Damage (Normal) | Special |
|---|---|---|---|
| Blaze | 20 | Melee: 6, Fireball: 5 | Flies, shoots 3 fireballs |
| Ghast | 10 | Fireball: up to 17 (explosion) | Flies, fireballs deflectable |
| Wither Skeleton | 20 | 8 + Wither effect (10 sec) | Drops skulls (rare) for Wither boss |
| Magma Cube | Tiny: 1, Small: 4, Big: 16 | Tiny: 3, Small: 4, Big: 6 | Like slimes but fireproof |
| Hoglin | 40 | 3-8 (varies by difficulty) | Breedable, avoids warped fungus |
| Piglin | 16 | 5-8 (crossbow or sword) | Trades gold, hostile without gold armor |
| Piglin Brute | 50 | 13 (Normal), 19 (Hard) | Always hostile, golden axe |
| Zoglin | 40 | 3-8 | Zombified hoglin, hostile to everything |

### End Mobs

| Mob | Health (HP) | Attack Damage | Special |
|---|---|---|---|
| Enderman | 40 | 7 | Same as overworld |
| Shulker | 30 | 4 + Levitation (10 sec) | Hides in shell (armor when closed) |
| Ender Dragon | 200 | Melee: 10 (body), 5 (wing) | Boss. Non-head hits deal 25% damage |
| Endermite | 8 | 2 | Rare spawn from ender pearl use |

### Boss Mobs

| Boss | Health (HP) | Attack | Special |
|---|---|---|---|
| Ender Dragon | 200 | 10 (charge), 5 (wing) | End crystals heal it, destroy crystals first |
| Wither | 300 (Java) | Skull: 8 + Wither II | At 50% HP gains Wither Armor (arrow immune), spawns Wither effect |
| Elder Guardian | 80 | 8 (spikes) + Mining Fatigue III (5 min) | Found in ocean monuments (3 per) |
| Warden | 500 | 16 (Normal), 22.5 (Hard) | Does not drop loot, spawned by sculk shriekers |

### Breeze (1.21 Tricky Trials)

| Stat | Value |
|---|---|
| Health | 30 HP |
| Attack | Wind charge: 1 damage + knockback |
| Special | Found in trial chambers, deflects projectiles |

### Passive/Neutral Mobs (Key ones for farming)

| Mob | Health | Drops |
|---|---|---|
| Cow | 10 | Leather (0-2), Raw Beef (1-3) |
| Pig | 10 | Raw Porkchop (1-3) |
| Sheep | 8 | Wool (1), Raw Mutton (1-2) |
| Chicken | 4 | Feathers (0-2), Raw Chicken (1) |
| Rabbit | 3 | Rabbit Hide (0-1), Raw Rabbit (0-1) |
| Horse | 15-30 (varies) | Leather (0-2) |
| Wolf (tamed) | 20 | N/A (pet) |
| Iron Golem | 100 | Iron Ingot (3-5), Poppy (0-2) |

---

## 10. Dimension Mechanics

### Overworld

- **Y range**: -64 to 320 (build height 384 blocks)
- **Bedrock layer**: Y = -64 to -60 (mixed with deepslate)
- **Sea level**: Y = 63
- **Day/night cycle**: 20 minutes (24,000 ticks)
  - Day: 10 min, Sunset: 1.5 min, Night: 7 min, Sunrise: 1.5 min
- **Hostile mob spawning**: Light level 0 only (as of 1.18+)
- **Weather**: Clear, rain, thunderstorm
- **Bed**: Sets spawn point, skips night (all players must sleep on server)

### The Nether

- **Y range**: 0 to 256 (usable: 0 to 127, bedrock ceiling at 127-128)
- **Coordinate ratio**: 1 Nether block = 8 Overworld blocks (X and Z)
- **No day/night cycle**: Always "night" ambient
- **No weather**: No rain, no thunderstorms
- **Water**: Instantly evaporates (cannot place water)
- **Beds**: **EXPLODE** when used (6 HP explosion damage, sets fire)
- **Compass/Clock**: Spin randomly (nonfunctional)
- **Respawn Anchor**: Charged with glowstone blocks (max 4 charges), sets spawn in Nether

### Nether Portal Mechanics

- **Construction**: Obsidian frame, minimum 4x5 (internal 2x3), maximum 23x23
- **Activation**: Flint and steel on inside face, or fire charge
- **Travel**: Stand in portal for 4 seconds (80 ticks) in Survival
- **Coordinate conversion**: Overworld (X, Z) -> Nether (X/8, Z/8)
- **Portal searching**: Game searches for existing portal within 128 blocks (Overworld) or 16 blocks (Nether) in destination dimension
- **Portal creation**: If no existing portal found, creates one at nearest valid location

### The End

- **Access**: End Portal in Stronghold (12 Eyes of Ender to activate, found 1-3 per world)
- **Spawning**: Player spawns on obsidian platform at fixed coordinates
- **Main island**: Central island with Ender Dragon and obsidian pillars with End Crystals
- **Outer islands**: Generated 1000+ blocks from center after defeating dragon
- **End Gateway**: Small portals to outer islands (generated when dragon defeated)
- **Return portal**: Activates when Ender Dragon defeated (bedrock fountain with dragon egg)
- **Dragon egg**: Drops once per world (first dragon kill only)
- **End Cities**: Found on outer islands, contain shulkers and elytra
- **Chorus Plants**: Found on outer islands, chorus fruit teleports randomly
- **Beds**: **EXPLODE** when used (same as Nether)
- **No day/night**: Constant dark sky with end stone
- **No weather**: No rain/thunderstorms
- **Void**: Below Y=0 (dangerous)
- **Endermen**: Spawn abundantly

### Stronghold Generation

- 128 strongholds per world (Java Edition)
- First ring: 3 strongholds, 1408-2688 blocks from origin
- Eyes of Ender: Thrown to find direction, 20% chance to break each throw

---

## 11. Ore Distribution by Y-Level (1.18+ / 1.21)

> Ores use triangular distribution: most common at peak Y, tapering off at edges.

| Ore | Y Range | Peak Y | Notes |
|---|---|---|---|
| **Coal** | 0 to 320 | 96 | Also 136 (smaller batch). Not below Y=0 |
| **Copper** | -16 to 112 | 48 | Extra common in Dripstone Caves |
| **Iron** | -64 to 320 | -16 and 232 (two peaks) | Massive range, two separate distributions |
| **Gold** | -64 to 32 | -16 | Extra common in Badlands (32 to 256) |
| **Lapis Lazuli** | -64 to 64 | -1 | Exposed to air less often (buried distribution) |
| **Redstone** | -64 to 16 | -59 | Concentrated deep underground |
| **Diamond** | -64 to 16 | -59 | Best at Y=-59 (just above bedrock) |
| **Emerald** | -16 to 320 | 236 | **Mountain biomes only** (Windswept, Stony Peaks, etc.) |
| **Nether Gold** | 10 to 117 | (Nether only) | Common, mined with any pickaxe |
| **Nether Quartz** | 10 to 117 | (Nether only) | Very common |
| **Ancient Debris** | 8 to 119 | 15 | Rare, blast-resistant, needs diamond+ pickaxe |

### Optimal Mining Strategies

| Target | Best Y-Level | Method |
|---|---|---|
| Diamond | Y = -59 | Strip mining (branch mining at Y=-59) |
| Iron (deep) | Y = -16 | Strip mining or caving |
| Iron (high) | Y = 232 | Mountain tops |
| Gold | Y = -16 | Strip mining |
| Redstone | Y = -59 | Same level as diamond |
| Lapis | Y = -1 | Cave exploring |
| Copper | Y = 48 | Mid-level caves |
| Coal | Y = 96 | Surface/hill caves |
| Emerald | Y = 236 | Mountain peaks (Windswept Hills, Meadow, etc.) |
| Ancient Debris | Y = 15 | TNT/bed mining in Nether |

---

## 12. Biome Types

### Overworld Biomes (54 total in Java Edition)

#### Temperate/Plains
- Plains
- Sunflower Plains
- Meadow
- Cherry Grove (1.20+)

#### Forest
- Forest
- Flower Forest
- Birch Forest
- Old Growth Birch Forest
- Dark Forest
- Pale Garden (1.21.4+)

#### Taiga
- Taiga
- Old Growth Pine Taiga
- Old Growth Spruce Taiga
- Snowy Taiga

#### Mountains
- Windswept Hills
- Windswept Gravelly Hills
- Windswept Forest
- Stony Peaks
- Frozen Peaks
- Jagged Peaks
- Snowy Slopes
- Grove

#### Desert/Warm
- Desert
- Badlands
- Eroded Badlands
- Wooded Badlands
- Savanna
- Savanna Plateau
- Windswept Savanna

#### Jungle
- Jungle
- Sparse Jungle
- Bamboo Jungle

#### Snowy
- Snowy Plains
- Ice Spikes
- Frozen River
- Snowy Beach

#### Swamp
- Swamp
- Mangrove Swamp

#### Ocean
- Ocean
- Deep Ocean
- Warm Ocean
- Lukewarm Ocean
- Deep Lukewarm Ocean
- Cold Ocean
- Deep Cold Ocean
- Frozen Ocean
- Deep Frozen Ocean

#### Beach/River
- Beach
- Stony Shore
- River

#### Cave (sub-biomes)
- Lush Caves
- Dripstone Caves
- Deep Dark

#### Mushroom
- Mushroom Fields (no hostile mob spawning)

### Nether Biomes (5 total)

| Biome | Key Features | Unique Mobs |
|---|---|---|
| Nether Wastes | Netherrack, lava, fortresses | Zombified Piglins, Ghasts |
| Crimson Forest | Crimson nylium, huge fungi | Piglins, Hoglins |
| Warped Forest | Warped nylium, blue fungi | Endermen (abundant) |
| Soul Sand Valley | Soul sand/soil, fossils | Skeletons, Ghasts |
| Basalt Deltas | Basalt columns, magma | Magma Cubes |

### End Biomes (5 total)

| Biome | Location |
|---|---|
| The End | Central island (dragon fight) |
| End Highlands | Outer islands with End Cities, chorus |
| End Midlands | Between highlands and barrens |
| End Barrens | Edge islands, minimal features |
| Small End Islands | Tiny floating islands |

---

## 13. Crafting Progression

### Tier Progression Path

```
Wood -> Stone -> Iron -> Diamond -> Netherite
  |        |        |        |          |
 Punch   Craft   Smelt   Mine at    Mine Ancient
 tree    stone   iron    Y=-59      Debris in
 -> logs tools   ore     with iron  Nether at
 -> planks       in      pickaxe    Y=15 with
 -> sticks       furnace            diamond pick
 -> crafting                        -> smelt to
    table                             scraps
 -> wooden                          -> 4 scraps +
    tools                             4 gold ingots
                                    -> 1 netherite
                                      ingot
                                    -> smithing table
                                      upgrade
```

### Stage 1: Wood (minutes 0-5)

1. Punch tree -> Oak Log
2. Log -> 4 Planks
3. 2 Planks -> 4 Sticks
4. 4 Planks -> Crafting Table
5. Craft: Wooden Pickaxe (3 planks + 2 sticks)
6. Craft: Wooden Axe, Wooden Sword (optional)

### Stage 2: Stone (minutes 5-15)

1. Mine stone with wooden pickaxe -> Cobblestone
2. Craft: Stone Pickaxe, Stone Axe, Stone Sword
3. Craft: Furnace (8 cobblestone)
4. Smelt: Charcoal (logs in furnace) for torches
5. Craft: Torches (charcoal + sticks)

### Stage 3: Iron (minutes 15-60)

1. Mine iron ore (stone+ pickaxe required)
2. Smelt in furnace -> Iron Ingots
3. Key crafts:
   - Iron Pickaxe (enables mining diamond, gold, redstone, emerald)
   - Iron Sword
   - Iron Armor (full set = 24 ingots)
   - Shield (1 iron ingot + 6 planks)
   - Bucket (3 iron ingots) -- critical for lava/water
   - Flint and Steel (1 iron + 1 flint)

### Stage 4: Diamond (hours 1-5+)

1. Mine diamond ore at Y=-59 (iron+ pickaxe required)
2. Key crafts:
   - Diamond Pickaxe (enables mining obsidian, ancient debris)
   - Diamond Sword
   - Diamond Armor (full set = 24 diamonds)
   - Enchanting Table (2 diamonds + 4 obsidian + 1 book)
   - Jukebox, Diamond Hoe (optional)
3. Build enchanting setup: 15 bookshelves around enchanting table

### Stage 5: Netherite (hours 5+)

1. Build Nether Portal (10-14 obsidian, need diamond pickaxe to mine obsidian)
2. Travel to Nether
3. Mine Ancient Debris at Y=15 (diamond+ pickaxe, blast resistant)
   - Method: TNT/bed mining or strip mining
   - Need: 4 Ancient Debris per Netherite Ingot
4. Smelt Ancient Debris -> Netherite Scrap (1:1)
5. Craft: 4 Netherite Scraps + 4 Gold Ingots = 1 Netherite Ingot
6. Find Netherite Upgrade Smithing Template (in Bastion Remnants)
7. Smithing Table: Template + Diamond Item + Netherite Ingot = Netherite Item
   - Preserves all enchantments
   - No XP cost
   - No prior work penalty increase
8. Full gear upgrade: 4 ingots for armor + 1 per tool/weapon = ~9 ingots = 36 Ancient Debris

### Key Intermediate Crafts

| Item | Materials | Importance |
|---|---|---|
| Crafting Table | 4 Planks | Enables all crafting |
| Furnace | 8 Cobblestone | Smelting ores, cooking food |
| Blast Furnace | 5 Iron + 3 Smooth Stone + Furnace | 2x smelting speed (ores only) |
| Smoker | 4 Logs + Furnace | 2x cooking speed (food only) |
| Anvil | 3 Iron Blocks + 4 Iron Ingots (31 total) | Combining enchantments, renaming |
| Enchanting Table | 2 Diamonds + 4 Obsidian + 1 Book | Enchanting items |
| Brewing Stand | 1 Blaze Rod + 3 Cobblestone | Brewing potions |
| Smithing Table | 2 Iron Ingots + 4 Planks | Netherite upgrades |
| Grindstone | 2 Sticks + 1 Stone Slab + 2 Planks | Remove enchantments, repair |
| Beacon | 3 Obsidian + 5 Glass + 1 Nether Star | Area buffs (needs pyramid of mineral blocks) |
| Respawn Anchor | 6 Crying Obsidian + 3 Glowstone | Nether spawn point |
| Ender Chest | 8 Obsidian + 1 Eye of Ender | Shared inventory across all ender chests |
| Shulker Box | 2 Shulker Shells + Chest | Portable storage (keeps items when broken) |

### Crafting Table Recipes: Iron Ingot Requirements Summary

| Item | Iron Ingots | Priority |
|---|---|---|
| Iron Pickaxe | 3 | Critical (enables diamond mining) |
| Iron Sword | 2 | High |
| Bucket | 3 | Critical (lava for portal, water for safety) |
| Shield | 1 | High (blocks 100% melee/projectile) |
| Iron Helmet | 5 | Medium |
| Iron Chestplate | 8 | Medium |
| Iron Leggings | 7 | Medium |
| Iron Boots | 4 | Medium |
| Full Iron Armor | 24 | Medium (total) |
| Anvil | 31 | Late iron stage |
| Flint & Steel | 1 | Medium (portal activation) |
| Minecart | 5 | Low (transport) |
| Rails (16) | 6 + 1 stick | Low |
| Hopper | 5 + 1 chest | Medium (automation) |
| Iron Door (2) | 6 | Low |
| Cauldron | 7 | Low |
| Compass | 4 + 1 redstone | Low |

---

## 14. Multiplayer-Specific Mechanics (SMP)

### PvP Combat (Java Edition)

- **Attack cooldown**: Must wait for full charge for max damage
- **Critical hits**: Jump and strike on way down = +50% damage (needs full charge)
- **Shield blocking**: Blocks 100% of melee and projectile damage from front
- **Axes disable shields**: 100% chance when fully charged (5 second disable)
- **Sweep attacks**: Hit multiple players/mobs in radius when standing still
- **Knockback**: Base KB + Knockback enchantment + sprint knockback (sprint-hit = big KB)

### Player Interaction

- Trading items by dropping
- /team commands for team PvP
- Respawn: At bed/respawn anchor or world spawn
- Friendly fire: On by default, toggleable by server

### Multiplayer Sleep

- All players (or percentage, configurable) must sleep to skip night
- One player sleeping starts "sleep vote" on many servers

### Server Tick Rate

- 20 TPS (ticks per second) standard
- 1 tick = 50 milliseconds
- All game mechanics tied to tick rate

---

## Sources

- [Minecraft Wiki - Armor](https://minecraft.wiki/w/Armor)
- [Minecraft Wiki - Damage](https://minecraft.wiki/w/Damage)
- [Minecraft Wiki - Hunger/Food Mechanics](https://minecraft.wiki/w/Hunger)
- [Minecraft Wiki - Experience](https://minecraft.wiki/w/Experience)
- [Minecraft Wiki - Sword](https://minecraft.wiki/w/Sword)
- [Minecraft Wiki - Axe](https://minecraft.wiki/w/Axe)
- [Minecraft Wiki - Weapon](https://minecraft.wiki/w/Weapon)
- [Minecraft Wiki - Durability](https://minecraft.wiki/w/Durability)
- [Minecraft Wiki - Effect](https://minecraft.wiki/w/Effect)
- [Minecraft Wiki - Brewing](https://minecraft.wiki/w/Brewing)
- [Minecraft Wiki - Enchanting](https://minecraft.wiki/w/Enchanting)
- [Minecraft Wiki - Biome](https://minecraft.wiki/w/Biome)
- [Minecraft Wiki - Ore](https://minecraft.wiki/w/Ore)
- [Minecraft Wiki - The Nether](https://minecraft.wiki/w/The_Nether)
- [Minecraft Wiki - Mob](https://minecraft.wiki/w/Mob)
- [Minecraft Wiki - Netherite](https://minecraft.wiki/w/Netherite)
- [Minecraft Wiki - Health](https://minecraft.wiki/w/Health)
- [Minecraft Ore Distribution Guide (1.21+)](https://blog.berrybyte.net/minecraft-ore-distribution-guide-1-21-best-y-levels-for-every-ore/)
- [Minecraft Food Guide](https://www.minecraftmaps.com/tools/food-reference)
- [Shockbyte Ore Levels](https://shockbyte.com/blog/minecraft-ore-levels)
- [Minecraft XP Calculator](https://www.minecraftmaps.com/tools/xp-calculator)
