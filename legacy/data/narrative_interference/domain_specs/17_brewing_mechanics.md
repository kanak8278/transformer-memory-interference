# Brewing Mechanics: Exhaustive Domain Specification for Batch State Simulation

## 1. Brewing Process Stages (In Order with Exact Timing)

### 1.1 Milling / Grain Preparation
- Crush malt to expose starchy endosperm while keeping husks intact
- Typical gap setting: 0.035-0.045 inches (0.9-1.1 mm)
- Duration: 10-30 minutes depending on volume
- Poor crush leads to low efficiency; over-crush leads to stuck sparge

### 1.2 Mashing
**Purpose**: Convert grain starches into fermentable sugars via enzymatic action.

**Temperature Ranges and Enzyme Activity**:
| Rest Name | Temp Range (F) | Temp Range (C) | Duration | Purpose |
|-----------|----------------|-----------------|----------|---------|
| Acid rest | 95-113 | 35-45 | 15-30 min | Lower mash pH (rarely used with modern malts) |
| Protein rest | 113-138 | 45-59 | 15-30 min | Break down proteins; aids head retention and clarity |
| Beta-amylase rest | 131-149 | 55-65 | 30-60 min | Produces maltose (fermentable); drier, more attenuated beer |
| Alpha-amylase rest | 154-162 | 68-72 | 30-60 min | Produces dextrins (unfermentable); fuller body, sweeter beer |
| Saccharification (combined) | 148-158 | 64-70 | 60 min | Most common single infusion target; balance of fermentable/unfermentable sugars |
| Mashout | 168-170 | 76-77 | 10 min | Denatures enzymes, locks sugar profile, improves flow |

**Chemistry**:
- Beta-amylase: cleaves maltose from non-reducing ends of starch chains. Optimal 130-150F (54-66C). Denatured above 160F (71C).
- Alpha-amylase: randomly cleaves starch into shorter dextrins. Optimal 155-162F (68-72C). Denatured above 175F (80C).
- Lower mash temp (148F) = more fermentable wort = drier beer = higher attenuation
- Higher mash temp (156F) = more dextrins = fuller body = lower attenuation
- Typical mash thickness: 1.25-1.5 quarts water per pound of grain (2.6-3.1 L/kg)
- Total mash duration: 60 minutes for most single-infusion mashes; 90-120 for step mashes

### 1.3 Lautering / Sparging
**Purpose**: Separate sweet wort from spent grain and rinse remaining sugars.

**Process**:
1. **Vorlauf (recirculation)**: 10-15 minutes. Run wort off and pour back on top of grain bed until clear. Grain bed acts as filter.
2. **Runoff**: Drain first runnings (highest gravity, typically 1.070-1.090 for a normal-gravity beer)
3. **Sparging**: Rinse grain bed with 168-170F (76C) water
   - **Fly sparge**: Continuous slow addition, 45-90 minutes. Most efficient.
   - **Batch sparge**: Add water in 1-2 batches, stir, let settle, drain. 20-30 minutes per batch.
   - Stop sparging when runnings drop below 1.008-1.010 SG (to avoid extracting tannins)
   - Sparge water pH should be below 6.0 to prevent tannin extraction

**Duration**: 30-90 minutes total depending on method
**Typical efficiency**: 70-80% for homebrew, 85-95% for commercial

### 1.4 Boiling
**Purpose**: Sterilize wort, isomerize hop alpha acids, drive off DMS, concentrate wort, precipitate proteins (hot break).

**Duration**: 60-90 minutes (90 min for pilsner malt to drive off DMS precursors)

**Hop Addition Schedule** (minutes remaining in boil):
| Addition Time | Purpose | Contribution |
|---------------|---------|--------------|
| 60-90 min | Bittering | Maximum IBU extraction; nearly all aroma/flavor boiled off |
| 30-45 min | Bittering + flavor | Moderate bitterness, some flavor retained |
| 15-20 min | Flavor | Moderate bitterness, noticeable flavor |
| 5-10 min | Flavor + aroma | Low bitterness, significant flavor and aroma |
| 0 min (flameout) | Aroma | Minimal bitterness, maximum aroma |

**Alpha acid utilization**: ~30-35% at 60 min, ~25% at 30 min, ~10% at 15 min, ~5% at 5 min

**Evaporation rate**: 10-15% per hour (affects gravity concentration)

**Hot break**: Occurs in first 5-15 minutes. Protein clumps form and coagulate. Indicates good boil vigor.

**DMS (dimethyl sulfide)**: Corn/cooked vegetable off-flavor. Precursor (SMM) is converted during boil; DMS is driven off by vigorous, uncovered boil. 90-min boil recommended for pilsner malt (higher SMM content).

### 1.5 Whirlpool / Hop Stand
**Purpose**: Separate trub (protein/hop debris) from wort; extract hop aroma at lower temps.

**Process**:
- Whirlpool: Stir wort vigorously in one direction, let settle 10-20 minutes. Trub collects in center cone.
- Hop stand: Add hops at 170-190F (77-88C) and steep for 15-30 minutes. Extracts aroma with minimal isomerization.
- Contributes approximately 5-10 IBU depending on time and temperature
- Increasingly popular in hazy/NEIPA styles

### 1.6 Cooling to Pitching Temperature
**Purpose**: Rapidly cool wort to yeast-safe temperature to prevent DMS, reduce infection risk, and promote cold break (protein precipitation).

**Target temps**:
- Ales: 62-68F (17-20C)
- Lagers: 45-55F (7-13C)
- Belgian ales: 65-72F (18-22C)

**Methods**:
- Immersion chiller: 20-40 minutes to reach pitching temp
- Counterflow/plate chiller: 5-15 minutes
- Ice bath (small batches): 30-60 minutes

**Cold break**: Fine protein particles precipitate out during rapid cooling. Improves clarity and flavor stability.

### 1.7 Fermentation (Primary)
**Purpose**: Yeast converts sugars to alcohol, CO2, and flavor compounds.

**Temperature Ranges by Yeast Type**:
| Yeast Type | Optimal Range (F) | Optimal Range (C) | Notes |
|------------|-------------------|---------------------|-------|
| American ale (US-05 etc.) | 60-72 | 15-22 | Clean profile at 62-66F |
| English ale | 62-72 | 17-22 | More esters at upper range |
| Belgian ale | 65-80 | 18-27 | Phenols and esters desired; often start low and ramp up |
| Hefeweizen | 62-75 | 17-24 | Higher temp = more banana (isoamyl acetate); lower = more clove (4VG) |
| Saison | 68-95 | 20-35 | Often starts at 68F and ramps to 80-95F to ensure full attenuation |
| Lager | 45-55 | 7-13 | Typically 48-52F for primary; very clean profile |
| California Common | 58-65 | 14-18 | Lager yeast at ale temperatures |
| Kolsch | 56-64 | 13-18 | Clean ale yeast at cooler temps |

**Duration**:
- Ales: 5-14 days (most activity in first 3-5 days)
- Lagers: 14-28 days (slower fermentation at cold temps)
- Belgian strongs: 14-21 days
- High gravity (>1.080): 14-28 days

**Pitching Rate**:
- Ales: 0.75 million cells per mL per degree Plato
- Lagers: 1.5 million cells per mL per degree Plato (double ale rate)
- For a typical 5-gallon (19L) batch at 1.050 OG (~12.5 Plato): ~175 billion cells for ale, ~350 billion for lager
- Underpitching causes: more esters, fusel alcohols, potential stalled fermentation
- Overpitching causes: fast but bland fermentation, less yeast character

### 1.8 Secondary Fermentation / Conditioning
**Purpose**: Clarify beer, allow yeast to clean up off-flavors (diacetyl, acetaldehyde), mature flavors.

- Transfer to secondary vessel (optional; many modern brewers skip this)
- Duration: 1-4 weeks for ales, 4-12 weeks for lagers (lagering)
- Temperature: same as primary or slightly lower
- Lagering: cold condition at 32-38F (0-3C) for 4-8 weeks
- Benefits: improved clarity, smoother flavor, reduced yeast autolysis risk for long conditioning

### 1.9 Cold Crash
**Purpose**: Rapidly drop temperature to precipitate yeast, proteins, and tannins for clarity.

- Temperature: 32-38F (0-3C)
- Duration: 24-72 hours (48 hours is typical)
- Can cause "chill haze" proteins to drop out
- Gelatin fining often added during cold crash (1 tsp per 5 gal dissolved in 150F water)
- Risk: suck-back of airlock liquid due to gas contraction; use a CO2 blowoff or balloon

### 1.10 Carbonation
**Natural Carbonation (Bottle/Keg Conditioning)**:
- Add priming sugar at packaging: typically 3/4 cup (4-5 oz / 113-140g) corn sugar per 5 gallons
- Target CO2 volumes: 2.0-2.6 for most ales; 2.4-2.8 for lagers; 3.0-4.5 for Belgian styles; 1.5-2.0 for English cask ales
- Conditioning time: 2-3 weeks at 68-75F (20-24C)
- Priming sugar calculator: use residual CO2 from fermentation temp + desired volumes

**Forced Carbonation (Kegging)**:
- Set CO2 regulator to target PSI based on temperature and desired volumes
- Quick carb: 30-40 PSI for 24-48 hours, then reduce to serving pressure (10-14 PSI at 38F)
- Slow carb: 10-14 PSI at 38F for 5-7 days (better, more consistent)
- Typical serving: 10-14 PSI at 38F = ~2.4-2.6 volumes CO2

**CO2 Volumes by Style**:
| Style | CO2 Volumes |
|-------|-------------|
| British ales / cask | 1.5-2.0 |
| American ales / IPA | 2.2-2.7 |
| Lagers / Pilsners | 2.4-2.8 |
| German wheat beer | 3.3-4.5 |
| Belgian ales | 3.0-4.5 |
| Stout (nitro) | 1.2-1.5 (with 70/30 N2/CO2 blend) |

### 1.11 Packaging
**Bottling**:
- Sanitize bottles; fill leaving 1 inch headspace
- Cap immediately
- Store upright at 68-75F for carbonation (2-3 weeks)
- Shelf life: 3-6 months for hoppy beers; 1+ year for high-ABV styles

**Kegging**:
- Purge keg with CO2 (3x pressurize to 15 PSI and release)
- Transfer via closed system (siphon or pressure transfer) to minimize oxidation
- Carbonate and serve from keg
- Shelf life: longer due to CO2 blanket; hoppy beers best within 1-2 months

---

## 2. Key Measurements with Exact Ranges

### 2.1 Original Gravity (OG)
| Beer Style | OG Range |
|------------|----------|
| Light Lager | 1.028-1.044 |
| German Pilsner | 1.044-1.050 |
| Czech Pilsner | 1.044-1.060 |
| American Wheat | 1.040-1.055 |
| Hefeweizen | 1.044-1.052 |
| American Pale Ale | 1.045-1.060 |
| English Bitter | 1.032-1.044 |
| Amber Ale | 1.045-1.060 |
| American Brown Ale | 1.045-1.060 |
| American IPA | 1.056-1.070 |
| Hazy/NEIPA | 1.060-1.085 |
| Double IPA | 1.065-1.100 |
| Irish Stout (Dry) | 1.036-1.044 |
| American Stout | 1.050-1.075 |
| Imperial Stout | 1.075-1.115 |
| English Porter | 1.040-1.052 |
| American Porter | 1.050-1.070 |
| Saison | 1.048-1.065 |
| Belgian Tripel | 1.075-1.085 |
| Belgian Dubbel | 1.062-1.075 |
| Belgian Golden Strong | 1.070-1.095 |
| Barleywine | 1.080-1.120 |
| Scottish Export | 1.040-1.060 |
| Kolsch | 1.044-1.050 |

### 2.2 Final Gravity (FG)
| Beer Style | FG Range |
|------------|----------|
| Light Lager | 0.998-1.008 |
| German Pilsner | 1.008-1.013 |
| Czech Pilsner | 1.013-1.017 |
| American Wheat | 1.008-1.013 |
| Hefeweizen | 1.010-1.014 |
| American Pale Ale | 1.010-1.015 |
| English Bitter | 1.007-1.011 |
| Amber Ale | 1.010-1.015 |
| American Brown Ale | 1.010-1.018 |
| American IPA | 1.008-1.014 |
| Hazy/NEIPA | 1.010-1.018 |
| Double IPA | 1.008-1.018 |
| Irish Stout (Dry) | 1.007-1.011 |
| American Stout | 1.010-1.022 |
| Imperial Stout | 1.018-1.030 |
| English Porter | 1.008-1.014 |
| American Porter | 1.012-1.018 |
| Saison | 1.002-1.008 |
| Belgian Tripel | 1.008-1.014 |
| Belgian Dubbel | 1.008-1.018 |
| Belgian Golden Strong | 1.005-1.016 |
| Barleywine | 1.018-1.030 |
| Scottish Export | 1.010-1.016 |
| Kolsch | 1.007-1.011 |

### 2.3 Specific Gravity During Fermentation (Day-by-Day Curve)
The SG drop follows a sigmoid/logarithmic curve: slow start (lag phase), rapid drop (active fermentation), then asymptotic approach to FG.

**General model for a typical ale (OG = 1.060, FG = 1.012)**:
- Gravity points to drop: 60 - 12 = 48 points
- Day 0: 1.060 (pitch)
- Day 1: 1.058-1.055 (lag phase ending, ~5-10% attenuation)
- Day 2: 1.045-1.040 (active fermentation, ~40-50% attenuation)
- Day 3: 1.030-1.025 (peak activity, ~65-75% attenuation)
- Day 4: 1.020-1.018 (slowing, ~85-90% attenuation)
- Day 5: 1.016-1.014 (~92-96% attenuation)
- Day 7: 1.013-1.012 (~98-100% attenuation)
- Day 10: 1.012 (terminal, stable)
- Day 14: 1.012 (confirmed stable)

**Approximate daily attenuation % of total gravity drop**:
| Day | Cumulative % of Total Attenuation | Daily % |
|-----|-----------------------------------|---------|
| 0 | 0% | 0% |
| 1 | 5-10% | 5-10% |
| 2 | 40-50% | 30-40% |
| 3 | 65-75% | 25-30% |
| 4 | 85-90% | 15-20% |
| 5 | 92-96% | 5-8% |
| 7 | 98-100% | 2-4% |
| 10-14 | 100% | 0-1% |

**For a lager (OG = 1.050, FG = 1.010)**:
- Much slower curve; stretch the above timeline by 2-3x
- Day 1-2: minimal drop (extended lag)
- Day 3-5: active fermentation begins
- Day 7-10: ~70-80% attenuation
- Day 14-21: approaching terminal gravity
- Day 21-28: stable FG

### 2.4 pH Measurements
| Stage | pH Range | Notes |
|-------|----------|-------|
| Strike water | 7.0-8.0 | Adjust with acid if needed |
| Mash (target) | 5.2-5.6 | Optimal enzyme activity at 5.2-5.4 |
| Mash (typical) | 5.2-5.8 | Higher with hard water |
| Pre-boil wort | 5.0-5.4 | Drops ~0.1-0.2 from mash |
| Post-boil wort | 4.8-5.2 | Drops another ~0.1-0.3 during boil |
| Early fermentation (Day 1-2) | 4.4-4.8 | Rapid pH drop as yeast produces organic acids |
| Mid fermentation (Day 3-5) | 4.0-4.4 | Continuing to drop |
| Finished beer | 4.0-4.4 | Most ales: 4.1-4.4; most lagers: 4.2-4.6 |
| Sour beer | 3.0-3.5 | Lactobacillus/Pediococcus driven |

**pH drop trajectory**: Wort at ~5.2 drops to ~4.2-4.4 during fermentation. Most of the drop occurs in the first 24-48 hours. Total drop is typically 0.8-1.2 pH units from pre-pitch wort to finished beer.

### 2.5 ABV Calculation
```
ABV = (OG - FG) x 131.25
```
**Example**: OG 1.065, FG 1.012 => (0.065 - 0.012) x 131.25 = 6.95% ABV

**More precise formula (for higher gravity beers)**:
```
ABV = (76.08 x (OG - FG) / (1.775 - OG)) x (FG / 0.794)
```

### 2.6 IBU Ranges by Style
| Style | IBU Range |
|-------|-----------|
| Light Lager | 8-15 |
| German Pilsner | 22-40 |
| Czech Pilsner | 25-45 |
| American Wheat | 15-30 |
| Hefeweizen | 8-15 |
| American Pale Ale | 30-50 |
| English Bitter (Ordinary) | 25-35 |
| Amber Ale | 25-40 |
| American Brown Ale | 20-30 |
| American IPA | 40-70 |
| Hazy/NEIPA | 25-60 |
| Double IPA | 60-100 |
| Irish Stout (Dry) | 25-45 |
| American Stout | 35-75 |
| Imperial Stout | 50-90 |
| English Porter | 18-35 |
| American Porter | 25-50 |
| Saison | 20-35 |
| Belgian Tripel | 20-40 |
| Belgian Dubbel | 15-25 |
| Belgian Golden Strong | 22-35 |
| Barleywine (American) | 50-100 |
| Scottish Export | 15-30 |
| Kolsch | 18-30 |

### 2.7 SRM (Color) Ranges
| Style | SRM Range | Visual Description |
|-------|-----------|-------------------|
| Light Lager | 2-4 | Pale straw |
| German Pilsner | 2-5 | Straw to light gold |
| Czech Pilsner | 3.5-6 | Gold |
| American Wheat | 3-6 | Pale gold |
| Hefeweizen | 3-6 | Pale gold (hazy) |
| American Pale Ale | 5-10 | Gold to light amber |
| English Bitter | 8-14 | Amber |
| Amber Ale | 10-17 | Amber to copper |
| American Brown Ale | 18-35 | Brown |
| American IPA | 6-14 | Gold to amber |
| Hazy/NEIPA | 3-7 | Pale gold (very hazy) |
| Double IPA | 6-14 | Gold to amber |
| Irish Stout (Dry) | 25-40 | Dark brown to black |
| American Stout | 30-40 | Black |
| Imperial Stout | 30-40+ | Opaque black |
| English Porter | 20-30 | Dark brown |
| American Porter | 22-40 | Dark brown to black |
| Saison | 5-14 | Gold to amber |
| Belgian Tripel | 4.5-7 | Deep gold |
| Belgian Dubbel | 10-17 | Amber to dark brown |
| Belgian Golden Strong | 3-6 | Pale gold |
| Barleywine | 10-19 (English) / 10-18 (American) | Amber to deep copper |
| Scottish Export | 13-22 | Amber to brown |
| Kolsch | 3.5-5 | Straw to light gold |

**SRM reference scale**:
- 1-2: Very pale straw
- 3-4: Pale straw to straw
- 5-6: Light gold
- 7-10: Gold to deep gold
- 11-14: Amber
- 15-17: Deep amber to copper
- 18-22: Copper to brown
- 23-30: Brown to dark brown
- 31-40: Very dark brown
- 40+: Black / opaque

### 2.8 Apparent Attenuation
```
Apparent Attenuation (%) = (OG - FG) / (OG - 1.000) x 100
```
**Example**: OG 1.060, FG 1.012 => (0.060 - 0.012) / 0.060 x 100 = 80%

| Range | Description | Typical Styles |
|-------|-------------|----------------|
| 65-70% | Low attenuation | English mild, sweet stout, Scottish ales |
| 70-75% | Medium-low | Porters, brown ales, amber ales |
| 75-80% | Medium (most common) | Pale ales, IPAs, stouts, lagers |
| 80-85% | Medium-high | Belgian ales, dry IPAs |
| 85-90% | High | Saisons, Belgian golden strongs, brut IPA |
| 90-95%+ | Very high | Saisons with special yeast strains (e.g., 3711) |

---

## 3. Yeast Behavior

### 3.1 Lag Phase
- Duration: 6-24 hours after pitching
- What happens: Yeast acclimates to wort, builds cell membrane sterols using dissolved oxygen, takes up nutrients, begins budding
- Observable signs: None initially; slight foam ring may appear at 8-12 hours
- Factors that extend lag phase: underpitching, cold wort, low oxygen, old/unhealthy yeast
- Factors that shorten it: proper pitching rate, good oxygenation (8-10 ppm O2), yeast starter, warm pitching temp

### 3.2 Active (Exponential) Fermentation
- Duration: 2-5 days for ales; 5-14 days for lagers
- Observable signs:
  - Airlock activity: 1 bubble per 1-3 seconds at peak
  - Krausen: thick foam layer 2-6 inches (5-15 cm) on surface
  - Krausen color: tan to brown, may have rocky/craggy appearance
  - Temperature rise: exothermic reaction; wort temp rises 3-8F (2-4C) above ambient without cooling
  - Sulfur smell: normal for lager yeasts (hydrogen sulfide production)
  - Gravity drops rapidly (see section 2.3)

### 3.3 Stationary / Conditioning Phase
- Duration: 2-7 days for ales; weeks for lagers
- Yeast cleans up fermentation byproducts:
  - Diacetyl reabsorbed and reduced to 2,3-butanediol (flavorless)
  - Acetaldehyde converted to ethanol
  - Sulfur compounds scrubbed
- Krausen falls; beer begins to clear
- Gravity stable (within 0.001-0.002 of terminal)

### 3.4 Flocculation
| Level | Behavior | Clarity | Yeast Strains (examples) |
|-------|----------|---------|--------------------------|
| High | Clumps early, drops out quickly (day 3-5) | Very clear beer but may under-attenuate | WLP002 English Ale, WY1968 London ESB |
| Medium | Moderate clumping, drops out over 5-10 days | Good clarity with proper conditioning | WLP001/US-05 American Ale, WY1056 |
| Low | Stays in suspension long (weeks) | Hazy without fining/cold crash | WLP400 Belgian Wit, hefeweizen yeasts, Kveik |

**Impact on simulation**: High-flocculating yeast may stall 2-4 gravity points above expected FG. Rousing (swirling fermenter) can restart. Low-flocculating yeast attenuates fully but requires cold crash or finings for clarity.

### 3.5 Off-Flavors from Temperature and Yeast Stress

| Off-Flavor | Compound | Sensory Description | Cause | Threshold |
|------------|----------|---------------------|-------|-----------|
| Butter/butterscotch | Diacetyl | Slick, buttery mouthfeel; butterscotch aroma | Premature flocculation, cold ferment, weak yeast, bacterial contamination | 10-15 ppb (lager), 10-150 ppb (ale) |
| Green apple / cidery | Acetaldehyde | Sharp, green apple, latex paint | Incomplete fermentation, premature packaging, underpitching, too cold | 5-15 ppm |
| Hot / solvent / spicy | Fusel alcohols (isoamyl, isobutanol, propanol) | Warming, harsh, paint thinner | Too hot fermentation (>75F for ale yeast), underpitching, low nutrients | Varies; >100 ppm total fusels |
| Clove / medicinal | 4-vinyl guaiacol (4VG) | Clove-like phenol | POF+ yeast strains (Belgian, weizen), wild yeast contamination | 0.3-0.5 ppm |
| Banana | Isoamyl acetate | Banana, Juicy Fruit gum | High fermentation temp, underpitching, low-FAN wort, weizen yeast | 1-2 ppm |
| Fruity (general) | Ethyl acetate + other esters | Pear, apple, strawberry | Warm fermentation, certain yeast strains, higher gravity | Varies by ester |
| Solvent | Ethyl acetate (excess) | Nail polish remover | Very high fermentation temp, bacterial infection | >30 ppm |
| Sulfur / rotten egg | Hydrogen sulfide (H2S) | Sulfur, struck match | Normal for lagers; stressed yeast, autolysis | 5-10 ppb |
| Corn / cooked veggie | DMS (dimethyl sulfide) | Sweet corn, canned corn | Inadequate boil, slow cooling, pilsner malt, bacterial | 30-50 ppb |
| Band-aid / smoky | Chlorophenol | Plastic, medicinal | Chlorine/chloramine in water reacting with phenols | 1-5 ppb |
| Cardboard / paper | Trans-2-nonenal | Wet cardboard, papery | Oxidation (O2 exposure during transfer/packaging) | 0.05-0.1 ppb |
| Sour / acidic | Lactic acid / acetic acid | Sour, vinegar | Wild yeast or bacterial infection (Lactobacillus, Acetobacter) | Varies |
| Autolysis / meaty | Various amino acids | Soy sauce, meaty, brothy | Yeast sitting on trub too long (>4-6 weeks), dead yeast | Varies |

**Temperature-specific effects**:
- Fermenting ale yeast **above 72-75F**: Elevated fusel alcohols, excessive esters, potential acetaldehyde
- Fermenting ale yeast **below 58-60F**: Sluggish fermentation, stalled fermentation, elevated diacetyl, elevated acetaldehyde
- Fermenting lager yeast **above 58F**: Ester and fusel production (loses clean lager character)
- Fermenting lager yeast **below 42F**: Very sluggish or dormant; potential incomplete fermentation
- **Temperature swings** (>5F in 24 hrs): Yeast stress, off-flavors, potential stall

---

## 4. Common Problems and Their Indicators

### 4.1 Stalled Fermentation
- **Indicator**: Gravity stops dropping before reaching expected FG; typically stalls at 60-70% apparent attenuation
- **Gravity reading**: Stuck 5-15 points above target FG
- **Common causes**:
  - Temperature too low (yeast dormant)
  - Underpitching
  - High-flocculating yeast dropping out early
  - Wort too high in unfermentable sugars (high mash temp)
  - Yeast nutrient deficiency (low FAN)
  - Excessive alcohol killing yeast (>10% ABV)
- **Remedies**:
  - Warm to upper range of yeast tolerance
  - Rouse yeast (swirl fermenter gently)
  - Pitch fresh, actively fermenting yeast (Krausening)
  - Add yeast nutrient
  - Pitch a more attenuative strain (e.g., champagne yeast for high-gravity)

### 4.2 Infection
- **Indicators**:
  - Unexpected gravity drop below expected FG (wild yeast/Brettanomyces consumes dextrins)
  - Pellicle formation on surface (white/cream film, sometimes bubbly)
  - Sour or vinegary taste
  - Unusual ropey or oily texture
  - Off aromas: band-aid, barnyard, horse blanket, vinegar
- **Gravity behavior**: Can drop to 1.000-1.004 or below if Brett/Lacto present
- **Timeline**: Bacterial infection may take weeks-months to manifest visually; sour taste may appear in days

### 4.3 Oxidation
- **Indicators**:
  - Cardboard/wet paper flavor (trans-2-nonenal)
  - Sherry-like flavors in lighter beers
  - Darkening of beer color (1-3 SRM increase)
  - Loss of hop aroma and flavor
  - Increased haze
- **Causes**: O2 exposure during transfer, splash-filling bottles, headspace air, poor purging
- **Timeline**: Noticeable within days to weeks for hoppy beers; months for malt-forward styles
- **Prevention**: Closed transfers, CO2 purging, minimize splashing, use antioxidants (sodium metabisulfite at 1/4 tsp per 5 gal)

### 4.4 Diacetyl Issues
- **Indicator**: Butter/butterscotch flavor; slick/oily mouthfeel
- **Common in**: Lagers (cold temps slow diacetyl reabsorption), under-attenuated ales, rushed fermentations
- **Remedy**: Diacetyl rest - raise temp to 65-68F (18-20C) for 2-3 days near end of fermentation
- **Forced diacetyl test**: Pull sample, heat to 140F (60C) for 15 min, cool, taste. If butter flavor present, yeast needs more time.
- **Timeline for rest**: Typically performed when fermentation is 2-5 gravity points from terminal

### 4.5 Over-Carbonation / Gushers
- **Indicator**: Bottles gush foam when opened; overcarbonated keg pours
- **Causes**:
  - Too much priming sugar
  - Bottled before fermentation complete (residual fermentable sugar)
  - Wild yeast/bacterial infection (super-attenuates remaining dextrins)
- **Danger**: Bottle bombs (glass explosion) in severe cases
- **Prevention**: Confirm stable FG before bottling; use priming sugar calculator; proper sanitation

### 4.6 Under-Attenuation
- **Indicator**: Beer is sweeter than expected, FG higher than target
- **Causes**: High mash temperature (>158F), poor yeast health, insufficient pitching rate, low wort fermentability
- **Remedy**: Cannot be easily fixed post-fermentation; prevention through proper mash temps and yeast management

---

## 5. Beer Styles and Their Complete Parameter Ranges

### Style Parameter Table (BJCP 2021 Based)

| Style | OG | FG | IBU | SRM | ABV (%) | Attenuation (%) |
|-------|-----|-----|------|------|---------|-----------------|
| American Light Lager | 1.028-1.040 | 0.998-1.008 | 8-12 | 2-3 | 2.8-4.2 | 78-88 |
| American Lager | 1.040-1.050 | 1.004-1.010 | 8-18 | 2-3.5 | 4.2-5.3 | 78-82 |
| German Pilsner | 1.044-1.050 | 1.008-1.013 | 22-40 | 2-5 | 4.4-5.2 | 75-83 |
| Czech Premium Pale Lager | 1.044-1.060 | 1.013-1.017 | 30-45 | 3.5-6 | 4.2-5.8 | 70-78 |
| Munich Helles | 1.044-1.048 | 1.006-1.012 | 16-22 | 3-5 | 4.7-5.4 | 75-86 |
| Hefeweizen | 1.044-1.052 | 1.010-1.014 | 8-15 | 3-6 | 4.3-5.6 | 72-78 |
| American Wheat | 1.040-1.055 | 1.008-1.013 | 15-30 | 3-6 | 4.0-5.5 | 75-80 |
| Kolsch | 1.044-1.050 | 1.007-1.011 | 18-30 | 3.5-5 | 4.4-5.2 | 78-84 |
| English Ordinary Bitter | 1.030-1.039 | 1.007-1.011 | 25-35 | 8-14 | 3.2-3.8 | 72-82 |
| English Best Bitter | 1.040-1.048 | 1.008-1.012 | 25-40 | 8-16 | 3.8-4.6 | 75-80 |
| American Pale Ale | 1.045-1.060 | 1.010-1.015 | 30-50 | 5-10 | 4.5-6.2 | 72-78 |
| American Amber Ale | 1.045-1.060 | 1.010-1.015 | 25-40 | 10-17 | 4.5-6.2 | 73-78 |
| American Brown Ale | 1.045-1.060 | 1.010-1.018 | 20-30 | 18-35 | 4.3-6.2 | 70-78 |
| American IPA | 1.056-1.070 | 1.008-1.014 | 40-70 | 6-14 | 5.5-7.5 | 78-85 |
| Hazy / New England IPA | 1.060-1.085 | 1.010-1.018 | 25-60 | 3-7 | 6.0-9.0 | 76-83 |
| Double IPA | 1.065-1.100 | 1.008-1.018 | 60-100 | 6-14 | 7.5-10.0 | 80-88 |
| English Porter | 1.040-1.052 | 1.008-1.014 | 18-35 | 20-30 | 4.0-5.4 | 72-80 |
| American Porter | 1.050-1.070 | 1.012-1.018 | 25-50 | 22-40 | 4.8-6.5 | 72-78 |
| Irish Stout (Dry) | 1.036-1.044 | 1.007-1.011 | 25-45 | 25-40 | 4.0-4.5 | 73-80 |
| American Stout | 1.050-1.075 | 1.010-1.022 | 35-75 | 30-40 | 5.0-7.0 | 72-80 |
| Imperial Stout | 1.075-1.115 | 1.018-1.030 | 50-90 | 30-40+ | 8.0-12.0 | 72-78 |
| Saison | 1.048-1.065 | 1.002-1.008 | 20-35 | 5-14 | 5.0-7.0 | 85-95 |
| Belgian Dubbel | 1.062-1.075 | 1.008-1.018 | 15-25 | 10-17 | 6.0-7.6 | 76-87 |
| Belgian Tripel | 1.075-1.085 | 1.008-1.014 | 20-40 | 4.5-7 | 7.5-9.5 | 83-89 |
| Belgian Golden Strong | 1.070-1.095 | 1.005-1.016 | 22-35 | 3-6 | 7.5-10.5 | 83-93 |
| American Barleywine | 1.080-1.120 | 1.016-1.030 | 50-100 | 10-18 | 8.0-12.0 | 73-80 |
| Scottish Export | 1.040-1.060 | 1.010-1.016 | 15-30 | 13-22 | 3.9-6.0 | 72-78 |

---

## 6. Day-by-Day Fermentation Log Examples

### 6.1 American IPA (OG 1.065, Target FG 1.012, Yeast: US-05, Ferment Temp: 66F)

| Day | Gravity (SG) | Temp (F) | pH | Airlock (bub/min) | Krausen | Notes |
|-----|-------------|----------|-----|-------------------|---------|-------|
| 0 | 1.065 | 66 | 5.1 | 0 | None | Pitched 2 packs US-05; wort aerated 60 sec with O2 stone |
| 1 | 1.060 | 67 | 4.8 | 5-10 | Thin foam 0.5" | Lag ending; first visible activity at ~14 hrs |
| 2 | 1.042 | 68 | 4.5 | 30-40 | Thick rocky 3-4" | Vigorous fermentation; temp rising from exotherm |
| 3 | 1.028 | 69 | 4.3 | 20-30 | Peak height 4-5" | Peak activity; krausen at highest; strongest sulfur/yeast aroma |
| 4 | 1.020 | 68 | 4.2 | 10-15 | Receding 2-3" | Slowing; krausen falling; less turbid |
| 5 | 1.016 | 67 | 4.1 | 5-8 | Thin 1" | Most fermentation complete |
| 7 | 1.013 | 66 | 4.1 | 1-2 | Ring only | Approaching terminal; add dry hops |
| 10 | 1.012 | 66 | 4.1 | <1 | None | Stable gravity reading #1 |
| 12 | 1.012 | 66 | 4.1 | 0 | None | Stable gravity reading #2; remove dry hops |
| 14 | 1.012 | 66 | 4.1 | 0 | None | Confirmed FG; cold crash to 34F |
| 16 | 1.012 | 34 | 4.1 | 0 | None | Cold crash day 2; add gelatin finings |
| 18 | 1.012 | 34 | 4.1 | 0 | None | Package to keg; begin force carbonation |

### 6.2 German Pilsner (OG 1.048, Target FG 1.010, Yeast: W-34/70, Ferment Temp: 50F)

| Day | Gravity (SG) | Temp (F) | pH | Notes |
|-----|-------------|----------|-----|-------|
| 0 | 1.048 | 50 | 5.2 | Pitched 2L starter of W-34/70 |
| 1 | 1.047 | 50 | 5.0 | Minimal activity |
| 2 | 1.045 | 50 | 4.9 | Light airlock activity begins |
| 3 | 1.040 | 51 | 4.7 | Steady fermentation; thin krausen |
| 5 | 1.030 | 51 | 4.5 | Active but controlled |
| 7 | 1.022 | 51 | 4.3 | Sulfur aroma (normal for lager) |
| 10 | 1.016 | 51 | 4.2 | Slowing |
| 14 | 1.012 | 51 | 4.1 | Near terminal |
| 16 | 1.011 | 65 | 4.1 | Diacetyl rest begins (raise to 65F over 24 hrs) |
| 19 | 1.010 | 65 | 4.1 | Diacetyl rest complete; forced diacetyl test passes |
| 20 | 1.010 | 50 | 4.1 | Begin crash cooling to lagering temp |
| 21 | 1.010 | 34 | 4.1 | Lagering begins |
| 49 | 1.010 | 34 | 4.1 | 4 weeks lagering complete; package |

### 6.3 Belgian Tripel (OG 1.080, Target FG 1.010, Yeast: WLP530, Start Temp: 65F ramp to 75F)

| Day | Gravity (SG) | Temp (F) | pH | Notes |
|-----|-------------|----------|-----|-------|
| 0 | 1.080 | 65 | 5.1 | Large starter (3L) pitched |
| 1 | 1.075 | 66 | 4.8 | Slow start; Belgian yeasts lag longer |
| 2 | 1.060 | 67 | 4.5 | Fermentation ramps; raise ambient 1F/day |
| 3 | 1.045 | 68 | 4.3 | Vigorous; fruity/spicy aroma develops |
| 4 | 1.035 | 70 | 4.2 | Continued rise; esters and phenols prominent |
| 5 | 1.025 | 72 | 4.1 | Allow free rise |
| 7 | 1.018 | 75 | 4.0 | Target high temp reached; hold here |
| 10 | 1.013 | 75 | 4.0 | Slowing but still active |
| 14 | 1.011 | 75 | 4.0 | Near terminal |
| 18 | 1.010 | 75 | 4.0 | Stable; begin cooling |
| 21 | 1.010 | 65 | 4.0 | Condition for 1-2 weeks before packaging |

---

## 7. Equipment Variables

### 7.1 Fermenter Types
| Type | Material | Pros | Cons | Cost Range |
|------|----------|------|------|------------|
| Plastic bucket (6.5 gal) | HDPE | Cheap ($15-25), easy to clean, wide opening | Oxygen permeable, scratches harbor bacteria, short lifespan | $15-25 |
| Glass carboy (5-6.5 gal) | Glass | Non-permeable, easy to see fermentation, long lifespan | Heavy (12-15 lbs empty), breakable (dangerous), hard to clean | $30-50 |
| PET carboy (5-6.5 gal) | PET plastic | Lighter than glass, see-through, less fragile | Slight O2 permeability, scratches over time | $25-40 |
| Stainless conical (7-14 gal) | 304 SS | Dump yeast/trub from bottom valve, no transfer needed, durable | Expensive, heavy, requires stand | $200-600 |
| Unitank (pressurized conical) | 304 SS | All-in-one ferment + carbonate + serve; pressure fermentation | Very expensive, complex, overkill for beginners | $300-1000 |
| Fermonster (7 gal) | PET wide-mouth | Wide opening, see-through, lightweight | Some O2 permeability | $25-35 |
| SS Brewtech Brew Bucket | 304 SS | Affordable stainless, good seals, thermowell port | No conical bottom (no yeast dump) | $150-250 |

### 7.2 Temperature Control Methods
| Method | Temp Control Range | Precision | Cost | Notes |
|--------|-------------------|-----------|------|-------|
| None (ambient) | Room temp +/- | +/- 5-10F | $0 | Unpredictable; works only in temperate climates |
| Swamp cooler (water bath + ice) | Room temp -5 to -10F | +/- 2-4F | $5-20 | Cheap; requires frequent ice changes; cannot heat |
| Wet T-shirt + fan (evaporative) | Room temp -5 to -15F | +/- 3-5F | $5-10 | Works in low humidity; no heating |
| Fermentation chamber (chest freezer + temp controller) | 32-80F | +/- 1F | $150-300 | Gold standard for homebrewers; Inkbird or STC-1000 controller |
| Glycol chiller | 28-75F | +/- 0.5F | $500-3000 | Commercial-grade; can cool multiple fermenters |
| Fermwrap heater + controller | Ambient to +20F | +/- 1-2F | $30-60 | Heating only; good for lagers in cold environments |
| SS Brewtech FTSs | 32-80F | +/- 1F | $200-400 | Thermoelectric; fits on Brew Bucket or conical |

### 7.3 Carbonation Methods
| Method | Time to Carbonate | Equipment | Precision | Notes |
|--------|-------------------|-----------|-----------|-------|
| Bottle conditioning (priming sugar) | 2-3 weeks | Bottles, caps, capper, priming sugar | Moderate | Slight yeast sediment in bottles; natural but variable |
| Keg force carb (set and forget) | 5-7 days | Corny keg, CO2 tank, regulator | High | Set to serving pressure at fridge temp; consistent |
| Keg force carb (burst/quick) | 24-48 hours | Same as above | Moderate | 30-40 PSI for 24-48 hrs then reduce; risk of over-carb |
| Keg force carb (shake method) | 15-30 minutes | Same as above | Low | Shake keg at 30 PSI; very fast but imprecise and foamy |
| Spunding valve (pressure fermentation) | During fermentation | Spunding valve, pressure-rated fermenter | High | Capture natural CO2; adjustable; very efficient |

---

## 8. Dry Hopping

### 8.1 Timing and Methods
| Method | When Added | Duration | Purpose |
|--------|-----------|----------|---------|
| Traditional dry hop | After fermentation complete (terminal gravity) | 3-7 days | Maximum aroma; minimal biotransformation |
| Biotransformation dry hop | During active fermentation (24-72 hrs after pitch, or at 50% attenuation) | 3-5 days | Yeast converts hop compounds; tropical/juicy character |
| Keg hop | In keg, at packaging | Ongoing (remove bag after 5-7 days or leave) | Fresh hop aroma; risk of grassy flavors if too long |
| Double dry hop (DDH) | Two additions: one during fermentation + one post-fermentation | 3-5 days each | Maximum aroma intensity; common in hazy IPAs |
| Hop torpedo / hop rancher | Post-fermentation, recirculating | 1-3 days | Controlled extraction; easy removal |

### 8.2 Amounts
| Style | Dry Hop Rate (oz per 5 gal) | Dry Hop Rate (g/L) |
|-------|------------------------------|---------------------|
| American Pale Ale | 1-2 oz | 1.5-3 g/L |
| American IPA | 2-4 oz | 3-6 g/L |
| Double IPA | 4-8 oz | 6-12 g/L |
| Hazy/NEIPA | 4-10 oz | 6-16 g/L |
| West Coast IPA | 2-4 oz | 3-6 g/L |

### 8.3 Impact on Gravity Readings
- Dry hops can cause a **1-3 gravity point drop** that is NOT from fermentation
- Mechanism: Hop material absorbs water and releases hop oils/sugars, changing the refractometric and hydrometric readings slightly
- Hops can also nucleate dissolved CO2, giving false high readings on hydrometers
- **Simulation note**: After dry hop addition, expect a small apparent gravity drop of 0.001-0.003 that should not be counted as attenuation

### 8.4 Aroma Compounds from Dry Hopping
| Compound | Aroma | Origin | Notes |
|----------|-------|--------|-------|
| Myrcene | Resinous, herbal, green | Direct extraction from hops | Most abundant hop oil; volatile |
| Linalool | Floral, citrus, lavender | Direct extraction | Survives well; key aroma compound |
| Geraniol | Rose, floral, geranium | Direct extraction | Converted to citronellol by yeast (biotransformation) |
| Citronellol | Citrus, lemon | Yeast biotransformation of geraniol | Only produced when yeast is active during dry hop |
| Humulene | Woody, spicy, herbal | Direct extraction | Noble hop character |
| Beta-pinene | Pine, resinous | Direct extraction | Piney hop character |
| 4-mercapto-4-methylpentan-2-one (4MMP) | Tropical, blackcurrant | Yeast biotransformation of thiol precursors | Key to tropical NEIPA character; requires specific yeast or enzyme (Phantasm) |
| 3-mercaptohexan-1-ol (3MH) | Grapefruit, passion fruit | Yeast biotransformation | Released from bound thiol precursors by beta-lyase active yeast |

### 8.5 Biotransformation Details
- **Definition**: The biochemical conversion of hop-derived compounds by actively fermenting yeast into new aromatic compounds
- **Key reaction**: Geraniol (floral) -> Citronellol (citrus) via yeast reductase enzyme
- **Thiol release**: Certain yeast strains with IRC7 gene activity can cleave thiol precursors (bound in hop material) into free thiols with intense tropical aromas (passion fruit, grapefruit)
- **Optimal timing**: Add dry hops when gravity has dropped 30-50% of the way from OG to FG (typically 24-48 hours into fermentation for a standard ale)
- **CO2 scrubbing concern**: Active fermentation produces CO2 which can strip volatile hop aromas. Biotransformation proponents argue the new compounds formed compensate for this loss.
- **Evidence quality**: Mixed. Controlled experiments (Brulosophy) show inconsistent results in blind triangle tests. However, many professional brewers report meaningful flavor differences.

### 8.6 Hop Creep
- **Definition**: Enzymes naturally present in hops (amyloglucosidase, limit dextrinase) can break down dextrins into fermentable sugars
- **Effect**: Beer continues to ferment slowly after dry hopping, potentially over-attenuating and over-carbonating
- **Gravity impact**: Can cause FG to drop 2-6 additional points over 1-2 weeks after dry hop
- **Danger**: Over-carbonation in bottles (bottle bombs); flavor changes
- **Prevention**: Use shorter dry hop contact time (3-5 days max); cold crash after dry hop; be aware when packaging that gravity may still be slowly dropping

---

## 9. Water Chemistry (Brief, Relevant to Simulation)

### 9.1 Key Ions and Their Effects
| Ion | Effect on Beer | Typical Range (ppm) | Notes |
|-----|---------------|---------------------|-------|
| Calcium (Ca2+) | Promotes enzyme activity, yeast health, clarity | 50-150 | Target 50+ for most styles |
| Magnesium (Mg2+) | Yeast nutrient; harsh/astringent at high levels | 0-30 | Important but modest amounts needed |
| Sodium (Na+) | Fullness/body at low levels; salty/harsh at high | 0-150 | <75 ppm preferred |
| Sulfate (SO4) | Enhances hop bitterness perception; dry, crisp | 0-350 | High in hop-forward styles (Burton water: 600+) |
| Chloride (Cl-) | Enhances malt fullness and sweetness | 0-250 | High in malt-forward styles |
| Bicarbonate (HCO3-) | Raises pH; needed for dark malts | 0-250 | Dark beers need more; pale beers need less |

### 9.2 Sulfate-to-Chloride Ratio
| Ratio (SO4:Cl) | Effect | Typical Styles |
|-----------------|--------|----------------|
| 2:1 to 3:1+ | Hop-forward, dry, bitter emphasis | IPA, West Coast IPA, DIPA |
| 1:1 | Balanced | Pale ale, amber, porter |
| 1:2 to 1:3+ | Malt-forward, round, full | Stout, brown ale, Scottish ale, NEIPA |

---

## 10. Simulation-Relevant State Variables Summary

For a batch state simulator, track these variables over time:

### Per-Batch State
```
batch_id: str
style: str
volume_gallons: float          # Typically 5.0-5.5 for homebrew
grain_bill: list[dict]         # {grain: str, weight_lbs: float, color_lovibond: float}
hop_schedule: list[dict]       # {hop: str, amount_oz: float, alpha_acid_pct: float, time_min: int, type: "bittering"|"flavor"|"aroma"|"dryhop"}
yeast_strain: str
yeast_cells_pitched: int       # in billions
water_profile: dict            # {Ca, Mg, Na, SO4, Cl, HCO3 in ppm}
mash_temp_f: float
mash_duration_min: int
boil_duration_min: int
fermentation_temp_f: float
```

### Time-Series State (per reading)
```
timestamp: datetime
day: int                       # Days since pitch
stage: str                     # "mashing"|"lautering"|"boiling"|"cooling"|"fermentation"|"conditioning"|"coldcrash"|"packaging"
gravity: float                 # 1.000-1.120
temperature_f: float           # 32-95
ph: float                      # 3.0-5.8
airlock_bubbles_per_min: float # 0-60
krausen_inches: float          # 0-6
attenuation_pct: float         # 0-100
abv_current: float             # calculated from OG and current gravity
co2_volumes: float             # 0-4.5
notes: str
```

### Calculated Fields
```
og: float                      # First gravity reading post-pitch
fg: float                      # Final stable gravity
abv: float                     # (OG - FG) * 131.25
apparent_attenuation: float    # (OG - FG) / (OG - 1.000) * 100
ibu: float                     # Tinseth or Rager formula
srm: float                     # Sum of MCU per grain, then Morey formula: SRM = 1.4922 * (MCU ^ 0.6859)
calories_per_12oz: float       # ~((6.9 * ABW + 4.0 * (RE - 0.1)) * FG * 3.55)
```

### Gravity Curve Model (Sigmoid Approximation)
For simulation, the fermentation gravity curve can be modeled as:
```
SG(t) = FG + (OG - FG) * exp(-k * (t - lag)^n)

Where:
  t = time in hours since pitch
  lag = lag phase duration in hours (6-24, depending on pitch rate and temp)
  k = fermentation rate constant (0.02-0.08 for ales; 0.005-0.02 for lagers)
  n = curve shape parameter (~1.2-1.8; higher = sharper transition)

Alternative (logistic model):
  attenuation(t) = A_max / (1 + exp(-r * (t - t_half)))

Where:
  A_max = maximum attenuation (e.g., 0.80 for 80%)
  r = rate parameter (0.3-0.8 per day for ales; 0.1-0.3 for lagers)
  t_half = time to 50% attenuation in days (1.5-2.5 for ales; 5-8 for lagers)
  SG(t) = OG - (OG - 1.000) * attenuation(t)
```

### pH Curve Model
```
pH(t) = pH_final + (pH_initial - pH_final) * exp(-k_pH * t)

Where:
  pH_initial = 5.0-5.2 (post-boil wort)
  pH_final = 4.0-4.4
  k_pH = 0.5-1.5 per day (rapid drop in first 24-48 hours)
  t = days since pitch
```

### Temperature Model (Exothermic Fermentation)
```
T_beer(t) = T_ambient + delta_T * fermentation_rate(t) / max_fermentation_rate

Where:
  delta_T = 3-8F for ales without cooling; 0-2F with temperature control
  fermentation_rate(t) = -dSG/dt (derivative of gravity curve)
```

---

## Sources and References

- BJCP 2021 Style Guidelines: https://www.bjcp.org/style/2021/beer/
- How to Brew (John Palmer): https://realbeer.com/jjpalmer/
- Braukaiser Mashing Theory: https://www.braukaiser.com/wiki/index.php/The_Theory_of_Mashing
- American Homebrewers Association Enzyme Guide: https://homebrewersassociation.org/how-to-brew/enzymes-in-beer-whats-happening-in-the-mash/
- BeerSmith Mash Enzymes: https://beersmith.com/blog/2020/03/17/enzymes-in-the-mash-and-mash-temperatures-for-beer-brewing/
- Escarpment Labs Off-Flavors Guide: https://escarpmentlabs.com/en-us/blogs/resources/5-off-flavours-beer-yeast
- Scott Janish Dry Hopping Research: https://scottjanish.com/a-case-for-short-and-cool-dry-hopping/
- Brulosophy Biotransformation Experiments: https://brulosophy.com/2017/01/23/biotransformation-vs-standard-dry-hop-exbeeriment-results/
- Precision Fermentation Gravity Curves: https://www.precisionfermentation.com/blog/brewmonitor-fermentation-data-curves-gravity-examples/
- MoreBeer pH Guide: https://www.morebeer.com/articles/checking_ph_of_beer
- Brewing Forward pH Reference: https://brewingforward.com/wiki/Brewing_pH
- Craft Master Stainless Fermentation Stages: https://www.craftmasterstainless.com/blog/2023/4/7/a-guide-to-the-stages-of-beer-fermentation
- Grainfather Fermentation Guide: https://us.grainfather.com/a/blog/master-homebrew-fermentation-complete-guide-for-beer-makers
