# Civilization VI -- Domain Mechanics Reference

## Overview

Civilization VI is a turn-based 4X strategy game where players lead a civilization from the Ancient Era to the Information Era. Players compete for victory through Science, Culture, Domination, Religion, Diplomacy, or Score. The game features **50 playable civilizations** with **67 leaders** (including all DLC/expansions).

---

## All Civilizations (50 Total)

### Base Game (20)
America (Teddy Roosevelt), Arabia (Saladin), Aztec (Montezuma), Brazil (Pedro II), China (Qin Shi Huang), Egypt (Cleopatra), England (Victoria), France (Catherine de Medici), Germany (Frederick Barbarossa), Greece (Pericles / Gorgo), India (Gandhi), Japan (Hojo Tokimune), Kongo (Mvemba a Nzinga), Norway (Harald Hardrada), Poland (Jadwiga), Rome (Trajan), Russia (Peter), Scythia (Tomyris), Spain (Philip II), Sumeria (Gilgamesh)

### DLC Packs (6)
Australia (John Curtin), Persia (Cyrus), Macedon (Alexander), Nubia (Amanitore), Khmer (Jayavarman VII), Indonesia (Gitarja)

### Rise and Fall Expansion (9)
Cree (Poundmaker), Georgia (Tamar), Korea (Seondeok), Mapuche (Lautaro), Mongolia (Genghis Khan), Netherlands (Wilhelmina), Scotland (Robert the Bruce), Zulu (Shaka), India (Chandragupta -- alternate leader)

### Gathering Storm Expansion (9)
Canada (Wilfrid Laurier), Hungary (Matthias Corvinus), Inca (Pachacuti), Mali (Mansa Musa), Maori (Kupe), Ottoman (Suleiman), Phoenicia (Dido), Sweden (Kristina), England (Eleanor of Aquitaine -- alternate leader), France (Eleanor of Aquitaine -- alternate leader)

### New Frontier Pass (6)
Maya (Lady Six Sky), Gran Colombia (Simon Bolivar), Ethiopia (Menelik II), Byzantium (Basil II), Gaul (Ambiorix), Vietnam (Ba Trieu), Babylon (Hammurabi), Portugal (Joao III), Kublai Khan (alternate leader for Mongolia/China)

---

## Era Progression

| Era | Approximate Turns (Standard) | Key Milestones |
|-----|------------------------------|----------------|
| Ancient | 1-50 | First cities, warriors, early wonders |
| Classical | 50-100 | Districts, first governments, early religion |
| Medieval | 100-150 | Knights, universities, mid-game expansion |
| Renaissance | 150-200 | Gunpowder, exploration, banking |
| Industrial | 200-250 | Factories, railroads, nationalism |
| Modern | 250-300 | Tanks, ideology, tourism |
| Atomic | 300-340 | Nuclear weapons, airports |
| Information | 340-500 | Space race, internet, endgame |
| Future (GS) | 340+ | Giant Death Robot, futuristic governments |

---

## Resource Types

### Yield Types (Core Resources)

| Yield | Icon | Primary Source | Used For |
|-------|------|---------------|----------|
| Food | Grain | Farms, pastures, fishing boats | City growth (population) |
| Production | Hammer | Mines, lumber mills, industrial zones | Building/unit construction |
| Gold | Coin | Commercial hubs, trade routes, harbors | Purchasing, maintenance |
| Science | Flask | Campuses, libraries, universities | Technology research |
| Culture | Lyre | Theater squares, monuments, wonders | Civic research, border expansion |
| Faith | Dove | Holy sites, shrines, temples | Religious units, purchases |
| Tourism | Suitcase | Great works, wonders, national parks | Culture victory |

### Strategic Resources
- **Ancient**: Horses, Iron
- **Industrial**: Niter, Coal
- **Modern**: Oil, Aluminum, Uranium

### Luxury Resources (examples)
Amber, Citrus, Cocoa, Coffee, Cosmetics, Cotton, Dyes, Diamonds, Furs, Gypsum, Incense, Ivory, Jade, Jeans, Marble, Mercury, Olives, Pearls, Perfume, Salt, Silk, Silver, Spices, Sugar, Tea, Tobacco, Toys, Truffles, Whales, Wine

### Bonus Resources
Bananas, Cattle, Copper, Crabs, Deer, Fish, Maize, Rice, Sheep, Stone, Wheat

---

## District Types

### Specialty Districts (count toward district cap)

| District | Yield | Adjacency Bonuses | Key Buildings |
|----------|-------|-------------------|---------------|
| **Campus** | Science | +1 per adjacent Mountain, +1 per 2 Rainforest | Library (+2 Science), University (+4 Science), Research Lab (+5 Science) |
| **Holy Site** | Faith | +1 per adjacent Natural Wonder, +1 per 2 adjacent Woods/Mountain | Shrine (+2 Faith), Temple (+4 Faith), Worship building |
| **Theater Square** | Culture | +1 per adjacent Wonder, +1 per 2 adjacent districts | Amphitheater (+2 Culture), Museum (+2 Culture), Broadcast Center (+4 Culture) |
| **Commercial Hub** | Gold | +2 per adjacent River, +2 per adjacent Harbor | Market (+3 Gold), Bank (+5 Gold), Stock Exchange (+7 Gold) |
| **Harbor** | Gold | +1 per adjacent sea resource, +2 per adjacent City Center | Lighthouse (+1 Food, +1 Gold), Shipyard (+2 Production), Seaport (+2 Gold, +2 Food) |
| **Industrial Zone** | Production | +1 per adjacent Mine/Quarry, +1 per 2 adjacent districts | Workshop (+2 Production), Factory (+3 Production), Power Plant (+4 Production) |
| **Entertainment Complex** | Amenities | None significant | Arena (+1 Amenity), Zoo (+1 Amenity to city and within 6 tiles), Stadium (+2 Amenities) |
| **Encampment** | Military | None | Barracks (+1 Production, +25% melee XP), Armory (+1 Production), Military Academy (+1 Production) |
| **Aerodrome** | Air units | None | Hangar (+2 Production), Airport (+2 Production, +1 Tourism per Culture) |
| **Spaceport** | Victory | None | Required for Science Victory projects |
| **Water Park** (GS) | Amenities | None | Ferris Wheel (+1 Amenity), Aquarium (+1 Amenity to nearby), Aquatics Center (+2 Amenity) |

### Non-Specialty Districts (do not count toward cap)

| District | Purpose | Notes |
|----------|---------|-------|
| City Center | Core | Automatically placed, contains Palace in capital |
| Aqueduct | Housing | +2 to +6 Housing depending on water source |
| Dam (GS) | Flood protection | Prevents flood damage, enables hydroelectric power |
| Canal (GS) | Naval passage | Allows ships through land tiles |
| Neighborhood | Housing | +2 to +6 Housing based on Appeal |
| Government Plaza (GS) | Government | +1 to all yields; unique buildings per tier |
| Diplomatic Quarter (GS) | Diplomacy | Consulate, Chancery |

### District Limit Formula
```
District cap = floor(Population / 3) + 1
```
- Pop 1-3: 2 districts
- Pop 4-6: 3 districts
- Pop 7-9: 4 districts
- Pop 10-12: 5 districts

---

## Wonder List (30+ Key Wonders by Era)

### Ancient Era Wonders
| Wonder | Cost | Requirement | Effect |
|--------|------|-------------|--------|
| Stonehenge | 180 | Adjacent to Stone | Free Great Prophet; +2 Faith |
| Great Bath | 180 | Floodplains | Faith from floods; +3 Faith, +1 Food |
| Hanging Gardens | 180 | Adjacent to River | +15% city growth |
| Pyramids | 220 | Desert/Floodplains | Free Builder; builders +1 charge |
| Temple of Artemis | 180 | Adjacent to Camp | +4 Food; +1 Amenity per Camp/Pasture/Plantation within 4 tiles |
| Oracle | 290 | Hills | +1 Culture, +1 Faith; Great People patronage -25% Faith |
| Etemenanki | 220 | Floodplains/Marsh | +2 Science, +1 Production per Floodplains/Marsh |

### Classical Era Wonders
| Wonder | Cost | Requirement | Effect |
|--------|------|-------------|--------|
| Colosseum | 400 | Adjacent to Entertainment Complex | +3 Culture, +3 Loyalty to cities within 6 tiles |
| Great Library | 400 | Adjacent to Campus | +2 Science; boosts Ancient/Classical techs |
| Petra | 400 | Desert (non-floodplain) | +2 Food, +2 Gold, +1 Production to Desert tiles |
| Colossus | 400 | Harbor; Coast | +3 Gold, +1 Trade Route, +1 Great Admiral point |
| Great Lighthouse | 290 | Harbor; Coast | +3 Gold; +1 Movement for naval units |
| Mahabodhi Temple | 400 | Holy Site with Temple; Woods | +4 Faith; 2 free Apostles |
| Jebel Barkal | 400 | Desert Hills | +4 Faith to city centers within 6 tiles |
| Apadana | 400 | Capital | +2 Great Work slots; +2 Envoys per wonder built |

### Medieval Era Wonders
| Wonder | Cost | Requirement | Effect |
|--------|------|-------------|--------|
| Hagia Sophia | 710 | Holy Site | +4 Faith; Missionaries/Apostles +1 spread charge |
| Kilwa Kisiwani | 710 | Flat coast | +15% yield bonus for being Suzerain of city-states |
| Mont St. Michel | 710 | Floodplains/Marsh | Apostles gain Martyr promotion; +2 Faith, +2 Relic slots |
| University of Sankore | 710 | Desert; adjacent to Campus | +3 Science, +1 Faith, +1 Gold per trade route |
| Meenakshi Temple | 710 | Holy Site | +2 Faith; Gurus +1 spread charge |
| Angkor Wat | 710 | Aqueduct | +1 Housing, +1 Population in all cities |

### Renaissance Era Wonders
| Wonder | Cost | Requirement | Effect |
|--------|------|-------------|--------|
| Forbidden City | 920 | Flat land | +1 Wildcard policy slot; +5 Culture |
| Potala Palace | 920 | Hills; adjacent to Mountain | +1 Diplomatic policy slot; +3 Culture, +2 Faith |
| Venetian Arsenal | 920 | Coast; Industrial Zone | Double naval unit production |
| Taj Mahal | 920 | Adjacent to River | +1 Era Score from Historic Moments |
| St. Basil's Cathedral | 920 | Adjacent to City Center; Tundra | +100% Tourism/Culture from Relics/Artifacts; +3 Relic slots |

### Industrial Era Wonders
| Wonder | Cost | Requirement | Effect |
|--------|------|-------------|--------|
| Ruhr Valley | 1240 | Along River; Industrial Zone | +20% Production in city; +1 Production per Mine/Quarry |
| Big Ben | 1240 | Adjacent to River; Commercial Hub | Double current Treasury; +1 Economic policy slot; +6 Gold |
| Oxford University | 1240 | Adjacent to Campus | +20% Science in city; 2 free techs; +3 Great Scientist points |
| Hermitage | 1240 | Adjacent to River | +3 Culture; 4 Great Work of Art slots |
| Eiffel Tower | 1240 | Flat land | +2 Appeal to all tiles in your empire |

### Modern/Atomic/Information Era Wonders
| Wonder | Cost | Requirement | Effect |
|--------|------|-------------|--------|
| Statue of Liberty | 1740 | Coast; Harbor | +4 Diplomatic Favor per turn; settlers gain +100% movement |
| Sydney Opera House | 1740 | Coast; adjacent to Harbor | +8 Culture; 3 Great Music slots |
| Cristo Redentor | 1740 | Hills | +4 Culture; Seaside Resorts provide +100% Tourism |
| Biosphere | 1740 | Along River | +1 Power from Geothermal/Solar/Wind; bonus resources give Tourism |
| Amundsen-Scott | 1740 | Snow tile | +20% Science for city; +5 Great Scientist points |
| Estadio do Maracana | 2835 | Adjacent to Entertainment Complex | +6 Culture; +2 Amenities to all cities |

---

## Technology Tree

### Ancient Era (11 technologies)
Pottery, Animal Husbandry, Mining, Sailing, Astrology, Irrigation, Writing, Archery, Masonry, Bronze Working, The Wheel

### Classical Era (8 technologies)
Celestial Navigation, Currency, Horseback Riding, Iron Working, Shipbuilding, Mathematics, Construction, Engineering

### Medieval Era (7 technologies)
Military Tactics, Apprenticeship, Stirrups, Machinery, Education, Military Engineering, Castles

### Renaissance Era (9 technologies)
Cartography, Mass Production, Banking, Gunpowder, Printing, Square Rigging, Astronomy, Metal Casting, Siege Tactics

### Industrial Era (7 technologies)
Industrialization, Scientific Theory, Ballistics, Military Science, Steam Power, Sanitation, Economics

### Modern Era (7 technologies)
Flight, Radio, Chemistry, Combustion, Steel, Electricity, Replaceable Parts

### Atomic Era (6 technologies)
Advanced Flight, Rocketry, Advanced Ballistics, Combined Arms, Plastics, Computers, Nuclear Fission

### Information Era (7 technologies)
Telecommunications, Satellites, Guidance Systems, Lasers, Composites, Stealth Technology, Robotics, Nuclear Fusion, Nanotechnology

### Future Era (GS) (4+ technologies)
Smart Materials, Predictive Systems, Seasteads, Advanced AI, Advanced Power Cells, Cybernetics, Offworld Mission

---

## Civic Tree

### Ancient Civics
Code of Laws, Craftsmanship, Foreign Trade, Early Empire, Mysticism, Military Tradition, State Workforce, Political Philosophy

### Classical Civics
Games and Recreation, Drama and Poetry, Theology, Defensive Tactics, Military Training, Recorded History, Naval Tradition

### Medieval Civics
Medieval Faires, Guilds, Mercenaries, Civil Engineering, Feudalism, Divine Right, Exploration

### Renaissance Civics
Humanism, Diplomatic Service, Reformed Church, Mercantilism, The Enlightenment

### Industrial Civics
Colonialism, Civil Engineering, Nationalism, Opera and Ballet, Natural History, Urbanization, Scorched Earth

### Modern/Atomic/Information Civics
Conservation, Mass Media, Capitalism, Ideology, Nuclear Program, Suffrage, Totalitarianism, Class Struggle, Cultural Heritage, Cold War, Professional Sports, Rapid Deployment, Space Race, Globalization, Social Media, Near Future Governance, Venture Politics, Distributed Sovereignty, Optimization Imperative

---

## Government Types

### Tier 1 (Ancient/Classical)

| Government | Unlocked By | Policy Slots | Legacy Bonus |
|------------|------------|--------------|--------------|
| Chiefdom | Code of Laws | 1 Military, 1 Economic | None |
| Autocracy | Political Philosophy | 2 Military, 1 Economic, 1 Wildcard | +1 to all yields in Capital |
| Oligarchy | Political Philosophy | 1 Military, 1 Economic, 1 Diplomatic, 1 Wildcard | +4 Combat Strength for melee/anti-cavalry |
| Classical Republic | Political Philosophy | 0 Military, 2 Economic, 1 Diplomatic, 1 Wildcard | +1 Great People points/district |

### Tier 2 (Medieval/Renaissance)

| Government | Policy Slots | Bonus |
|------------|--------------|-------|
| Monarchy | 1 Military, 1 Economic, 2 Diplomatic, 1 Wildcard | +50% influence points toward city-states |
| Theocracy | 2 Military, 2 Economic, 1 Diplomatic, 1 Wildcard | Can purchase land units with Faith; +5 combat for theological units |
| Merchant Republic | 1 Military, 2 Economic, 1 Diplomatic, 2 Wildcard | +2 Trade Route capacity |

### Tier 3 (Modern)

| Government | Policy Slots | Bonus |
|------------|--------------|-------|
| Democracy | 1 Military, 3 Economic, 2 Diplomatic, 2 Wildcard | +50% Great People points from patronage; district projects +100% |
| Communism | 3 Military, 1 Economic, 2 Diplomatic, 2 Wildcard | +10% Production toward all buildings |
| Fascism | 4 Military, 1 Economic, 1 Diplomatic, 2 Wildcard | +5 combat strength to all military units; +50% unit production |

### Tier 4 (Future, Gathering Storm)

| Government | Policy Slots | Bonus |
|------------|--------------|-------|
| Corporate Libertarianism | 2 Military, 1 Economic, 2 Diplomatic, 5 Wildcard | +4 Gold per specialty district |
| Digital Democracy | 1 Military, 3 Economic, 1 Diplomatic, 5 Wildcard | +1 Trade Route; builder +2 charges |
| Synthetic Technocracy | 1 Military, 1 Economic, 3 Diplomatic, 5 Wildcard | +20% Science; +10% Production toward Space Race |

---

## Military Units with Combat Strength

### Melee Line
| Unit | Era | Combat Strength | Production Cost | Upgrade From |
|------|-----|----------------|----------------|--------------|
| Warrior | Ancient | 20 | 40 | -- |
| Swordsman | Classical | 36 | 90 | Warrior |
| Man-at-Arms | Medieval | 45 | 160 | Swordsman |
| Musketman | Renaissance | 55 | 240 | Man-at-Arms |
| Line Infantry | Industrial | 65 | 360 | Musketman |
| Infantry | Modern | 70 | 430 | Line Infantry |
| Mechanized Infantry | Atomic | 85 | 520 | Infantry |

### Anti-Cavalry Line
| Unit | Era | Combat Strength | Production Cost |
|------|-----|----------------|----------------|
| Spearman | Ancient | 25 | 65 |
| Pikeman | Medieval | 41 | 200 |
| Pike and Shot | Renaissance | 55 | 250 |
| AT Crew | Modern | 75 | 400 |

### Ranged Line
| Unit | Era | Ranged Strength | Range | Production Cost |
|------|-----|----------------|-------|----------------|
| Slinger | Ancient | 15 | 1 | 35 |
| Archer | Ancient | 25 | 2 | 60 |
| Crossbowman | Medieval | 40 | 2 | 180 |
| Field Cannon | Renaissance | 50 | 2 | 330 |
| Machine Gun | Modern | 75 | 2 | 440 |

### Cavalry Line
| Unit | Era | Combat Strength | Production Cost |
|------|-----|----------------|----------------|
| Horseman | Classical | 36 | 80 |
| Knight | Medieval | 48 | 180 |
| Cavalry | Industrial | 62 | 330 |
| Helicopter | Atomic | 86 | 520 |

### Siege Line
| Unit | Era | Ranged Strength | Range | Production Cost |
|------|-----|----------------|-------|----------------|
| Catapult | Classical | 35 | 2 | 120 |
| Bombard | Renaissance | 55 | 2 | 280 |
| Artillery | Industrial | 70 | 3 | 420 |
| Rocket Artillery | Atomic | 100 | 3 | 600 |

### Special Units
| Unit | Era | Combat Strength | Notes |
|------|-----|----------------|-------|
| Warrior Monk | Classical | 35 | Religious, unique promotion tree |
| Giant Death Robot | Future | 130 | Most powerful land unit |
| Nuclear Device | Atomic | -- | Area damage, fallout |
| Thermonuclear Device | Information | -- | Massive area damage |

---

## Great People

### Types (9)
| Type | Generated By | Key Uses |
|------|-------------|----------|
| Great Scientist | Campus | Free techs, bonus Science, special abilities |
| Great Engineer | Industrial Zone | Rush wonders, bonus Production, special abilities |
| Great Merchant | Commercial Hub / Harbor | Gold bonuses, unique trade abilities |
| Great Prophet | Holy Site / Stonehenge | Found/enhance religions (limited to number of religions) |
| Great General | Encampment | +5 combat to land units within 2 tiles |
| Great Admiral | Harbor | +5 combat to naval units within 2 tiles |
| Great Writer | Theater Square | Creates Great Works of Writing (+2 Culture, +2 Tourism) |
| Great Artist | Theater Square | Creates Great Works of Art (+3 Culture, +3 Tourism) |
| Great Musician | Theater Square | Creates Great Works of Music (+4 Culture, +4 Tourism) |

---

## Victory Conditions (Exact Requirements)

### Science Victory
1. **Launch Earth Satellite** (Spaceport project, ~1500 Production)
2. **Launch Moon Landing** (Spaceport project, ~1500 Production)
3. **Launch Mars Colony** - 3 separate modules:
   - Mars Habitation (~1500 Production)
   - Mars Hydroponics (~1500 Production)
   - Mars Reactor (~1500 Production)
4. **Exoplanet Expedition** (Gathering Storm, ~2000 Production)
   - Then wait 50 turns (reduced by light-year projects at Spaceports)

Must complete in order. Total: ~7 Spaceport projects.

### Culture Victory
- **Condition**: Your civilization's **foreign tourists** exceed every other civilization's **domestic tourists**
- **Domestic Tourists formula**: total lifetime Culture / 150 (approximately)
- **Foreign Tourists**: generated by Tourism output attracting visitors from other civs
- Key sources: Great Works, Wonders, National Parks, Seaside Resorts, Holy City tourism, Open Borders bonus (+25%), Trade Route bonus (+25%)

### Domination Victory
- **Condition**: Control every other civilization's **original capital**
- Capitals are marked with a star icon
- You must maintain control; losing a capital means losing progress
- You do NOT need to destroy other civilizations entirely

### Religious Victory
- **Condition**: Your religion must be the **predominant religion** in every other civilization still in the game
- Predominant = more than 50% of a civilization's cities follow your religion
- Requires founding a religion (Great Prophet) and spreading it with Missionaries/Apostles

### Diplomatic Victory (Gathering Storm only)
- **Condition**: Earn **20 Diplomatic Victory Points**
- Sources: World Congress resolutions, Scored Competitions, Statue of Liberty (+4), Carbon Recapture project
- Diplomatic Victory Points can be lost through adverse World Congress votes
- Typically earned 1-2 per World Congress session

### Score Victory
- **Condition**: Highest score when turn limit expires (default turn 500 on Standard)
- Score components: Civics, Technologies, Cities, Districts, Population, Great People, Religion, Wonders, Territory
- Tiebreaker priority: Civics > Cities > Districts > Population > Great People > Religion > Technologies > Wonders

---

## City Mechanics

### Population & Housing
- Population grows based on surplus Food
- **Housing cap**: growth slows dramatically when Population >= Housing
  - Growth -50% when within 1 of cap
  - Growth -75% when at cap
  - Growth stops when 5+ over cap
- Housing sources: City Center (2 base + water bonus: +3 coast, +5 fresh water), farms, Granary (+2), districts, Aqueduct (+2-6)

### Amenities
- Cities need 1 Amenity per 2 Population (above 2)
- Surplus amenities: +5% non-food yields at Ecstatic (+3)
- Deficit: -15% non-food yields at Displeased (-1 to -2), -30% at Unhappy (-3 to -4), rebels spawn at Unrest (-5+)
- Sources: Luxury resources (1 per type, +1 amenity to 4 cities), Entertainment Complex, Water Park, policies, Great People, religion

### Loyalty (Rise and Fall / Gathering Storm)
- Cities have Loyalty 0-100
- At 0 Loyalty: city becomes a Free City (independent, can be conquered)
- Loyalty pressure from nearby civilizations based on population, proximity, governors
- +8 base loyalty per turn; modifiers from population, amenities, governors, governments

### District Placement
- Districts must be placed on tiles within city borders
- Each district occupies 1 tile (removes any improvements/features)
- Production cost scales: first district 54 Production, increases with number built across empire

---

## Trade Routes

- **Domestic routes**: transfer Food and Production between your cities
- **International routes**: generate Gold (base ~3-6 per route), plus bonus yields
- Trade route capacity: 1 base, +1 per Commercial Hub/Harbor with appropriate buildings, +1 from some wonders/governments
- Route range: 15 tiles base (increased by buildings/policies)
- Duration: 20 turns (standard speed)
- Plunder: enemy units can plunder trade routes for gold

---

## Interference-Relevant Complexity

This domain is rich for narrative interference because:
- **50 civilizations** with unique abilities create massive factual space
- **Numerical systems overlap**: Production vs Gold vs Faith all function as "currencies" with different conversion rates
- **Victory conditions have precise thresholds** (20 Diplomatic Points, 50% religious majority, etc.)
- **Unit upgrade lines** create temporal dependencies (Warrior -> Swordsman -> Man-at-Arms -> Musketman)
- **District adjacency bonuses** involve spatial reasoning with exact numeric yields
- **Wonder requirements and costs** vary by era with specific placement rules
- **Government policy slots** differ by type and tier with exact slot counts
