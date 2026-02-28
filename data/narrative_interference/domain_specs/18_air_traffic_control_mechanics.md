# Air Traffic Control Mechanics Reference for Sector Simulation

## 1. Airspace Classification (FAA / ICAO)

### Class A Airspace
- **Altitude**: 18,000 ft MSL (FL180) to FL600 (60,000 ft)
- **Flight rules**: IFR only
- **ATC clearance**: Required
- **Separation**: ATC provides separation to all aircraft
- **Equipment required**: Two-way radio, transponder with Mode C, ADS-B Out
- **Pilot requirements**: Instrument rating required
- **Speed limit**: None below Mach 1 (supersonic flight prohibited without authorization)
- **Weather minimums**: Not applicable (IFR only)
- **Altimeter setting**: Above 18,000 ft MSL, all altimeters set to 29.92 inHg (standard pressure)

### Class B Airspace
- **Altitude**: Surface to typically 10,000 ft MSL (varies by location; depicted on sectional charts as an "upside-down wedding cake")
- **Location**: Surrounds the 37 busiest US airports (e.g., LAX, JFK, ORD, ATL, DFW)
- **ATC clearance**: Required ("Cleared into Class Bravo airspace")
- **Separation**: ATC provides separation to all aircraft (IFR and VFR)
- **Equipment required**: Two-way radio, transponder with Mode C (within and above Class B), ADS-B Out
- **Pilot requirements**: Private pilot certificate minimum (student pilots need logbook endorsement for specific Class B airports)
- **Speed limit**: 250 KIAS below 10,000 ft MSL; 200 KIAS in VFR corridor or beneath Class B shelf
- **VFR weather minimums**: 3 SM visibility, clear of clouds

### Class C Airspace
- **Altitude**: Surface to typically 4,000 ft AGL (two-tiered: inner circle 5 nm radius surface to 4,000 AGL; outer circle 10 nm radius 1,200-4,000 AGL)
- **Location**: Airports with operational control tower, radar approach control, and significant traffic (approximately 120 US airports)
- **ATC clearance**: Two-way radio communication required before entry (not a formal clearance, but "establishment of communication")
- **Separation**: IFR separated from IFR and VFR; VFR receives traffic advisories and sequencing
- **Equipment required**: Two-way radio, transponder with Mode C (within and above to 10,000 ft), ADS-B Out
- **Speed limit**: 200 KIAS at or below 2,500 ft AGL within 4 nm of primary airport; 250 KIAS below 10,000 ft MSL
- **VFR weather minimums**: 3 SM visibility; 500 ft below, 1,000 ft above, 2,000 ft horizontal from clouds

### Class D Airspace
- **Altitude**: Surface to typically 2,500 ft AGL (depicted by dashed blue line on sectional)
- **Location**: Airports with operational control tower (~500 US airports)
- **ATC clearance**: Two-way radio communication required
- **Separation**: IFR separated from IFR; VFR receives traffic advisories (sequencing for landing)
- **Equipment required**: Two-way radio; transponder with Mode C if above Class D to 10,000 ft, ADS-B Out
- **Speed limit**: 200 KIAS at or below 2,500 ft AGL within 4 nm of primary airport; 250 KIAS below 10,000 ft MSL
- **VFR weather minimums**: 3 SM visibility; 500 ft below, 1,000 ft above, 2,000 ft horizontal from clouds

### Class E Airspace
- **Altitude**: Varies; typically starts at 1,200 ft AGL (en route), 700 ft AGL (near airports with instrument approaches), or surface (designated as Class E surface area)
- **Upper bound**: Up to but not including 18,000 ft MSL (where Class A begins)
- **ATC clearance**: Not required for VFR; IFR requires ATC clearance
- **Separation**: ATC provides separation to IFR aircraft only
- **Equipment required**: Transponder with Mode C above 10,000 ft MSL (excluding below 2,500 ft AGL); ADS-B Out in certain areas
- **Speed limit**: 250 KIAS below 10,000 ft MSL
- **VFR weather minimums (below 10,000 ft MSL)**: 3 SM visibility; 500 ft below, 1,000 ft above, 2,000 ft horizontal from clouds
- **VFR weather minimums (at/above 10,000 ft MSL)**: 5 SM visibility; 1,000 ft below, 1,000 ft above, 1 SM horizontal from clouds

### Class G Airspace (Uncontrolled)
- **Altitude**: Surface to where Class E begins (typically surface to 700 ft or 1,200 ft AGL)
- **ATC clearance**: Not required
- **Separation**: None provided by ATC
- **Equipment required**: None mandated (transponder/ADS-B in certain areas)
- **Speed limit**: 250 KIAS below 10,000 ft MSL
- **VFR weather minimums (day, below 1,200 ft AGL)**: 1 SM visibility; clear of clouds
- **VFR weather minimums (night, below 1,200 ft AGL)**: 3 SM visibility; 500 ft below, 1,000 ft above, 2,000 ft horizontal
- **VFR weather minimums (above 1,200 ft AGL, below 10,000 ft MSL)**: 1 SM day / 3 SM night; 500 ft below, 1,000 ft above, 2,000 ft horizontal
- **VFR weather minimums (above 10,000 ft MSL)**: 5 SM visibility; 1,000 ft below, 1,000 ft above, 1 SM horizontal

---

## 2. Separation Minimums

### Vertical Separation

| Altitude Range | Standard Vertical Separation | Notes |
|---|---|---|
| Surface to FL290 (29,000 ft) | 1,000 ft | Standard for all aircraft |
| FL290 to FL410 (RVSM airspace) | 1,000 ft | Requires RVSM-approved aircraft, autopilot, dual altimeters |
| FL290 to FL410 (non-RVSM) | 2,000 ft | For non-RVSM-approved aircraft |
| Above FL410 | 2,000 ft | Standard above upper RVSM limit |
| Visual separation | N/A | Controller or pilot sees the other aircraft; used in VMC |

**RVSM (Reduced Vertical Separation Minimum)**: Implemented in US domestic airspace since January 20, 2005. Aircraft must meet specific equipment and training requirements: two independent altitude measurement systems, altitude alerting system, automatic altitude-hold system.

### Horizontal (Radar) Separation

| Airspace Type | Minimum Separation | Notes |
|---|---|---|
| Terminal (TRACON) | 3 nm | Within approximately 40 nm of airport, below ~10,000-15,000 ft |
| En route (ARTCC/Center) | 5 nm | Standard en route radar separation |
| Non-radar | Various | Time-based or procedural: 5-minute intervals, 10 nm DME, etc. |
| Oceanic (non-radar) | 50 nm lateral / 10 min longitudinal | Reduced with ADS-C/CPDLC to 30 nm lateral |
| Tower (visual) | Visual separation | Controller ensures adequate spacing visually |

### Longitudinal (Same-Track) Separation Without Radar

| Scenario | Minimum Separation |
|---|---|
| Same altitude, same direction | 5 minutes (or 3 minutes with speed differential) |
| Same altitude, opposite direction | 10 minutes |
| Climbing/descending through another's altitude | 5 minutes, or 10 nm DME |

---

## 3. Flight Levels and Altitude Conventions

### Transition Altitude and Transition Level
- **US domestic**: Transition altitude is 18,000 ft MSL. Below 18,000 ft, use local altimeter setting. At/above 18,000 ft (FL180), set altimeter to 29.92 inHg (standard pressure).
- **International**: Varies by country. Transition altitude can be as low as 3,000 ft (some countries) to 18,000 ft.

### Flight Level Assignment (Semicircular Rule — US)

| Heading (Magnetic Course) | IFR Altitudes/Flight Levels | VFR Altitudes |
|---|---|---|
| 0°-179° (Eastbound) | Odd thousands: 3,000; 5,000; 7,000... FL230, FL250, FL270... | Odd thousands + 500: 3,500; 5,500; 7,500... |
| 180°-359° (Westbound) | Even thousands: 4,000; 6,000; 8,000... FL240, FL260, FL280... | Even thousands + 500: 4,500; 6,500; 8,500... |

### RVSM Flight Levels (FL290-FL410)

| Eastbound (0°-179°) | Westbound (180°-359°) |
|---|---|
| FL290 | FL300 |
| FL310 | FL320 |
| FL330 | FL340 |
| FL350 | FL360 |
| FL370 | FL380 |
| FL390 | FL400 |
| FL410 | — |

### Above FL410

| Eastbound (0°-179°) | Westbound (180°-359°) |
|---|---|
| FL430 | FL440 |
| FL450 | FL460 |
| FL470 | FL480 |
| FL490 | FL500 |
| FL510 | FL520 |

---

## 4. Speed Restrictions

### FAR 91.117 Speed Restrictions

| Condition | Maximum Speed | Regulation |
|---|---|---|
| Below 10,000 ft MSL | 250 KIAS | 91.117(a) — ATC CANNOT waive |
| In/beneath Class B airspace | 200 KIAS | 91.117(b) — in VFR corridor or beneath shelf |
| Within 4 nm of Class C/D primary airport, at/below 2,500 ft AGL | 200 KIAS | 91.117(c) — ATC CAN waive |
| Class A airspace | No limit below Mach 1 | Supersonic flight requires special authorization |

### ATC Speed Assignments (FAA Order 7110.65)

| Phase | Typical Speed Assignment | Notes |
|---|---|---|
| Departure (below 10,000 ft) | 250 KIAS max (regulatory) | "Maintain two five zero knots" |
| En route climb | 280-310 KIAS or Mach 0.74-0.84 | Type-dependent |
| En route cruise | Mach 0.76-0.85 | Type-dependent |
| Arrival (above 10,000 ft) | 280 KIAS | "Descend via, maintain two eight zero knots" |
| Approach (10,000-7,000 ft) | 250 KIAS | Regulatory limit |
| Approach (below 7,000 ft) | 210 KIAS | "Reduce speed to two one zero" |
| Base turn / downwind | 180 KIAS | "Reduce speed to one eight zero" |
| Final approach (10+ nm) | 170 KIAS | "Reduce speed to one seven zero" |
| Final approach (5 nm) | 160 KIAS | Common for jet traffic |
| Short final | Approach speed (Vref + 5-10 kt) | Aircraft-type dependent |

### Minimum Speed Assignments by ATC
- Jets: Not below 150 KIAS (unless on final approach)
- Turboprops: Not below 120 KIAS
- All aircraft: ATC cannot assign a speed less than the aircraft's minimum safe speed

---

## 5. Aircraft Types and Performance Data

### Narrowbody Jets

| Aircraft | ICAO Code | Wake Cat | MTOW (lbs) | Max Altitude (ft) | Cruise Speed (KTAS / Mach) | Approach Speed Vref (KIAS) | Climb Rate (ft/min) | Descent Rate (ft/min) | Fuel Flow Cruise (lbs/hr) | Range (nm) | Pax |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Boeing 737-700 | B737 | L (Large) | 154,500 | 41,000 | 453 / M0.785 | 128-134 | 3,000-3,500 | 1,500-2,500 | 4,400 | 3,010 | 126-149 |
| Boeing 737-800 | B738 | L | 174,200 | 41,000 | 460 / M0.789 | 135-142 | 2,800-3,400 | 1,500-2,500 | 4,900 | 2,935 | 162-189 |
| Boeing 737-900ER | B739 | L | 187,700 | 41,000 | 460 / M0.789 | 138-146 | 2,600-3,200 | 1,500-2,500 | 5,100 | 2,950 | 177-215 |
| Boeing 737 MAX 8 | B38M | L | 182,200 | 41,000 | 460 / M0.79 | 134-140 | 3,000-3,500 | 1,500-2,500 | 4,500 | 3,515 | 162-210 |
| Boeing 737 MAX 9 | B39M | L | 194,700 | 41,000 | 460 / M0.79 | 138-144 | 2,800-3,300 | 1,500-2,500 | 4,700 | 3,515 | 178-220 |
| Boeing 757-200 | B752 | L (H wake) | 255,000 | 42,000 | 461 / M0.80 | 130-140 | 3,000-3,800 | 1,500-3,000 | 5,800 | 3,900 | 200-239 |
| Boeing 757-300 | B753 | L (H wake) | 272,500 | 42,000 | 461 / M0.80 | 135-145 | 2,700-3,500 | 1,500-3,000 | 6,200 | 3,395 | 243-295 |
| Airbus A319 | A319 | L | 166,500 | 39,800 | 447 / M0.78 | 126-132 | 3,000-3,500 | 1,500-2,500 | 4,500 | 3,700 | 124-156 |
| Airbus A320 | A320 | L | 172,000 | 39,800 | 450 / M0.78 | 131-137 | 2,800-3,400 | 1,500-2,500 | 4,800 | 3,300 | 150-186 |
| Airbus A320neo | A20N | L | 174,200 | 39,800 | 450 / M0.78 | 131-137 | 3,000-3,500 | 1,500-2,500 | 4,200 | 3,500 | 150-194 |
| Airbus A321 | A321 | L | 206,100 | 39,800 | 450 / M0.78 | 136-142 | 2,500-3,200 | 1,500-2,500 | 5,500 | 3,200 | 185-230 |
| Airbus A321neo | A21N | L | 213,800 | 39,800 | 450 / M0.78 | 136-142 | 2,800-3,400 | 1,500-2,500 | 4,800 | 4,000 | 185-244 |

### Widebody Jets

| Aircraft | ICAO Code | Wake Cat | MTOW (lbs) | Max Altitude (ft) | Cruise Speed (KTAS / Mach) | Approach Speed Vref (KIAS) | Climb Rate (ft/min) | Descent Rate (ft/min) | Fuel Flow Cruise (lbs/hr) | Range (nm) | Pax |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Boeing 767-300ER | B763 | H | 412,000 | 43,100 | 459 / M0.80 | 135-145 | 2,500-3,500 | 1,500-2,500 | 8,900 | 5,990 | 218-351 |
| Boeing 777-200ER | B772 | H | 656,000 | 43,100 | 490 / M0.84 | 140-150 | 2,500-3,500 | 1,500-2,500 | 14,000 | 7,725 | 301-440 |
| Boeing 777-200LR | B77L | H | 766,000 | 43,100 | 490 / M0.84 | 142-152 | 2,300-3,300 | 1,500-2,500 | 14,500 | 9,395 | 301-440 |
| Boeing 777-300ER | B77W | H | 775,000 | 43,100 | 490 / M0.84 | 145-155 | 2,200-3,200 | 1,500-2,500 | 15,500 | 7,370 | 365-550 |
| Boeing 787-8 | B788 | H | 502,500 | 43,100 | 487 / M0.85 | 135-145 | 2,800-3,500 | 1,500-2,500 | 10,500 | 7,355 | 242-381 |
| Boeing 787-9 | B789 | H | 560,000 | 43,100 | 487 / M0.85 | 140-148 | 2,600-3,400 | 1,500-2,500 | 11,500 | 7,635 | 290-420 |
| Boeing 787-10 | B78X | H | 560,000 | 41,100 | 487 / M0.85 | 142-150 | 2,500-3,200 | 1,500-2,500 | 12,000 | 6,430 | 318-440 |
| Boeing 747-400 | B744 | H | 875,000 | 45,100 | 490 / M0.855 | 148-160 | 2,000-3,000 | 1,500-2,500 | 20,000 | 7,260 | 416-524 |
| Boeing 747-8 | B748 | H | 987,000 | 43,100 | 490 / M0.855 | 150-162 | 2,000-2,800 | 1,500-2,500 | 19,500 | 7,730 | 410-605 |
| Airbus A330-300 | A333 | H | 513,700 | 41,100 | 472 / M0.82 | 135-145 | 2,500-3,500 | 1,500-2,500 | 11,400 | 6,350 | 277-440 |
| Airbus A330-900neo | A339 | H | 533,500 | 41,450 | 472 / M0.82 | 135-143 | 2,600-3,500 | 1,500-2,500 | 10,200 | 7,200 | 260-440 |
| Airbus A340-600 | A346 | H | 811,300 | 41,100 | 473 / M0.82 | 140-152 | 2,000-3,000 | 1,500-2,500 | 17,000 | 7,900 | 326-475 |
| Airbus A350-900 | A359 | H | 617,300 | 43,100 | 487 / M0.85 | 135-145 | 2,800-3,500 | 1,500-2,500 | 11,000 | 8,100 | 300-440 |
| Airbus A350-1000 | A35K | H | 695,000 | 41,450 | 487 / M0.85 | 140-150 | 2,500-3,200 | 1,500-2,500 | 12,500 | 8,700 | 350-480 |
| Airbus A380-800 | A388 | J (Super) | 1,268,000 | 43,100 | 490 / M0.85 | 145-155 | 1,800-2,800 | 1,500-2,000 | 22,000 | 8,000 | 525-853 |

### Regional Jets

| Aircraft | ICAO Code | Wake Cat | MTOW (lbs) | Max Altitude (ft) | Cruise Speed (KTAS / Mach) | Approach Speed Vref (KIAS) | Climb Rate (ft/min) | Descent Rate (ft/min) | Fuel Flow Cruise (lbs/hr) | Range (nm) | Pax |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CRJ-200 | CRJ2 | L | 53,000 | 41,000 | 424 / M0.74 | 126-132 | 3,000-3,800 | 1,500-2,500 | 2,600 | 1,700 | 50 |
| CRJ-700 | CRJ7 | L | 75,000 | 41,000 | 447 / M0.78 | 128-134 | 2,800-3,500 | 1,500-2,500 | 3,200 | 1,800 | 66-78 |
| CRJ-900 | CRJ9 | L | 84,500 | 41,000 | 447 / M0.78 | 132-138 | 2,600-3,300 | 1,500-2,500 | 3,500 | 1,550 | 76-90 |
| Embraer E170 | E170 | L | 82,000 | 41,000 | 430 / M0.75 | 124-130 | 3,000-3,500 | 1,500-2,500 | 3,200 | 2,100 | 66-78 |
| Embraer E175 | E75L/E75S | L | 89,000 | 41,000 | 445 / M0.78 | 126-132 | 2,800-3,400 | 1,500-2,500 | 3,400 | 2,200 | 76-88 |
| Embraer E190 | E190 | L | 114,200 | 41,000 | 447 / M0.78 | 130-136 | 2,600-3,200 | 1,500-2,500 | 3,900 | 2,400 | 97-114 |
| Embraer E195 | E195 | L | 118,400 | 41,000 | 447 / M0.78 | 132-138 | 2,400-3,000 | 1,500-2,500 | 4,100 | 2,200 | 108-132 |

### Turboprops

| Aircraft | ICAO Code | Wake Cat | MTOW (lbs) | Max Altitude (ft) | Cruise Speed (KTAS) | Approach Speed Vref (KIAS) | Climb Rate (ft/min) | Descent Rate (ft/min) | Range (nm) | Pax |
|---|---|---|---|---|---|---|---|---|---|---|
| Dash 8-Q400 | DH8D | M | 65,200 | 27,000 | 360 | 110-120 | 1,500-2,200 | 1,000-1,800 | 1,360 | 68-90 |
| ATR 72-600 | AT76 | M | 50,700 | 25,000 | 275 | 105-115 | 1,200-1,800 | 1,000-1,500 | 825 | 70-78 |
| ATR 42-600 | AT46 | M | 41,000 | 25,000 | 265 | 100-110 | 1,200-1,800 | 1,000-1,500 | 740 | 48-50 |
| Saab 340B | SF34 | L | 29,000 | 31,000 | 282 | 102-108 | 1,500-2,000 | 1,000-1,500 | 910 | 33-36 |

### Business / General Aviation Jets

| Aircraft | ICAO Code | Wake Cat | MTOW (lbs) | Max Altitude (ft) | Cruise Speed (KTAS / Mach) | Approach Speed Vref (KIAS) | Range (nm) | Pax |
|---|---|---|---|---|---|---|---|---|
| Cessna Citation CJ3+ | C25B | L | 14,800 | 45,000 | 415 / M0.72 | 99-108 | 2,040 | 7-9 |
| Cessna Citation X | C750 | L | 36,100 | 51,000 | 525 / M0.92 | 114-122 | 3,460 | 8-12 |
| Cessna Citation Latitude | C680 | L | 30,800 | 45,000 | 446 / M0.74 | 108-116 | 2,700 | 7-9 |
| Gulfstream G550 | GLF5 | L | 91,000 | 51,000 | 488 / M0.85 | 120-130 | 6,750 | 14-19 |
| Gulfstream G650 | GLF6 | L | 99,600 | 51,000 | 516 / M0.90 | 122-132 | 7,000 | 14-19 |
| Bombardier Global 7500 | GL7T | L | 114,850 | 51,000 | 516 / M0.90 | 125-135 | 7,700 | 14-19 |
| Learjet 75 | LJ75 | L | 21,500 | 51,000 | 465 / M0.81 | 118-126 | 2,040 | 6-9 |
| Dassault Falcon 7X | FA7X | L | 70,000 | 51,000 | 481 / M0.84 | 108-118 | 5,950 | 12-16 |

**Note on Boeing 757 wake turbulence**: The B757 is classified as Large (L) by weight but produces heavy-class wake turbulence. FAA applies special wake separation rules: aircraft following a B757 use Heavy wake separation distances (4-5 nm behind).

---

## 6. Approach Types and Minimums

### Precision Approaches

| Approach Type | Decision Height (DH) | RVR / Visibility | Equipment Required | Key Characteristics |
|---|---|---|---|---|
| **ILS CAT I** | 200 ft AGL | RVR 2,400 ft (1,800 ft with TDZ/CL lights) | ILS receiver, marker beacon or DME | Standard precision approach; most common |
| **ILS CAT II** | 100 ft AGL | RVR 1,200 ft | Autopilot coupled, HUD optional, special crew training | Requires CAT II certified runway and aircraft |
| **ILS CAT IIIa** | 50 ft AGL (or no DH) | RVR 700 ft | Autoland capable, dual autopilot, HUD | Zero ceiling operations |
| **ILS CAT IIIb** | 50 ft AGL (or no DH) | RVR 150-700 ft | Autoland, fail-operational autopilot, rollout guidance | Near-zero visibility |
| **ILS CAT IIIc** | No DH | No RVR minimum | Full autoland + auto-taxi (theoretical) | Not currently operational in US |
| **GLS (GBAS)** | 200 ft (CAT I) | RVR 2,400 ft | GBAS receiver | GPS-augmented precision approach |

### Non-Precision Approaches

| Approach Type | MDA (Minimum Descent Altitude) | Visibility | Equipment Required | Notes |
|---|---|---|---|---|
| **VOR** | Typically 500-800 ft AGL | 1 SM (5,280 ft RVR equivalent) | VOR receiver | Being phased out; wider approach course |
| **VOR/DME** | Typically 400-700 ft AGL | 1 SM | VOR + DME | Step-down fixes provide better vertical guidance |
| **NDB** | Typically 500-900 ft AGL | 1 SM | ADF receiver | Largely decommissioned in US |
| **RNAV (GPS) LNAV** | Typically 400-600 ft AGL | 1 SM | GPS with RAIM | Lateral guidance only |
| **RNAV (GPS) LNAV/VNAV** | Typically 300-500 ft AGL | 1 SM | GPS + baro-VNAV | Advisory vertical guidance |
| **RNAV (GPS) LPV** | 200-250 ft AGL | RVR 2,400 ft (or lower) | WAAS GPS | Near-precision; equivalent to CAT I minimums |
| **LOC (Localizer only)** | Typically 400-600 ft AGL | 1 SM | ILS localizer receiver | No glideslope; uses MDA not DH |

### Visual Approaches

| Approach Type | Ceiling Minimum | Visibility Minimum | Notes |
|---|---|---|---|
| **Visual approach** | 1,000 ft AGL | 3 SM | Pilot must have airport or preceding aircraft in sight |
| **Contact approach** | None specified | 1 SM | Pilot-requested only; must remain clear of clouds |
| **Charted visual** | Varies (published) | Varies (published) | RNAV-based visual with published fixes (e.g., EXPRESSWAY VISUAL RWY 31 at LGA) |

---

## 7. Runway Operations

### Landing and Takeoff Intervals

#### Wake Turbulence Distance-Based Separation (Approach/Landing)

| Leading Aircraft | Trailing Aircraft | Minimum Separation |
|---|---|---|
| Super (J) — A380 | Heavy | 6 nm |
| Super (J) | Large | 7 nm |
| Super (J) | Small | 8 nm |
| Heavy | Heavy | 4 nm |
| Heavy | Large | 5 nm |
| Heavy | Small | 6 nm |
| B757 | Large | 4 nm |
| B757 | Small | 5 nm |
| Large | Small | 4 nm |
| Large | Large | 2.5 nm (radar minimum) |
| Small | Small | 2.5 nm (radar minimum) |

#### Wake Turbulence Time-Based Separation (Departures — Same Runway)

| Leading Aircraft | Trailing Aircraft | Minimum Interval |
|---|---|---|
| Super | Heavy | 2 minutes |
| Super | Large/Medium | 3 minutes |
| Super | Small/Light | 3 minutes |
| Heavy | Large | 2 minutes |
| Heavy | Small | 2 minutes |
| B757 | Small | 2 minutes |

#### Standard Runway Occupancy Times

| Operation | Time | Notes |
|---|---|---|
| Landing roll to runway exit | 45-60 seconds | Varies by aircraft type and exit location |
| Takeoff roll to airborne | 30-45 seconds | Varies by aircraft type and weight |
| Intersection departure | 25-35 seconds | Shorter roll from intersection |
| Touch-and-go | 20-30 seconds | Training operations |

#### Typical Runway Throughput

| Configuration | Operations per Hour | Notes |
|---|---|---|
| Single runway (arrivals only) | 30-36 | Depends on aircraft mix |
| Single runway (departures only) | 40-50 | Faster due to shorter intervals |
| Single runway (mixed) | 30-40 | Alternating arrivals/departures |
| Dual parallel runways (>4,300 ft apart) | 60-75 | Independent simultaneous approaches |
| Dual parallel runways (2,500-4,300 ft apart) | 50-60 | Dependent approaches (1.5 nm stagger) |
| Dual parallel runways (<2,500 ft apart) | 40-50 | SOIA/PRM procedures required |
| Triple parallel (e.g., ATL, DFW) | 90-110 | Complex configurations |

---

## 8. Wake Turbulence Categories

### ICAO Categories (Traditional)

| Category | Code | MTOW | Examples |
|---|---|---|---|
| Super | J | >560,000 kg (1,234,600 lbs) | A380 |
| Heavy | H | >136,000 kg (300,000 lbs) | B747, B777, B787, A330, A340, A350, B767 |
| Medium | M | >7,000 kg (15,400 lbs) to 136,000 kg | B737, A320, E175, CRJ-900, ATR 72 |
| Light | L | ≤7,000 kg (15,400 lbs) | Cessna 172, Piper Cherokee, Beechcraft Baron |

### FAA RECAT (Re-categorization) — Used at Major US Airports

| Category | Code | Description | Examples |
|---|---|---|---|
| A | Super Heavy | Highest wake | A380 |
| B | Upper Heavy | | B747, B777-300ER, A340-600, C-5, C-17 |
| C | Lower Heavy | | B787, B777-200, A330, A350, B767 |
| D | Upper Large | | B757, B737-900, A321, MD-90 |
| E | Lower Large | | B737-700/800, A319, A320, E190 |
| F | Small/Light | | CRJ-200, E175, Dash 8, Citations, GA aircraft |

---

## 9. Common Callsigns and Airline Codes

### Major US Airlines

| Airline | IATA | ICAO | Radio Callsign |
|---|---|---|---|
| American Airlines | AA | AAL | "American" |
| Delta Air Lines | DL | DAL | "Delta" |
| United Airlines | UA | UAL | "United" |
| Southwest Airlines | WN | SWA | "Southwest" |
| JetBlue Airways | B6 | JBU | "JetBlue" |
| Alaska Airlines | AS | ASA | "Alaska" |
| Spirit Airlines | NK | NKS | "Spirit Wings" |
| Frontier Airlines | F9 | FFT | "Frontier Flight" |
| Hawaiian Airlines | HA | HAL | "Hawaiian" |
| Allegiant Air | G4 | AAY | "Allegiant" |

### Regional Airlines

| Airline | IATA | ICAO | Radio Callsign | Operates As |
|---|---|---|---|---|
| SkyWest Airlines | OO | SKW | "SkyWest" | United Express, Delta Connection, American Eagle, Alaska |
| Republic Airways | YX | RPA | "Brickyard" | American Eagle, United Express, Delta Connection |
| Endeavor Air | 9E | EDV | "Endeavor" | Delta Connection |
| Envoy Air | MQ | ENY | "Envoy" | American Eagle |
| PSA Airlines | OH | JIA | "Blue Streak" | American Eagle |
| Mesa Airlines | YV | ASH | "Air Shuttle" | United Express, American Eagle |
| Piedmont Airlines | PT | PDT | "Piedmont" | American Eagle |
| Horizon Air | QX | QXE | "Horizon Air" | Alaska |
| CommutAir | C5 | UCA | "CommutAir" | United Express |

### Cargo Airlines

| Airline | IATA | ICAO | Radio Callsign |
|---|---|---|---|
| FedEx Express | FX | FDX | "FedEx" |
| UPS Airlines | 5X | UPS | "UPS" |
| Atlas Air | 5Y | GTI | "Giant" |
| ABX Air | GB | ABX | "Abex" |
| Kalitta Air | K4 | CKS | "Connie" |
| Polar Air Cargo | PO | PAC | "Polar" |
| Western Global Airlines | KD | WGN | "Western Global" |

### International Airlines (Commonly in US Airspace)

| Airline | IATA | ICAO | Radio Callsign |
|---|---|---|---|
| Air Canada | AC | ACA | "Air Canada" |
| British Airways | BA | BAW | "Speedbird" |
| Lufthansa | LH | DLH | "Lufthansa" |
| Air France | AF | AFR | "Air France" |
| Emirates | EK | UAE | "Emirates" |
| Qatar Airways | QR | QTR | "Qatari" |
| Cathay Pacific | CX | CPA | "Cathay" |
| Japan Airlines | JL | JAL | "Japan Air" |
| Korean Air | KE | KAL | "Korean Air" |
| Singapore Airlines | SQ | SIA | "Singapore" |
| Qantas | QF | QFA | "Qantas" |
| Aer Lingus | EI | EIN | "Shamrock" |
| KLM | KL | KLM | "KLM" |
| Turkish Airlines | TK | THY | "Turkish" |
| Aeromexico | AM | AMX | "Aeromexico" |

### Callsign Format
- **Airline flights**: [Radio Callsign] [Flight Number] — e.g., "Delta one four seven two" (DAL1472)
- **General aviation**: Full registration number — e.g., "November one two three four alpha" (N1234A)
- **Abbreviated GA**: Last three characters after initial contact — e.g., "three four alpha"
- **Military**: Callsign + numbers — e.g., "Air Force One", "Reach two three", "Navy four five six"
- **Heavy/Super suffix**: Required on initial contact — e.g., "United four five heavy", "Emirates three eight super"

---

## 10. ATC Phraseology

### Taxi Phraseology

| Instruction | Controller Says | Pilot Reads Back |
|---|---|---|
| Taxi to runway | "Delta 1472, taxi to runway two seven left via alpha, bravo" | "Taxi to runway two seven left via alpha, bravo, Delta 1472" |
| Hold short | "Hold short of runway two seven right" | "Hold short runway two seven right, Delta 1472" |
| Cross runway | "Cross runway three three at taxiway charlie" | "Cross runway three three at charlie, Delta 1472" |
| Give way | "Give way to the company seven thirty-seven on your right" | "Give way to traffic on the right, Delta 1472" |
| Monitor ground | "Monitor ground point niner" | "Monitor ground point niner, Delta 1472" |

### Takeoff / Departure Phraseology

| Instruction | Controller Says | Pilot Reads Back |
|---|---|---|
| Line up and wait | "Delta 1472, runway two seven left, line up and wait" | "Line up and wait, runway two seven left, Delta 1472" |
| Cleared for takeoff | "Delta 1472, runway two seven left, cleared for takeoff, wind two four zero at one two" | "Cleared for takeoff, runway two seven left, Delta 1472" |
| Cancel takeoff clearance | "Delta 1472, cancel takeoff clearance, hold position" | "Cancelling takeoff, holding position, Delta 1472" |
| Fly heading | "Delta 1472, fly heading two seven zero" | "Heading two seven zero, Delta 1472" |
| Contact departure | "Delta 1472, contact departure one two four point three" | "Departure one two four point three, Delta 1472" |
| Climb and maintain | "Delta 1472, climb and maintain flight level three five zero" | "Climb and maintain flight level three five zero, Delta 1472" |

### En Route Phraseology

| Instruction | Controller Says | Pilot Reads Back |
|---|---|---|
| Frequency change | "Delta 1472, contact center on one three two point four five" | "Center one three two point four five, Delta 1472" |
| Altitude change | "Delta 1472, descend and maintain flight level two four zero" | "Descend and maintain flight level two four zero, Delta 1472" |
| Direct routing | "Delta 1472, proceed direct BOSTN" | "Direct BOSTN, Delta 1472" |
| Speed assignment | "Delta 1472, reduce speed to two five zero knots" | "Reduce speed two five zero knots, Delta 1472" |
| Traffic advisory | "Delta 1472, traffic twelve o'clock, five miles, opposite direction, heavy seven seven seven, flight level three six zero" | "Looking for traffic, Delta 1472" |
| Altimeter setting | "Delta 1472, altimeter two niner niner two" | "Two niner niner two, Delta 1472" |
| Hold instruction | "Delta 1472, hold east of MERIT on the zero nine zero radial, right turns, expect further clearance one four three zero zulu" | "Hold east of MERIT, zero nine zero radial, right turns, expect further clearance one four three zero, Delta 1472" |

### Approach / Landing Phraseology

| Instruction | Controller Says | Pilot Reads Back |
|---|---|---|
| Expect approach | "Delta 1472, expect ILS runway two seven left approach" | "Expecting ILS two seven left, Delta 1472" |
| Vectors for approach | "Delta 1472, turn left heading one eight zero, vectors ILS runway two seven left" | "Left heading one eight zero, vectors ILS two seven left, Delta 1472" |
| Cleared approach | "Delta 1472, four miles from JIMME, cleared ILS runway two seven left approach" | "Cleared ILS runway two seven left, Delta 1472" |
| Cleared visual | "Delta 1472, cleared visual approach runway two seven left, traffic to follow is a seven three seven on a two mile final" | "Cleared visual approach two seven left, traffic in sight, Delta 1472" |
| Speed on final | "Delta 1472, maintain one seven zero knots to the marker" | "One seven zero to the marker, Delta 1472" |
| Cleared to land | "Delta 1472, runway two seven left, cleared to land, wind two five zero at one zero" | "Cleared to land, two seven left, Delta 1472" |
| Go around | "Delta 1472, go around, fly heading two seven zero, climb and maintain three thousand" | "Going around, heading two seven zero, climb three thousand, Delta 1472" |
| Missed approach | "Delta 1472, execute missed approach, climb to four thousand, direct MERIT" | "Missed approach, climb four thousand, direct MERIT, Delta 1472" |
| Exit runway | "Delta 1472, turn left at taxiway echo, contact ground one two one point niner" | "Left at echo, ground one two one point niner, Delta 1472" |

### Critical Readback Items (MUST be read back)
1. Altimeter settings
2. Altitude assignments (climb/descend/maintain)
3. Heading assignments
4. Speed assignments
5. Runway assignments (takeoff/landing)
6. Hold short instructions
7. Frequency changes
8. "Cleared for takeoff" / "Cleared to land"
9. SID/STAR/approach clearances

---

## 11. Flight Phases

### Phase 1: Pre-Flight / Clearance Delivery
- Pilot obtains ATIS information
- Pilot contacts Clearance Delivery for IFR clearance
- Clearance format (CRAFT): **C**leared to [destination], via [**R**oute], [**A**ltitude] expect [cruise altitude] [time] minutes after departure, **F**requency [departure freq], **T**ransponder [squawk code]
- Example: "Delta 1472, cleared to Chicago O'Hare airport via the RNAV departure, runway heading, then as filed, climb via SID, departure frequency one two four point three, squawk four two three one"

### Phase 2: Taxi
- Pilot contacts Ground Control
- Receives taxi instructions with specific route (taxiway designators)
- Must hold short of all runways unless specifically cleared to cross
- Typical taxi speed: 15-20 knots (groundspeed)
- Must have received current ATIS before contacting ground

### Phase 3: Takeoff
- Pilot contacts Tower when approaching runway
- "Line up and wait" or immediate "Cleared for takeoff"
- Takeoff roll: 30-45 seconds for commercial jets
- V-speeds: V1 (decision speed), VR (rotation speed), V2 (takeoff safety speed)
- Initial climb gradient: Minimum 200 ft/nm for obstacle clearance
- Noise abatement procedures may apply (thrust cutback, turn restrictions)

### Phase 4: Departure / Initial Climb
- Climb to initial assigned altitude (typically 3,000-10,000 ft)
- Follow SID (Standard Instrument Departure) if assigned
- Frequency change from Tower to Departure (TRACON)
- 250 KIAS max below 10,000 ft
- Typical climb: 2,000-3,000 ft/min initially, decreasing with altitude
- Above 10,000 ft: accelerate to 280-310 KIAS
- Transition from KIAS to Mach above ~24,000-28,000 ft

### Phase 5: Cruise
- At assigned flight level (typically FL310-FL410 for jets)
- Mach 0.78-0.85 depending on aircraft type
- May request step climbs as fuel burns off (lower weight = higher optimal altitude)
- Frequency changes as aircraft transitions between ARTCC sectors
- Typical handoff: "Contact [Center name] on [frequency]"

### Phase 6: Descent / Arrival
- Top of descent (TOD) typically 100-150 nm from destination
- Follows STAR (Standard Terminal Arrival Route) if assigned
- "Descend via" clearance: follow published altitude and speed restrictions on STAR
- Cross-check: 3:1 rule (3 nm per 1,000 ft to lose; start descent 120 nm out for 40,000 ft)
- Decelerate below 250 KIAS before crossing 10,000 ft
- Transfer from Center to TRACON (Approach Control)

### Phase 7: Approach
- Vectors or cleared direct to approach fix
- Speed reductions: 210 > 180 > 170 > approach speed
- Intercept ILS/RNAV approach course (typically 30-45 degree intercept)
- Glideslope intercept at ~3,000-4,000 ft AGL
- Final approach: 3-degree glideslope, ~700-800 ft/min descent rate
- Decision height or MDA: go around or continue to land
- Typical final approach course: 5-10 nm

### Phase 8: Landing
- Cross threshold at 50 ft AGL, Vref + 5-10 kt
- Touchdown zone: first 3,000 ft of runway
- Deploy thrust reversers (if equipped), spoilers, brakes
- Deceleration to taxi speed: 30-45 seconds
- Exit runway at designated taxiway
- Contact Ground Control

### Phase 9: Taxi to Gate
- Follow ground control taxi instructions to assigned gate
- Typical taxi time: 5-15 minutes at major airports
- Arrive at gate, set parking brake, engines off

---

## 12. ATIS Information

### ATIS Format and Content

ATIS broadcasts are updated hourly or when conditions change significantly. Each update receives a sequential phonetic letter identifier.

**Standard ATIS Content (in order)**:

1. **Airport name**: "[Airport] information"
2. **Phonetic identifier**: "Alpha", "Bravo", "Charlie"... through "Zulu"
3. **Time of observation**: "[Hour][Minutes] zulu observation"
4. **Weather**:
   - Wind: direction (magnetic) and speed, gusts if applicable — "Wind two five zero at one two, gusts two zero"
   - Visibility: in statute miles — "Visibility one zero"
   - Weather phenomena: rain, snow, fog, etc. — "Light rain, mist"
   - Sky condition: "Few clouds at two thousand five hundred, ceiling broken at four thousand five hundred, overcast at eight thousand"
   - Temperature and dew point: in Celsius — "Temperature two two, dew point one five"
   - Altimeter: in inches of mercury — "Altimeter three zero one two"
5. **Approaches in use**: "ILS runway two seven left approach in use"
6. **Runways in use**: "Landing runway two seven left, departing runway two seven right"
7. **NOTAMs**: Taxiway closures, runway closures, construction, navaids OTS
8. **Special instructions**: "Read back all runway hold short instructions"
9. **Closing**: "Advise on initial contact you have information [letter]"

### Example ATIS Broadcast
"Chicago O'Hare information Hotel, two three five three zulu observation. Wind two seven zero at one four. Visibility one zero. Few clouds at three thousand, ceiling broken at one two thousand. Temperature minus zero three, dew point minus one zero. Altimeter two niner eight six. ILS runway two seven left approach in use. Landing and departing runways two seven left and two seven right. Taxiway mike between taxiway charlie and echo is closed. NOTAM: crane two miles southeast of field, four hundred feet. Simultaneous ILS approaches in progress. Read back all runway hold short instructions. Advise on initial contact you have information Hotel."

### ATIS Update Triggers
- Wind direction change ≥ 30 degrees
- Wind speed change ≥ 5 knots
- Visibility change crossing a breakpoint (e.g., 3 SM, 1 SM)
- Ceiling change crossing a breakpoint
- Altimeter change ≥ 0.04 inHg
- Runway change
- Approach type change
- Any new NOTAM affecting operations

---

## 13. Weather Minimums

### VFR Minimums by Airspace

| Airspace | Flight Visibility | Cloud Clearance |
|---|---|---|
| Class A | N/A (IFR only) | N/A |
| Class B | 3 SM | Clear of clouds |
| Class C | 3 SM | 500 ft below, 1,000 ft above, 2,000 ft horizontal |
| Class D | 3 SM | 500 ft below, 1,000 ft above, 2,000 ft horizontal |
| Class E (below 10,000 MSL) | 3 SM | 500 ft below, 1,000 ft above, 2,000 ft horizontal |
| Class E (at/above 10,000 MSL) | 5 SM | 1,000 ft below, 1,000 ft above, 1 SM horizontal |
| Class G (day, ≤1,200 AGL) | 1 SM | Clear of clouds |
| Class G (night, ≤1,200 AGL) | 3 SM | 500 ft below, 1,000 ft above, 2,000 ft horizontal |
| Class G (day, >1,200 AGL, <10,000 MSL) | 1 SM | 500 ft below, 1,000 ft above, 2,000 ft horizontal |
| Class G (night, >1,200 AGL, <10,000 MSL) | 3 SM | 500 ft below, 1,000 ft above, 2,000 ft horizontal |
| Class G (at/above 10,000 MSL) | 5 SM | 1,000 ft below, 1,000 ft above, 1 SM horizontal |

### IFR Approach Weather Minimums

| Approach Type | Decision Height / MDA | Minimum Visibility / RVR |
|---|---|---|
| ILS CAT I | 200 ft AGL | RVR 2,400 ft (1,800 ft with TDZ+CL) |
| ILS CAT II | 100 ft AGL | RVR 1,200 ft |
| ILS CAT IIIa | 50 ft or no DH | RVR 700 ft |
| ILS CAT IIIb | 50 ft or no DH | RVR 150-700 ft |
| ILS CAT IIIc | No DH | No RVR limit |
| RNAV (GPS) LPV | 200-250 ft | RVR 2,400 ft |
| RNAV (GPS) LNAV/VNAV | 300-500 ft | 3/4-1 SM |
| RNAV (GPS) LNAV only | 400-600 ft | 1 SM |
| VOR | 500-800 ft | 1 SM |
| LOC (no glideslope) | 400-600 ft | 3/4-1 SM |
| Visual approach | 1,000 ft ceiling | 3 SM |
| Contact approach | Clear of clouds | 1 SM |

### Ceiling Definitions
- **Few**: 1/8 to 2/8 sky covered (NOT a ceiling)
- **Scattered**: 3/8 to 4/8 sky covered (NOT a ceiling)
- **Broken**: 5/8 to 7/8 sky covered (IS a ceiling — first BKN or OVC layer)
- **Overcast**: 8/8 sky covered (IS a ceiling)

### Wind Limits (Typical Airline Operations)
- Maximum crosswind component: 25-35 knots (aircraft-type dependent)
- Maximum tailwind component: 10-15 knots
- Maximum headwind: No limit (but approach speed adjustments required)
- Low-level windshear: Go-around mandatory if encountered below 1,000 ft AGL
- Microburst alert: Avoid area; 15-knot loss or gain in approach path

---

## 14. Fuel Endurance and Reserve Requirements

### FAA Fuel Reserve Requirements

| Operation | Rule | Reserve Requirement |
|---|---|---|
| Part 91 VFR (day) | 91.151 | Fly to destination + 30 minutes at normal cruise |
| Part 91 VFR (night) | 91.151 | Fly to destination + 45 minutes at normal cruise |
| Part 91 IFR | 91.167 | Fly to destination, then to alternate, then 45 minutes at normal cruise |
| Part 121 (airlines, domestic) | 121.639 | Fly to destination + 45 minutes at normal cruise |
| Part 121 (airlines, international) | 121.645 | Fly to destination, then to alternate, then 45 minutes at normal cruise (or 10% of total flight time) |
| Part 135 (charter, IFR) | 135.223 | Fly to destination, then to alternate, then 45 minutes at normal cruise |

### Typical Fuel Endurance by Aircraft Type

| Aircraft | Max Fuel (lbs) | Cruise Burn (lbs/hr) | Max Endurance (hrs) | Typical Mission (hrs) |
|---|---|---|---|---|
| B737-800 | 46,063 | 4,900 | ~8.4 | 4-5 |
| A320 | 42,000 | 4,800 | ~7.8 | 3-5 |
| B757-200 | 82,400 | 5,800 | ~12.7 | 5-7 |
| B767-300ER | 162,000 | 8,900 | ~16.2 | 8-11 |
| B777-200ER | 301,000 | 14,000 | ~19.2 | 10-14 |
| B777-300ER | 303,000 | 15,500 | ~17.4 | 10-14 |
| B787-9 | 223,000 | 11,500 | ~17.3 | 10-14 |
| B747-400 | 387,000 | 20,000 | ~17.3 | 10-13 |
| A330-300 | 245,000 | 11,400 | ~19.2 | 8-12 |
| A350-900 | 229,000 | 11,000 | ~18.6 | 10-14 |
| A380-800 | 559,000 | 22,000 | ~22.7 | 12-16 |
| CRJ-900 | 18,600 | 3,500 | ~4.7 | 2-3 |
| E175 | 20,500 | 3,400 | ~5.4 | 2-3.5 |

### Fuel Emergency Declarations

| Declaration | Phraseology | Meaning | ATC Response |
|---|---|---|---|
| **Minimum fuel** | "[Callsign], minimum fuel" | Not an emergency. Advisory that any unplanned delay may result in emergency. | ATC provides priority handling, expedites routing; no emergency services |
| **Emergency fuel** | "[Callsign], declaring emergency, fuel emergency" + squawk 7700 | Unable to accept any delay. Fuel on board insufficient for normal approach. | Full emergency response: priority handling, crash/fire/rescue standby, vectors for shortest approach |

### Fuel Planning Terms
- **Trip fuel**: Fuel from departure to destination
- **Contingency fuel**: Typically 5% of trip fuel (or statistical analysis)
- **Alternate fuel**: Fuel from destination to alternate airport
- **Reserve fuel**: 45 minutes at normal cruise (regulatory minimum)
- **Taxi fuel**: Fuel for ground operations (typically 10-15 minutes)
- **Extra fuel**: Captain's discretion for weather, ATC delays, etc.
- **Total fuel**: Sum of all above

---

## 15. Emergency Procedures

### Transponder Emergency Codes

| Code | Meaning | Phraseology | ATC Response |
|---|---|---|---|
| **7700** | General emergency (Mayday) | "Mayday, mayday, mayday, [callsign], [nature of emergency], [intentions], [fuel remaining], [souls on board]" | Priority handling, clear airspace, vector to nearest suitable airport, alert crash/fire/rescue, notify ARFF |
| **7600** | Radio communication failure (NORDO) | N/A (no radio) | ATC clears airspace along expected route, expects pilot to follow lost comm procedures (FAR 91.185): route — Assigned, Vectored, Expected, Filed; altitude — highest of MEA, expected, assigned |
| **7500** | Hijack / unlawful interference | May be set silently without radio call | ATC acknowledges discreetly ("verify squawk 7500"), coordinates with law enforcement, military intercept possible, clears airspace |

### Emergency Priority

| Priority | Type | Phraseology | Description |
|---|---|---|---|
| **Distress (Mayday)** | Highest | "MAYDAY, MAYDAY, MAYDAY" | Imminent danger to aircraft or persons (engine failure, fire, structural failure, medical emergency requiring diversion) |
| **Urgency (Pan-Pan)** | Second | "PAN PAN, PAN PAN, PAN PAN" | Serious situation but no immediate danger (precautionary engine shutdown, non-critical system failure, passenger medical) |

### Mayday Call Format
1. "Mayday, mayday, mayday"
2. "[ATC facility being called]"
3. "[Aircraft callsign]"
4. "[Type of aircraft]"
5. "[Nature of emergency]"
6. "[Intentions / request]"
7. "[Present position and altitude]"
8. "[Fuel remaining in time]"
9. "[Souls on board]"

**Example**: "Mayday, mayday, mayday. Chicago Approach, Delta one four seven two, Boeing seven three seven, engine failure number two, request immediate return to O'Hare, runway two seven left. Currently two zero miles southwest at five thousand feet. Three hours fuel remaining. One five two souls on board."

### Common Emergency Scenarios

| Emergency | Immediate Actions | ATC Actions |
|---|---|---|
| Engine failure (multi-engine) | Secure failed engine, declare emergency, request vectors | Priority handling, shortest approach, ARFF standby |
| Engine failure (single-engine) | Immediate landing; "unable" any delay | Clear all traffic, vectors to nearest runway |
| Fire (engine) | Engine shutdown, fire bottles, divert immediately | Priority handling, ARFF deployed, clear runway |
| Fire (cabin) | Emergency descent, nearest airport | Emergency descent clearance, clear airspace below |
| Smoke in cockpit | Don oxygen masks, emergency descent | Immediate descent clearance to breathable altitude |
| Hydraulic failure | Declare emergency, request long straight-in approach | Clear runway, ARFF standby, foam runway if needed |
| Pressurization failure | Emergency descent to 10,000 ft or MEA | Clear airspace below, issue descent clearance immediately |
| Medical emergency | Request priority handling, nearest suitable airport | Vector to nearest airport with medical facilities, notify EMS |
| Fuel emergency | Request immediate approach, no delay tolerable | Priority vectors, shortest approach, clear runway |
| Bird strike | Assess damage, declare emergency if warranted | ARFF standby, priority if requested |
| Gear malfunction | Request low approach for visual inspection, declare emergency | Tower inspects gear visually, ARFF standby |
| Bomb threat | Request isolated area, may request immediate landing | Isolate aircraft on remote taxiway/runway, law enforcement |

### Emergency Fuel Criteria (Approximate)
- **Minimum fuel**: Fuel remaining is such that the flight can accept little or no delay (no emergency declaration, advisory only)
- **Emergency fuel**: Fuel remaining is less than the planned reserve OR pilot determines fuel on board is insufficient for a safe landing with normal reserves

---

## 16. ATC Facilities and Sectors

### Facility Types

| Facility | Abbreviation | Coverage | Typical Altitude |
|---|---|---|---|
| **Air Traffic Control Tower (ATCT)** | TWR | Airport surface and immediate vicinity (~5 nm) | Surface to ~3,000-5,000 ft AGL |
| **Terminal Radar Approach Control** | TRACON | Terminal area (~30-50 nm from airport) | Surface to ~10,000-18,000 ft |
| **Air Route Traffic Control Center** | ARTCC (Center) | En route airspace (millions of sq mi per center) | Above TRACON to FL600 |
| **Flight Service Station** | FSS | All altitudes; no separation | All altitudes (advisory, not control) |

### Major US ARTCCs (20 Centers)

| Center | Identifier | Coverage |
|---|---|---|
| Boston | ZBW | New England |
| New York | ZNY | NY/NJ metro area |
| Washington | ZDC | Mid-Atlantic |
| Cleveland | ZOB | Ohio/Western PA |
| Indianapolis | ZID | Indiana/Kentucky |
| Atlanta | ZTL | Southeast |
| Jacksonville | ZJX | Florida/Southeast |
| Miami | ZMA | South Florida/Caribbean |
| Chicago | ZAU | Upper Midwest |
| Minneapolis | ZMP | Northern Midwest |
| Kansas City | ZKC | Central Plains |
| Memphis | ZME | Mid-South |
| Fort Worth | ZFW | North Texas/Oklahoma |
| Houston | ZHU | South Texas/Gulf |
| Denver | ZDV | Mountain West |
| Albuquerque | ZAB | Southwest |
| Salt Lake City | ZLC | Northern Rockies |
| Los Angeles | ZLA | Southern California |
| Oakland | ZOA | Northern California/Pacific |
| Seattle | ZSE | Pacific Northwest |

### Frequency Ranges

| Facility | Frequency Range |
|---|---|
| Clearance Delivery | 118.0 - 136.975 MHz |
| Ground Control | 121.6 - 121.9 MHz |
| Tower (local control) | 118.0 - 136.975 MHz |
| TRACON (approach/departure) | 118.0 - 136.975 MHz |
| ARTCC (center) | 118.0 - 136.975 MHz (HF for oceanic) |
| ATIS | 118.0 - 136.975 MHz |
| Emergency | 121.5 MHz (guard frequency — monitored by all ATC) |
| Unicom / CTAF | 122.7, 122.8, 123.0, various |
| FSS | 122.2 MHz (universal) |

---

## 17. Standard Instrument Departures (SIDs) and STARs

### SID Components
- **Runway transition**: Specific heading or turn from each runway to join the common route
- **Common route**: Sequence of waypoints with altitude/speed restrictions
- **En route transition**: Multiple transition fixes to connect to the en route structure
- **Altitude restrictions**: "At or above", "at or below", "at" specific altitude at each fix
- **Speed restrictions**: Typically 250 KIAS max, some SIDs have lower restrictions

### STAR Components
- **En route transitions**: Multiple entry points from different directions
- **Common route**: Sequence of waypoints descending toward the airport
- **Runway transitions**: Branch to specific final approach courses
- **Altitude restrictions**: "Cross FIXXX at FL240", "Cross FIXXX at or above 11,000"
- **Speed restrictions**: "Cross FIXXX at 250 KIAS", "Cross FIXXX at or below 210 KIAS"
- **Expect clearance**: "Expect runway XX approach"

### "Descend Via" and "Climb Via" Clearances
- **"Climb via SID"**: Follow all altitude and speed restrictions published on the SID
- **"Descend via STAR"**: Follow all altitude and speed restrictions published on the STAR
- **Override**: ATC can override individual restrictions — "Cross FIXXX at FL280" overrides the published restriction at that fix but maintains all others
- **Amended altitude**: "Descend via the STAR except maintain FL240" — follow STAR but stop descent at FL240

---

## 18. Holding Patterns

### Standard Holding Pattern
- **Direction**: Right turns (standard); left turns if specified
- **Leg length**: 1 minute inbound below 14,000 ft MSL; 1.5 minutes at/above 14,000 ft MSL
- **DME leg**: May be specified in nm instead of time (e.g., "10 DME fix")
- **Speed limits**:
  - Up to 6,000 ft MSL: 200 KIAS
  - 6,001 to 14,000 ft MSL: 230 KIAS
  - Above 14,000 ft MSL: 265 KIAS

### Holding Clearance Format
"[Callsign], hold [direction] of [fix] on the [radial/course/airway], [right/left] turns, expect further clearance [time], maintain [altitude]"

**Example**: "Delta 1472, hold east of MERIT on the zero nine zero radial, right turns, expect further clearance at one four three zero zulu, maintain seven thousand"

### Expected Approach Time (EAT) / Expected Further Clearance (EFC)
- ATC provides EFC time so pilots can plan fuel
- If no EFC received and communication lost, depart holding at filed ETA

---

## 19. Units and Conventions

| Parameter | Unit | Convention |
|---|---|---|
| Altitude (below 18,000 ft) | Feet MSL | "Climb and maintain one two thousand" |
| Altitude (at/above 18,000 ft) | Flight Level | "Climb and maintain flight level three five zero" (=35,000 ft std pressure) |
| Speed (below ~28,000 ft) | KIAS (knots indicated) | "Maintain two five zero knots" |
| Speed (above ~28,000 ft) | Mach number | "Maintain Mach point eight four" |
| Distance | Nautical miles (nm) | "Traffic one two o'clock, five miles" |
| Visibility | Statute miles (SM) | "Visibility three miles" |
| RVR | Feet | "RVR runway two seven left, two thousand four hundred" |
| Wind direction | Magnetic degrees | "Wind two five zero at one two" |
| Wind speed | Knots | "Gusts two zero" |
| Temperature | Celsius | "Temperature minus zero five" |
| Altimeter | Inches of mercury (inHg) | "Altimeter two niner niner two" |
| Time | UTC (Zulu) | "Expect further clearance at one four three zero zulu" |
| Heading | Magnetic degrees | "Fly heading two seven zero" |
| Vertical rate | Feet per minute | Pilot/controller: implied (e.g., "expedite descent") |
| Barometric pressure | Hectopascals (hPa) | International ICAO standard (not US domestic) |

### Number Pronunciation (ATC)

| Number | Pronunciation |
|---|---|
| 0 | "Zero" |
| 1 | "One" (or "Wun" in ICAO) |
| 2 | "Two" |
| 3 | "Three" (or "Tree" in ICAO) |
| 4 | "Four" (or "Fow-er" in ICAO) |
| 5 | "Five" (or "Fife" in ICAO) |
| 6 | "Six" |
| 7 | "Seven" |
| 8 | "Eight" (or "Ait" in ICAO) |
| 9 | "Nine" (or "Niner" in ICAO — always "niner" in US ATC) |
| 100 | "Hundred" |
| 1,000 | "Thousand" |
| . (decimal) | "Point" |

### Altitude Pronunciation Examples
- 3,000 ft: "three thousand"
- 12,000 ft: "one two thousand"
- FL350: "flight level three five zero"
- FL410: "flight level four one zero"

---

## 20. Simulation-Relevant Timing Constants

### Typical Event Durations

| Event | Duration | Notes |
|---|---|---|
| ATIS update cycle | Every 30-60 minutes | Or upon significant change |
| Frequency change + readback | 8-15 seconds | Time for instruction + readback |
| Takeoff clearance to airborne | 30-45 seconds | From "cleared for takeoff" to wheels up |
| Landing to runway clear | 45-75 seconds | Touchdown to clear of runway |
| Approach clearance to touchdown | 5-10 minutes | From vectors to final to landing |
| Sector transit time (en route) | 15-30 minutes | Average time in one ARTCC sector |
| TRACON transit time | 10-20 minutes | From handoff to tower |
| Ground taxi (major airport) | 10-25 minutes | Gate to runway or runway to gate |
| Standard rate turn | 3 degrees/second | 2-minute turn = 360 degrees |
| Descent planning | 3 nm per 1,000 ft | 3:1 rule for idle descent |
| ILS final approach (10 nm) | 3-4 minutes | At 160-180 KIAS groundspeed |
| Go-around to re-sequence | 10-20 minutes | Depends on traffic |
| Holding pattern (one circuit) | 4-6 minutes | 1-minute legs + turns |

### Controller Workload Limits

| Metric | Value | Notes |
|---|---|---|
| Max aircraft per sector | 15-20 | Typical operational limit |
| Max aircraft at high workload | 25+ | Requires reduced services |
| Communications per hour | 200-300+ transmissions | At busy facilities |
| Sector handoff frequency | Every 15-30 minutes per aircraft | En route |
| Landing rate (single runway) | 30-36 per hour | Peak; drops in adverse weather |

---

## 21. Common Waypoint / Fix Naming Conventions

### Waypoint Name Rules
- **5-letter pronounceable names**: MERIT, BOSTN, DIXIE, GRAVY, FRIED
- **Named after local geography**: JFKXX (near JFK), etc.
- **Humor/culture-based**: SPICY, WINGS, BRGRS (near certain cities)
- **Final approach fixes**: Often 5 letters, start with airport-relevant letters
- **Runway fixes**: "RW27L" format for runway thresholds

### Standard Fix Types (in approach context)
- **IAF**: Initial Approach Fix
- **IF**: Intermediate Fix
- **FAF**: Final Approach Fix
- **MAP**: Missed Approach Point
- **MAHP**: Missed Approach Holding Point

---

## Sources and References

- FAA Aeronautical Information Manual (AIM)
- FAA Order 7110.65 (Air Traffic Control)
- 14 CFR Part 91 (General Operating Rules)
- 14 CFR Part 121 (Air Carrier Operations)
- ICAO Annex 2 (Rules of the Air)
- ICAO Doc 4444 (Air Traffic Management)
- SKYbrary Aviation Safety
- FAA Aircraft Type Designators (ICAO Doc 8643)
- Boeing/Airbus aircraft specifications and performance manuals
