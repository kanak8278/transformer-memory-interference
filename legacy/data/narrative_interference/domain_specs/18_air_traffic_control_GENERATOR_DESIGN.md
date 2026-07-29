# Air Traffic Control Narrative Interference Generator — System Design

## 1. What We're Building

A function: `generate_atc_trial(num_keys, num_updates, condition, seed) → (narrative, question, expected_answer, entity_tracking)`

Where:
- `num_keys` = number of aircraft tracked in the narrative (2-20)
- `num_updates` = number of times each aircraft's tracked attribute changes (3-25)
- `condition` = "RI" (ask about first value) or "PI" (ask about last value)
- `seed` = random seed for reproducibility

## 2. Core Design Decisions

### Why ATC is uniquely suited for interference testing

ATC creates interference through **rigid structural similarity**. Every aircraft gets the same types of instructions in the same phraseology. "Delta 1472 descend and maintain flight level two four zero" vs "United 893 descend and maintain flight level two six zero." The model must track which callsign has which altitude — and altitudes are constantly changing as aircraft climb and descend through similar ranges.

This is structurally analogous to Dota's gold tracking, but the narrative voice is radically different — terse, standardized, no creative prose. Pure procedural language with numbers embedded in rigid sentence frames.

### What is the "tracked attribute"?

**Tracked attribute candidates (pool of 18+):**

**Numeric, continuous, constantly changing:**
- **altitude_ft** — best primary candidate. All jets in en route sector share FL200-FL410 range. In TRACON, all arrivals descend through 3,000-12,000 ft. Altitudes change with every climb/descend instruction. Maximum confusability.
- **speed_knots** — 150-350 KIAS. Changes with speed assignments. Tighter range during approach (160-250). Good confusability.
- **heading_degrees** — 0-360. Changes with heading assignments. Somewhat confusable but value space is large.

**Numeric, less frequent changes:**
- **assigned_altitude** — what ATC cleared them to (distinct from current altitude during transition).
- **assigned_speed** — what ATC told them to fly.
- **vertical_rate_fpm** — 1000-3500 ft/min climb/descent. Less frequently mentioned.
- **fuel_remaining_hrs** — decreasing. Only mentioned in emergencies or planning.
- **delay_minutes** — holds, sequencing delays. Integer, accumulates.
- **distance_from_airport_nm** — 0-150 nm. Decreasing for arrivals.
- **separation_from_nearest_nm** — 3-10 nm. Critical safety metric.

**Categorical (named values) — each pool must have 15+ distinct values for interference:**

- **clearance_status** (~22 values):
  awaiting_clearance_delivery, clearance_received_at_gate, pushback_approved,
  taxi_to_runway, holding_short_of_runway, line_up_and_wait,
  cleared_for_takeoff, airborne_initial_climb, contact_departure,
  climbing_via_SID, en_route_level_flight, descending_via_STAR,
  holding_at_fix, released_from_hold, radar_vectors_for_sequencing,
  cleared_ILS_approach, cleared_RNAV_approach, cleared_visual_approach,
  cleared_to_land, on_final_short, go_around_climbing,
  missed_approach_procedure, exiting_runway_contact_ground

- **approach_type** (~16 values):
  ILS_CAT_I, ILS_CAT_II, ILS_CAT_IIIa, ILS_CAT_IIIb,
  RNAV_GPS_LPV, RNAV_GPS_LNAV_VNAV, RNAV_GPS_LNAV_only,
  VOR_approach, VOR_DME_approach, LOC_only (localizer, no glideslope),
  visual_approach, charted_visual_approach, contact_approach,
  GLS_GBAS_approach, NDB_approach (rare, legacy),
  circling_approach, none_assigned

- **runway_assigned** — specific runway designator per airport config. Pool size 4-10 per airport:
  Examples from ORD: 10L, 10C, 10R, 9L, 9R, 27L, 28C, 28R, 32L, 14R, 14L, 15, 33, 22L, 22R, 4R, 4L
  Examples from ATL: 8L, 8R, 9L, 9R, 10, 26L, 26R, 27L, 27R, 28
  Examples from JFK: 4L, 4R, 13L, 13R, 22L, 22R, 31L, 31R
  Pool is airport-dependent; major hubs have 8-10+ runways.

- **flight_phase** (~18 values):
  pre_flight_at_gate, pushback_and_taxi_out, holding_short,
  takeoff_roll, initial_climb_below_1000, departure_climb_to_10000,
  departure_climb_above_10000, transition_to_cruise, cruise_level_flight,
  step_climb_in_cruise, top_of_descent_initiated, en_route_descent_via_STAR,
  terminal_descent_below_10000, base_turn_or_downwind, final_approach_course,
  short_final_inside_FAF, landing_roll_and_exit, taxi_to_gate

- **squawk_code** — 4-digit octal transponder code (0000-7777, digits 0-7 only). Unique per aircraft.
  Special codes: 7700 (emergency), 7600 (comm failure), 7500 (hijack), 1200 (VFR default).
  Pool is large (4096 possible codes); assigned randomly but avoiding special codes.

- **special_status** (~15 values):
  normal_operations, priority_handling_ATC_request, minimum_fuel_advisory,
  emergency_declared_engine_failure, emergency_declared_fire,
  emergency_declared_hydraulic_failure, emergency_declared_pressurization,
  emergency_declared_medical, emergency_declared_fuel,
  emergency_declared_bird_strike, emergency_declared_gear_malfunction,
  pan_pan_precautionary, comm_failure_squawk_7600,
  security_concern_squawk_7500, VIP_movement (Air Force One, etc.)

- **controller_instruction_type** (~20 values) — NEW attribute tracking what type of instruction was last given:
  altitude_assignment, speed_assignment, heading_assignment,
  direct_to_fix, descend_via_STAR, climb_via_SID,
  cross_fix_at_altitude, cross_fix_at_speed, cross_fix_at_altitude_and_speed,
  hold_instruction, expect_further_clearance, frequency_change,
  approach_clearance, landing_clearance, takeoff_clearance,
  line_up_and_wait, taxi_instruction, go_around_instruction,
  traffic_advisory, ident_request

- **weather_condition** (~18 values) — NEW, what conditions the aircraft is operating in:
  clear_VMC_unlimited, clear_above_scattered_layer, few_clouds_no_restriction,
  scattered_clouds_good_vis, broken_ceiling_3000ft, broken_ceiling_1500ft,
  overcast_ceiling_1000ft, overcast_ceiling_500ft, overcast_ceiling_200ft,
  fog_visibility_1SM, fog_visibility_half_SM, rain_moderate_vis_3SM,
  rain_heavy_vis_1SM, thunderstorm_in_vicinity, thunderstorm_on_field,
  snow_moderate_vis_2SM, icing_reported_in_clouds, turbulence_light_to_moderate

- **wake_turbulence_category** (6 values — **FAA RECAT system used throughout**):
  A (Super: A380),
  B (Upper Heavy: B747, B777-300ER, A340-600, C-5, C-17),
  C (Lower Heavy: B787, B777-200, A330, A350, B767),
  D (Upper Large: B757, B737-900, A321, MD-90),
  E (Lower Large: B737-700/800, A319, A320, E190),
  F (Small/Light: CRJ, E175, Dash 8, Citations, GA aircraft)
  Note: B757 is RECAT-D but produces Heavy-class wake. When B757 is the LEADING
  aircraft, trailing aircraft use one-category-stricter separation (row C).

- **STAR_assigned** (~15+ values per airport) — NEW, which arrival route:
  Examples from ORD: BENKY3, ENDEE4, GIPPER4, KRENA2, PANGG4, ROYKO1
  Examples from ATL: PECHY3, ERLIN3, FLCON4, HONIE3
  Examples from LAX: SADDE6, IRNMN2, RIKKN1, SEAVU5
  Pool is airport-dependent; each airport has 4-8 STARs from different directions.

- **SID_assigned** (~12+ values per airport) — NEW, which departure route:
  Examples from ORD: DUPAGE7, EARLS1, OHARE8, PETTY7
  Examples from JFK: DEEZZ5, GREKI1, SKORR4
  Pool is airport-dependent.

- **position_relative_to_fix** (~20+ values) — NEW compound categorical:
  "15 nm northwest of MERIT", "8 nm south of BOSTN", "on the KRENA3 arrival 40 nm out",
  "intercepting localizer 27L from the north", "established on ILS 28R, 6 nm final",
  "over GRAVY at FL240", "abeam DIXIE descending through FL180",
  "in the hold at JIMME, 3rd circuit", "on left downwind for 13L",
  "on right base for 27R", "2 nm final 28C", "crossing departure end of 9R climbing",
  "15 nm southwest of field at 8000", "entering Class B from the west",
  "over the numbers runway 25L", "missed approach climbing to 3000 heading 270"

**Compound (reported as string):**
- **position_report** — "20 miles southwest at FL280" — distance + direction + altitude. Combinatorial: 8 directions × 30 distances × 40 altitudes = huge unique pool.
- **traffic_advisory** — "traffic 2 o'clock, 5 miles, same altitude, Boeing 737." Combinatorial: 12 clock positions × 10 distances × altitude relationships × aircraft types.
- **last_ATC_exchange** — full controller-pilot exchange as a string. Every exchange is unique.

**Total: 25+ tracked attribute options.** Altitude and speed are the interference powerhouses for numeric. The expanded categoricals now have 15-22 values each, and compound/combinatorial attributes have effectively unlimited pools.

### Entity homogeneity: same-type vs mixed-type

**Same-type mode (~55% of trials):** All entities are commercial jets (B737, A320, B787, etc.). Similar performance envelopes → similar altitudes, speeds, climb rates. Values overlap heavily → maximum interference.

Same-type options:
- **All narrowbody jets** — B737/A320 family. Cruise FL350-FL390, approach 130-140 KIAS. Best interference.
- **All widebody jets** — B777/B787/A350. FL310-FL410, heavier wake, different separation rules.
- **All regional jets** — CRJ/E175. FL280-FL370, slower approach speeds.

**Mixed-type mode (~45% of trials):** Mix of narrowbodies, widebodies, regionals, turboprops, GA. Performance differences create some value divergence (turboprop cruises at FL250 vs jet at FL370), but approach speeds still overlap (120-155 KIAS). Wake turbulence categories add complexity.

Typical mixes:
- 8 narrowbodies + 2 widebodies + 2 regionals (typical TRACON mix)
- 5 jets + 3 turboprops + 2 GA (mixed-use airport)
- 10 narrowbodies + 3 widebodies + 2 heavies (major hub arrival rush)

### How do non-tracked attributes fit in?

Same principle: non-tracked attributes are filler.

> "14:32 — United 1523, descending through **flight level two eight zero** for **flight level one eight zero**, speed 310 knots. Assigned heading 270 for radar vectors ILS 28R. Traffic 2 o'clock, 5 miles, same altitude, Delta 445 also descending."

Here altitude (bold) is tracked. Speed, heading, approach clearance, and traffic advisory are filler.

## 2.1 TRACKABLE_ATTRIBUTES Specification

```python
TRACKABLE_ATTRIBUTES = [
    {
        "name": "altitude_ft",
        "type": "int",
        "range": [0, 45000],
        "format_str": "format_altitude(value)",  # see format_altitude() below
        "precision": 100,  # round to nearest 100 ft below FL180; 1000 ft at/above FL180
        "direction": "volatile",
        "update_events": ["altitude_assignment", "position_report", "go_around", "hold_instruction"],
        "interference_quality": "excellent",
        "notes": "PRIMARY candidate. Format CHANGES at FL180 transition. See format_altitude()."
    },
    {
        "name": "speed_knots",
        "type": "int_or_mach",
        "range": [110, 490],  # KIAS range; Mach 0.70-0.92 above FL280
        "format_str": "format_speed(value, altitude)",  # see format_speed() below
        "precision": 5,  # speed assigned in multiples of 5 (250, 210, etc.) or Mach 0.01
        "direction": "volatile",
        "update_events": ["speed_assignment", "position_report", "approach_clearance"],
        "interference_quality": "good",
        "notes": "Format changes at FL280: KIAS below, Mach above. See Section 2.3."
    },
    {
        "name": "heading_degrees",
        "type": "int",
        "range": [1, 360],
        "format_str": "f'{value:03d}'",  # always 3 digits: "270", "045"
        "precision": 5,  # headings assigned in multiples of 5 (rarely 1)
        "direction": "volatile",
        "update_events": ["heading_assignment", "go_around", "position_report"],
        "interference_quality": "good",
        "notes": "Larger value space (72 possible at precision=5) reduces interference vs altitude."
    },
    {
        "name": "assigned_altitude",
        "type": "int",
        "range": [0, 45000],
        "format_str": "format_altitude(value)",
        "precision": 1000,
        "direction": "volatile",
        "update_events": ["altitude_assignment", "hold_instruction", "go_around"],
        "interference_quality": "excellent",
        "notes": "What ATC cleared them to. Distinct from current altitude during transitions."
    },
    {
        "name": "distance_nm",
        "type": "float",
        "range": [0.0, 150.0],
        "format_str": "f'{value:.0f} miles'",
        "precision": 1,
        "direction": "mostly_down",  # arrivals approach; departures move away
        "update_events": ["position_report", "approach_clearance", "traffic_advisory"],
        "interference_quality": "moderate",
        "notes": "Distance from airport/fix. Monotonic for arrivals (decreasing)."
    },
    {
        "name": "clearance_status",
        "type": "str",
        "range": "22 categorical values (see Section 2 pool)",
        "format_str": "CLEARANCE_LABELS[value]",
        "precision": "N/A",
        "direction": "categorical",
        "update_events": ["approach_clearance", "landing_clearance", "takeoff_clearance",
                          "hold_instruction", "go_around", "frequency_change"],
        "interference_quality": "moderate",
        "notes": "Follows strict phase ordering. Values are domain-specific strings."
    },
    {
        "name": "approach_type",
        "type": "str",
        "range": "16 categorical values (see Section 2 pool)",
        "format_str": "APPROACH_LABELS[value]",
        "precision": "N/A",
        "direction": "categorical",
        "update_events": ["approach_clearance", "go_around"],
        "interference_quality": "moderate",
        "notes": "Constrained by weather. Only changes once or twice per aircraft."
    },
    {
        "name": "flight_phase",
        "type": "str",
        "range": "18 categorical values (see Section 2 pool)",
        "format_str": "PHASE_LABELS[value]",
        "precision": "N/A",
        "direction": "monotonic_forward",  # follows strict departure/arrival sequence
        "update_events": ["altitude_assignment", "approach_clearance", "landing_clearance",
                          "takeoff_clearance", "hold_instruction", "go_around", "handoff"],
        "interference_quality": "low",
        "notes": "Mostly monotonic progression, poor for interference."
    },
    {
        "name": "assigned_speed",
        "type": "int_or_mach",
        "range": [110, 490],
        "format_str": "format_speed(value, altitude)",
        "precision": 5,
        "direction": "volatile",
        "update_events": ["speed_assignment"],
        "interference_quality": "good",
        "notes": "What ATC told them. Distinct from indicated speed."
    },
    {
        "name": "vertical_rate_fpm",
        "type": "int",
        "range": [-3500, 3500],
        "format_str": "f'{abs(value)} feet per minute {'climb' if value > 0 else 'descent'}'",
        "precision": 100,
        "direction": "volatile",
        "update_events": ["altitude_assignment", "position_report"],
        "interference_quality": "moderate",
        "notes": "Negative = descending, positive = climbing, 0 = level."
    },
    {
        "name": "fuel_remaining_hrs",
        "type": "float",
        "range": [0.5, 20.0],
        "format_str": "f'{value:.1f} hours fuel remaining'",
        "precision": 0.1,
        "direction": "monotonic_down",
        "update_events": ["position_report", "hold_instruction", "emergency_declaration"],
        "interference_quality": "low",
        "notes": "Monotonic decrease. Only mentioned in planning or emergencies."
    },
    {
        "name": "delay_minutes",
        "type": "int",
        "range": [0, 120],
        "format_str": "f'{value} minute delay'",
        "precision": 5,
        "direction": "volatile",
        "update_events": ["hold_instruction", "weather_update"],
        "interference_quality": "moderate",
        "notes": "Accumulates in holds; resets on release."
    },
    {
        "name": "squawk_code",
        "type": "str",
        "range": "4-digit octal (0000-7777, digits 0-7 only)",
        "format_str": "f'{value}'",  # always 4 digits
        "precision": "N/A",
        "direction": "categorical",
        "update_events": ["emergency_declaration", "handoff"],
        "interference_quality": "low",
        "notes": "Rarely changes. Special codes: 7700, 7600, 7500, 1200."
    },
    {
        "name": "runway_assigned",
        "type": "str",
        "range": "airport-dependent pool (4-10 values)",
        "format_str": "f'runway {value}'",
        "precision": "N/A",
        "direction": "categorical",
        "update_events": ["approach_clearance", "landing_clearance", "takeoff_clearance",
                          "runway_change"],
        "interference_quality": "moderate",
        "notes": "Can change on runway change event. Pool size is airport-dependent."
    },
    {
        "name": "separation_nm",
        "type": "float",
        "range": [2.5, 20.0],
        "format_str": "f'{value:.1f} miles'",
        "precision": 0.5,
        "direction": "volatile",
        "update_events": ["position_report", "traffic_advisory"],
        "interference_quality": "moderate",
        "notes": "Separation from nearest traffic. Critical safety metric."
    },
]

# ── Altitude formatting function (CRITICAL: format changes at FL180) ──

def format_altitude(value_ft):
    """Format altitude per real ATC conventions.
    Below 18,000 ft: reported in feet, comma-separated thousands.
      e.g., 12000 -> "12,000 feet" or "one two thousand" in voice
    At/above 18,000 ft: reported as flight level (hundreds of feet on 29.92).
      e.g., 24000 -> "FL240" or "flight level two four zero" in voice
    """
    if value_ft >= 18000:
        fl = value_ft // 100
        return f"FL{fl}"
    else:
        return f"{value_ft:,}"


# ── Altitude rendering for transcript voice ──

def format_altitude_voice(value_ft):
    """For transcript voice: spoken altitude per ATC phraseology."""
    if value_ft >= 18000:
        fl = value_ft // 100
        digits = " ".join(DIGIT_WORDS[d] for d in str(fl))
        return f"flight level {digits}"
    else:
        # e.g., 12000 -> "one two thousand", 8000 -> "eight thousand"
        # 4500 -> "four thousand five hundred"
        thousands = value_ft // 1000
        hundreds = (value_ft % 1000) // 100
        parts = [f"{DIGIT_WORDS[str(thousands)[0]]}"]
        if len(str(thousands)) > 1:
            parts.append(DIGIT_WORDS[str(thousands)[1]])
        parts.append("thousand")
        if hundreds > 0:
            parts.append(f"{DIGIT_WORDS[str(hundreds)]} hundred")
        return " ".join(parts)


DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "niner",
}
```

---

## 2.2 AIRCRAFT_DATABASE

All values sourced from mechanics reference Section 5 and Section 8 (RECAT). RECAT categories used consistently throughout (see Major Decision: RECAT over ICAO).

```python
AIRCRAFT_DATABASE = {
    # ── Narrowbody Jets ──
    "B738": {
        "name": "Boeing 737-800",
        "icao_code": "B738",
        "wake_category": "E",  # RECAT: Lower Large
        "max_altitude_ft": 41000,
        "cruise_speed_mach": 0.789,
        "cruise_speed_kias": 460,
        "approach_speed_vref_range": (135, 142),
        "climb_rate_initial": (2800, 3400),
        "climb_rate_high_alt": (500, 1200),  # above FL350
        "descent_rate_range": (1500, 2500),
        "fuel_burn_cruise_lbs_hr": 4900,
        "fuel_burn_hold_lbs_hr": 6000,
    },
    "B739": {
        "name": "Boeing 737-900ER",
        "icao_code": "B739",
        "wake_category": "D",  # RECAT: Upper Large (heavier 737)
        "max_altitude_ft": 41000,
        "cruise_speed_mach": 0.789,
        "cruise_speed_kias": 460,
        "approach_speed_vref_range": (138, 146),
        "climb_rate_initial": (2600, 3200),
        "climb_rate_high_alt": (400, 1000),
        "descent_rate_range": (1500, 2500),
        "fuel_burn_cruise_lbs_hr": 5100,
        "fuel_burn_hold_lbs_hr": 6200,
    },
    "B38M": {
        "name": "Boeing 737 MAX 8",
        "icao_code": "B38M",
        "wake_category": "E",  # RECAT: Lower Large
        "max_altitude_ft": 41000,
        "cruise_speed_mach": 0.79,
        "cruise_speed_kias": 460,
        "approach_speed_vref_range": (134, 140),
        "climb_rate_initial": (3000, 3500),
        "climb_rate_high_alt": (500, 1200),
        "descent_rate_range": (1500, 2500),
        "fuel_burn_cruise_lbs_hr": 4500,
        "fuel_burn_hold_lbs_hr": 5500,
    },
    "A320": {
        "name": "Airbus A320",
        "icao_code": "A320",
        "wake_category": "E",  # RECAT: Lower Large
        "max_altitude_ft": 39800,
        "cruise_speed_mach": 0.78,
        "cruise_speed_kias": 450,
        "approach_speed_vref_range": (131, 137),
        "climb_rate_initial": (2800, 3400),
        "climb_rate_high_alt": (500, 1100),
        "descent_rate_range": (1500, 2500),
        "fuel_burn_cruise_lbs_hr": 4800,
        "fuel_burn_hold_lbs_hr": 5800,
    },
    "A20N": {
        "name": "Airbus A320neo",
        "icao_code": "A20N",
        "wake_category": "E",  # RECAT: Lower Large
        "max_altitude_ft": 39800,
        "cruise_speed_mach": 0.78,
        "cruise_speed_kias": 450,
        "approach_speed_vref_range": (131, 137),
        "climb_rate_initial": (3000, 3500),
        "climb_rate_high_alt": (500, 1200),
        "descent_rate_range": (1500, 2500),
        "fuel_burn_cruise_lbs_hr": 4200,
        "fuel_burn_hold_lbs_hr": 5100,
    },
    "A321": {
        "name": "Airbus A321",
        "icao_code": "A321",
        "wake_category": "D",  # RECAT: Upper Large
        "max_altitude_ft": 39800,
        "cruise_speed_mach": 0.78,
        "cruise_speed_kias": 450,
        "approach_speed_vref_range": (136, 142),
        "climb_rate_initial": (2500, 3200),
        "climb_rate_high_alt": (400, 1000),
        "descent_rate_range": (1500, 2500),
        "fuel_burn_cruise_lbs_hr": 5500,
        "fuel_burn_hold_lbs_hr": 6700,
    },
    "A21N": {
        "name": "Airbus A321neo",
        "icao_code": "A21N",
        "wake_category": "D",  # RECAT: Upper Large
        "max_altitude_ft": 39800,
        "cruise_speed_mach": 0.78,
        "cruise_speed_kias": 450,
        "approach_speed_vref_range": (136, 142),
        "climb_rate_initial": (2800, 3400),
        "climb_rate_high_alt": (500, 1100),
        "descent_rate_range": (1500, 2500),
        "fuel_burn_cruise_lbs_hr": 4800,
        "fuel_burn_hold_lbs_hr": 5800,
    },
    "B752": {
        "name": "Boeing 757-200",
        "icao_code": "B752",
        "wake_category": "D",  # RECAT: Upper Large (but Heavy wake — special separation rules)
        "max_altitude_ft": 42000,
        "cruise_speed_mach": 0.80,
        "cruise_speed_kias": 461,
        "approach_speed_vref_range": (130, 140),
        "climb_rate_initial": (3000, 3800),
        "climb_rate_high_alt": (600, 1500),
        "descent_rate_range": (1500, 3000),
        "fuel_burn_cruise_lbs_hr": 5800,
        "fuel_burn_hold_lbs_hr": 7000,
        "special_wake": True,  # B757 produces Heavy-class wake; trailing aircraft use Heavy separation
    },
    "B753": {
        "name": "Boeing 757-300",
        "icao_code": "B753",
        "wake_category": "D",  # RECAT: Upper Large (Heavy wake like B752)
        "max_altitude_ft": 42000,
        "cruise_speed_mach": 0.80,
        "cruise_speed_kias": 461,
        "approach_speed_vref_range": (135, 145),
        "climb_rate_initial": (2700, 3500),
        "climb_rate_high_alt": (500, 1300),
        "descent_rate_range": (1500, 3000),
        "fuel_burn_cruise_lbs_hr": 6200,
        "fuel_burn_hold_lbs_hr": 7500,
        "special_wake": True,
    },
    "B763": {
        "name": "Boeing 767-300ER",
        "icao_code": "B763",
        "wake_category": "C",  # RECAT: Lower Heavy
        "max_altitude_ft": 43100,
        "cruise_speed_mach": 0.80,
        "cruise_speed_kias": 459,
        "approach_speed_vref_range": (135, 145),
        "climb_rate_initial": (2500, 3500),
        "climb_rate_high_alt": (500, 1200),
        "descent_rate_range": (1500, 2500),
        "fuel_burn_cruise_lbs_hr": 8900,
        "fuel_burn_hold_lbs_hr": 10700,
    },
    "B772": {
        "name": "Boeing 777-200ER",
        "icao_code": "B772",
        "wake_category": "C",  # RECAT: Lower Heavy
        "max_altitude_ft": 43100,
        "cruise_speed_mach": 0.84,
        "cruise_speed_kias": 490,
        "approach_speed_vref_range": (140, 150),
        "climb_rate_initial": (2500, 3500),
        "climb_rate_high_alt": (500, 1200),
        "descent_rate_range": (1500, 2500),
        "fuel_burn_cruise_lbs_hr": 14000,
        "fuel_burn_hold_lbs_hr": 17000,
    },
    "B77W": {
        "name": "Boeing 777-300ER",
        "icao_code": "B77W",
        "wake_category": "B",  # RECAT: Upper Heavy
        "max_altitude_ft": 43100,
        "cruise_speed_mach": 0.84,
        "cruise_speed_kias": 490,
        "approach_speed_vref_range": (145, 155),
        "climb_rate_initial": (2200, 3200),
        "climb_rate_high_alt": (400, 1000),
        "descent_rate_range": (1500, 2500),
        "fuel_burn_cruise_lbs_hr": 15500,
        "fuel_burn_hold_lbs_hr": 18600,
    },
    "B789": {
        "name": "Boeing 787-9",
        "icao_code": "B789",
        "wake_category": "C",  # RECAT: Lower Heavy
        "max_altitude_ft": 43100,
        "cruise_speed_mach": 0.85,
        "cruise_speed_kias": 487,
        "approach_speed_vref_range": (140, 148),
        "climb_rate_initial": (2600, 3400),
        "climb_rate_high_alt": (500, 1200),
        "descent_rate_range": (1500, 2500),
        "fuel_burn_cruise_lbs_hr": 11500,
        "fuel_burn_hold_lbs_hr": 13800,
    },
    "A333": {
        "name": "Airbus A330-300",
        "icao_code": "A333",
        "wake_category": "C",  # RECAT: Lower Heavy
        "max_altitude_ft": 41100,
        "cruise_speed_mach": 0.82,
        "cruise_speed_kias": 472,
        "approach_speed_vref_range": (135, 145),
        "climb_rate_initial": (2500, 3500),
        "climb_rate_high_alt": (500, 1200),
        "descent_rate_range": (1500, 2500),
        "fuel_burn_cruise_lbs_hr": 11400,
        "fuel_burn_hold_lbs_hr": 13700,
    },
    "A359": {
        "name": "Airbus A350-900",
        "icao_code": "A359",
        "wake_category": "C",  # RECAT: Lower Heavy
        "max_altitude_ft": 43100,
        "cruise_speed_mach": 0.85,
        "cruise_speed_kias": 487,
        "approach_speed_vref_range": (135, 145),
        "climb_rate_initial": (2800, 3500),
        "climb_rate_high_alt": (500, 1200),
        "descent_rate_range": (1500, 2500),
        "fuel_burn_cruise_lbs_hr": 11000,
        "fuel_burn_hold_lbs_hr": 13200,
    },
    "A388": {
        "name": "Airbus A380-800",
        "icao_code": "A388",
        "wake_category": "A",  # RECAT: Super Heavy
        "max_altitude_ft": 43100,
        "cruise_speed_mach": 0.85,
        "cruise_speed_kias": 490,
        "approach_speed_vref_range": (145, 155),
        "climb_rate_initial": (1800, 2800),
        "climb_rate_high_alt": (300, 800),
        "descent_rate_range": (1500, 2000),
        "fuel_burn_cruise_lbs_hr": 22000,
        "fuel_burn_hold_lbs_hr": 26400,
    },
    "CRJ9": {
        "name": "CRJ-900",
        "icao_code": "CRJ9",
        "wake_category": "F",  # RECAT: Small/Light
        "max_altitude_ft": 41000,
        "cruise_speed_mach": 0.78,
        "cruise_speed_kias": 447,
        "approach_speed_vref_range": (132, 138),
        "climb_rate_initial": (2600, 3300),
        "climb_rate_high_alt": (400, 1000),
        "descent_rate_range": (1500, 2500),
        "fuel_burn_cruise_lbs_hr": 3500,
        "fuel_burn_hold_lbs_hr": 4200,
    },
    "E75L": {
        "name": "Embraer E175",
        "icao_code": "E75L",
        "wake_category": "F",  # RECAT: Small/Light
        "max_altitude_ft": 41000,
        "cruise_speed_mach": 0.78,
        "cruise_speed_kias": 445,
        "approach_speed_vref_range": (126, 132),
        "climb_rate_initial": (2800, 3400),
        "climb_rate_high_alt": (400, 1000),
        "descent_rate_range": (1500, 2500),
        "fuel_burn_cruise_lbs_hr": 3400,
        "fuel_burn_hold_lbs_hr": 4000,
    },
    "E170": {
        "name": "Embraer E170",
        "icao_code": "E170",
        "wake_category": "F",  # RECAT: Small/Light
        "max_altitude_ft": 41000,
        "cruise_speed_mach": 0.75,
        "cruise_speed_kias": 430,
        "approach_speed_vref_range": (124, 130),
        "climb_rate_initial": (3000, 3500),
        "climb_rate_high_alt": (400, 1000),
        "descent_rate_range": (1500, 2500),
        "fuel_burn_cruise_lbs_hr": 3200,
        "fuel_burn_hold_lbs_hr": 3800,
    },
    "DH8D": {
        "name": "Dash 8-Q400",
        "icao_code": "DH8D",
        "wake_category": "F",  # RECAT: Small/Light (turboprop)
        "max_altitude_ft": 27000,
        "cruise_speed_mach": None,  # turboprop — no Mach reference
        "cruise_speed_kias": 360,
        "approach_speed_vref_range": (110, 120),
        "climb_rate_initial": (1500, 2200),
        "climb_rate_high_alt": (500, 1000),
        "descent_rate_range": (1000, 1800),
        "fuel_burn_cruise_lbs_hr": 3100,
        "fuel_burn_hold_lbs_hr": 3700,
    },
    "AT76": {
        "name": "ATR 72-600",
        "icao_code": "AT76",
        "wake_category": "F",  # RECAT: Small/Light (turboprop)
        "max_altitude_ft": 25000,
        "cruise_speed_mach": None,
        "cruise_speed_kias": 275,
        "approach_speed_vref_range": (105, 115),
        "climb_rate_initial": (1200, 1800),
        "climb_rate_high_alt": (400, 800),
        "descent_rate_range": (1000, 1500),
        "fuel_burn_cruise_lbs_hr": 1900,
        "fuel_burn_hold_lbs_hr": 2300,
    },
}

# ── RECAT Separation Matrix (nm, approach/landing) ──
# Usage: RECAT_SEPARATION[leading_cat][trailing_cat] → minimum nm
RECAT_SEPARATION = {
    "A": {"A": 6.0, "B": 7.0, "C": 7.0, "D": 7.0, "E": 8.0, "F": 8.0},
    "B": {"A": 4.0, "B": 4.0, "C": 5.0, "D": 5.0, "E": 6.0, "F": 7.0},
    "C": {"A": 4.0, "B": 3.5, "C": 4.0, "D": 4.0, "E": 5.0, "F": 6.0},
    "D": {"A": 3.0, "B": 3.0, "C": 3.0, "D": 3.0, "E": 4.0, "F": 5.0},
    "E": {"A": 3.0, "B": 3.0, "C": 3.0, "D": 3.0, "E": 3.0, "F": 4.0},
    "F": {"A": 3.0, "B": 3.0, "C": 3.0, "D": 3.0, "E": 3.0, "F": 3.0},
}
# Note: B757 (RECAT D) uses special_wake=True flag — when B757 is LEADING,
# apply one category stricter separation (use row C instead of row D).

# ── RECAT Departure Time Separation (seconds, same runway) ──
RECAT_DEPARTURE_SEPARATION = {
    "A": {"A": 120, "B": 120, "C": 140, "D": 160, "E": 160, "F": 180},
    "B": {"A": 100, "B": 100, "C": 120, "D": 120, "E": 140, "F": 160},
    "C": {"A": 80,  "B": 80,  "C": 100, "D": 100, "E": 120, "F": 120},
    "D": {"A": 80,  "B": 80,  "C": 80,  "D": 80,  "E": 100, "F": 100},
    "E": {"A": 60,  "B": 60,  "C": 60,  "D": 60,  "E": 60,  "F": 80},
    "F": {"A": 60,  "B": 60,  "C": 60,  "D": 60,  "E": 60,  "F": 60},
}
```

---

## 2.3 Speed Unit Conventions

Speed representation in ATC changes with altitude. The state machine, narrative renderer, and entity tracking must all be consistent.

```
STATE MACHINE INTERNAL REPRESENTATION:
  - Speed is ALWAYS stored as KIAS (knots indicated airspeed) in the state dict.
  - This is the single source of truth.
  - Mach is a derived display format, NOT stored separately.

NARRATIVE RENDERING RULES:
  Below FL280 (< 28,000 ft):
    Transcript: "two five zero knots"
    Log: "250 knots"
    Entity tracking format: "250" (bare number, unit=knots implied)

  At/above FL280 (>= 28,000 ft):
    Transcript: "Mach point eight four"
    Log: "M0.84" or "Mach 0.84"
    Entity tracking format: "M0.84"

  Conversion at transition (FL280):
    Speed assignment transitions from KIAS to Mach when climbing through FL280,
    and from Mach to KIAS when descending through FL280.

APPROXIMATE KIAS ↔ MACH CONVERSION TABLE BY ALTITUDE:
  (Standard atmosphere; Mach = KTAS / speed_of_sound; KIAS ~ KTAS at low alt)
  FL280: M0.78 ≈ 280 KIAS, M0.80 ≈ 287 KIAS, M0.84 ≈ 301 KIAS
  FL310: M0.78 ≈ 265 KIAS, M0.80 ≈ 271 KIAS, M0.84 ≈ 285 KIAS
  FL350: M0.78 ≈ 252 KIAS, M0.80 ≈ 258 KIAS, M0.84 ≈ 271 KIAS
  FL390: M0.78 ≈ 240 KIAS, M0.80 ≈ 246 KIAS, M0.84 ≈ 258 KIAS
  FL410: M0.78 ≈ 234 KIAS, M0.80 ≈ 240 KIAS, M0.84 ≈ 252 KIAS

  Simplified conversion function:
    def kias_to_mach(kias, altitude_ft):
        """Approximate. For narrative generation, not flight planning."""
        # Speed of sound decreases with altitude (standard atmo)
        # At FL350 (ISA): ~573 KTAS per Mach 1
        sos_ktas = {
            28000: 594, 31000: 586, 33000: 580, 35000: 574,
            37000: 574, 39000: 574, 41000: 574,
        }
        # KIAS ≈ KTAS * sqrt(rho/rho0) — use lookup
        # Simplified: at FL350, KIAS ≈ KTAS * 0.69
        correction = {
            28000: 0.76, 31000: 0.73, 33000: 0.71, 35000: 0.69,
            37000: 0.67, 39000: 0.65, 41000: 0.63,
        }
        fl = min(sorted(correction.keys()), key=lambda x: abs(x - altitude_ft))
        ktas = kias / correction[fl]
        mach = ktas / sos_ktas[fl]
        return round(mach, 2)

    def mach_to_kias(mach, altitude_ft):
        """Inverse of above."""
        sos_ktas = {28000: 594, 31000: 586, 35000: 574, 39000: 574, 41000: 574}
        correction = {28000: 0.76, 31000: 0.73, 35000: 0.69, 39000: 0.65, 41000: 0.63}
        fl = min(sorted(sos_ktas.keys()), key=lambda x: abs(x - altitude_ft))
        ktas = mach * sos_ktas[fl]
        kias = ktas * correction[fl]
        return int(round(kias / 5) * 5)  # round to nearest 5

ENTITY TRACKING FORMAT:
  When tracked_attribute is "speed_knots":
    - Below FL280: record as "250" (integer KIAS)
    - At/above FL280: record as "M0.84" (Mach string)
    - The RI/PI expected_answer uses the format that was active at the time of
      that particular update. First update at FL350 → "M0.82". Last update at
      8,000 ft → "210".
```

---

## 3. State Machine Design

### 3.1 Scenario Initialization

```
Input: num_keys aircraft, seed
Output: initial state for all aircraft + sector metadata

Steps:
1. Pick facility type: TRACON (60%) or ARTCC (40%)
2. Pick scenario type: arrival_rush / departure_push / mixed_ops / en_route_sector
3. Pick aircraft type mode: same-type (55%) or mixed-type (45%)
4. For each aircraft:
   - Generate callsign: [airline][flight_number] (e.g., "DAL1472", "UAL893", "SWA2241")
   - Pick aircraft type (B738, A320, B789, CRJ9, etc.)
   - Set wake turbulence category (RECAT A-F, from AIRCRAFT_DATABASE)
   - Set initial state based on scenario:
     - Arrival: position 50-150 nm out, at cruise altitude, assigned STAR
     - Departure: at gate or taxiing, initial clearance received
     - En route: at assigned flight level, cruising
   - Set origin and destination airports
5. Pick airport from pool (major US airports with full runway configs)
6. Pick weather conditions (ATIS info: wind, visibility, ceiling, altimeter)
7. Generate runway configuration based on wind
```

### 3.1.1 Weather State Generation

Each weather category maps to specific meteorological value ranges. The weather state is generated at scenario initialization and drives all downstream operations (runway selection, approach types, holding probability, go-around rate).

```python
WEATHER_STATE_RANGES = {
    "vfr_clear": {
        "wind_speed_kt": (5, 15),
        "wind_gust_kt": None,  # no gusts
        "visibility_sm": (10, 10),  # 10+ SM, report as "10" or "unlimited"
        "ceiling_type": "FEW",  # FEW or CLR
        "ceiling_ft_agl": (5000, 25000),  # FEW above 5000, or clear
        "altimeter_inhg": (29.85, 30.20),
        "rvr": None,  # not reported in VFR
        "precipitation": None,
        "remarks": ["visual approaches in use"],
        "approach_types_available": ["all"],  # ILS, RNAV, VOR, visual, contact
        "visual_approaches": True,
        "go_around_rate": 0.01,
        "holding_probability": 0.0,
    },
    "mvfr": {
        "wind_speed_kt": (8, 20),
        "wind_gust_kt": (None, 28),  # optional gusts, up to 28 kt
        "visibility_sm": (3, 5),
        "ceiling_type": "BKN",  # BKN (broken)
        "ceiling_ft_agl": (1500, 3000),
        "altimeter_inhg": (29.70, 30.10),
        "rvr": None,
        "precipitation": ["light rain", "mist", None],  # sample one
        "remarks": ["ILS approaches in use, visual approaches available"],
        "approach_types_available": ["ILS_CAT_I", "RNAV_GPS_LPV", "RNAV_GPS_LNAV_VNAV", "visual"],
        "visual_approaches": True,  # if pilot has airport in sight at MDA
        "go_around_rate": 0.035,
        "holding_probability": 0.10,
    },
    "ifr_low": {
        "wind_speed_kt": (10, 25),
        "wind_gust_kt": (20, 35),  # gusts likely
        "visibility_sm": (1, 3),
        "ceiling_type": "BKN",  # BKN or OVC
        "ceiling_ft_agl": (500, 1000),
        "altimeter_inhg": (29.50, 30.00),
        "rvr": None,  # RVR not required for CAT I
        "precipitation": ["rain", "drizzle", "mist", "fog"],
        "remarks": ["ILS approaches in use, ceiling {ceiling} broken, visibility {vis}"],
        "approach_types_available": ["ILS_CAT_I", "RNAV_GPS_LPV"],
        "visual_approaches": False,
        "go_around_rate": 0.075,
        "holding_probability": 0.30,
    },
    "ifr_cat2": {
        "wind_speed_kt": (10, 20),
        "wind_gust_kt": (18, 30),
        "visibility_sm": (0.5, 1.0),
        "ceiling_type": "OVC",  # overcast
        "ceiling_ft_agl": (200, 500),
        "altimeter_inhg": (29.50, 30.00),
        "rvr": (1200, 2400),  # RVR in feet
        "precipitation": ["fog", "drizzle", "light rain"],
        "remarks": ["ILS CAT II approaches in use, RVR runway {rwy} {rvr}"],
        "approach_types_available": ["ILS_CAT_II"],
        "visual_approaches": False,
        "go_around_rate": 0.15,
        "holding_probability": 0.50,
    },
    "ifr_cat3": {
        "wind_speed_kt": (5, 15),  # calm winds needed for CAT III
        "wind_gust_kt": None,  # gusts incompatible with CAT III
        "visibility_sm": (0.0, 0.25),  # near-zero or zero
        "ceiling_type": "OVC",
        "ceiling_ft_agl": (0, 200),  # 0 = indefinite ceiling
        "altimeter_inhg": (29.60, 30.00),
        "rvr": (150, 1200),  # RVR in feet; <700 = CAT IIIb, 700-1200 = CAT IIIa
        "precipitation": ["fog", "dense fog"],
        "remarks": ["CAT III ILS approaches in use, RVR runway {rwy} {rvr}"],
        "approach_types_available": ["ILS_CAT_IIIa", "ILS_CAT_IIIb"],
        "visual_approaches": False,
        "go_around_rate": 0.20,
        "holding_probability": 0.70,
        "special_rules": [
            "Only autoland-capable aircraft may attempt approach",
            "Non-equipped aircraft must hold or divert",
            "Low-visibility taxi procedures in effect",
            "Single-runway operations likely (reduced capacity)",
        ],
    },
    "thunderstorm": {
        "wind_speed_kt": (15, 40),
        "wind_gust_kt": (30, 60),  # gusts to 60 kt possible
        "visibility_sm": (1, 5),  # highly variable
        "ceiling_type": "variable",  # CB (cumulonimbus) reported
        "ceiling_ft_agl": (500, 5000),  # variable with storm cells
        "altimeter_inhg": (29.40, 29.90),  # falling pressure
        "rvr": None,  # variable, depends on rain intensity
        "precipitation": ["thunderstorm", "heavy rain", "hail possible"],
        "remarks": ["thunderstorms in vicinity, wind shear advisories in effect",
                     "LLWAS alert, microburst advisories in effect"],
        "approach_types_available": ["ILS_CAT_I"],  # if storm not on field
        "visual_approaches": False,
        "go_around_rate": 0.30,
        "holding_probability": 0.80,
        "special_rules": [
            "Aircraft requesting deviations around cells",
            "Ground stops possible (no departures)",
            "SIGMET active for convective activity",
            "Wind shear PIREPs likely",
        ],
    },
}

def generate_weather_state(weather_category, airport, rng):
    """Generate a complete weather state from category ranges."""
    cfg = WEATHER_STATE_RANGES[weather_category]

    wind_speed = rng.randint(*cfg["wind_speed_kt"])
    wind_gust = None
    if cfg["wind_gust_kt"] is not None:
        if cfg["wind_gust_kt"][0] is None:
            wind_gust = rng.randint(wind_speed + 5, cfg["wind_gust_kt"][1]) if rng.random() < 0.4 else None
        else:
            wind_gust = rng.randint(*cfg["wind_gust_kt"])

    # Wind direction: sample from full 360, but bias toward directions
    # compatible with the airport's runway configuration.
    # Primary runway heading determines most-likely wind direction (within ±30 deg).
    airport_cfg = AIRPORTS[airport]
    primary_rwy_heading = get_primary_runway_heading(airport_cfg)
    wind_dir = int((primary_rwy_heading + rng.randint(-30, 30)) % 360)
    if wind_dir == 0:
        wind_dir = 360

    visibility = round(rng.uniform(*cfg["visibility_sm"]), 1)
    if visibility >= 10:
        visibility = 10  # report as "10" or "greater than 10"

    ceiling_ft = rng.randint(*cfg["ceiling_ft_agl"]) if cfg["ceiling_ft_agl"][0] > 0 else 0
    ceiling_ft = (ceiling_ft // 100) * 100  # round to nearest 100

    altimeter = round(rng.uniform(*cfg["altimeter_inhg"]), 2)

    rvr = None
    if cfg["rvr"] is not None:
        rvr = rng.randint(*cfg["rvr"])
        rvr = (rvr // 100) * 100  # RVR reported in 100-ft increments

    precip = None
    if cfg["precipitation"]:
        precip = rng.choice([p for p in cfg["precipitation"] if p is not None]) \
                 if any(p is not None for p in cfg["precipitation"]) else None

    return {
        "category": weather_category,
        "wind_dir": wind_dir,
        "wind_speed": wind_speed,
        "wind_gust": wind_gust,
        "visibility_sm": visibility,
        "ceiling_type": cfg["ceiling_type"],
        "ceiling_ft_agl": ceiling_ft,
        "altimeter_inhg": altimeter,
        "rvr": rvr,
        "precipitation": precip,
        "approach_types_available": cfg["approach_types_available"],
        "visual_approaches": cfg["visual_approaches"],
        "go_around_rate": cfg["go_around_rate"],
        "holding_probability": cfg["holding_probability"],
    }
```

### 3.2 Aircraft State Initialization by Scenario

```python
SCENARIO_PROFILES = {
    "arrival_rush": {
        # All aircraft inbound, descending, sequencing for approach
        "initial_altitude": {"range": (18000, 39000)},  # feet, spread across STAR
        "initial_speed": {"range": (250, 310)},  # knots
        "initial_distance_nm": {"range": (30, 120)},  # from airport
        "flight_phase": "descent",
        "clearance": "descend_via_star",
        # Arrivals are staggered by distance: closest at lowest altitude
    },
    "departure_push": {
        # All aircraft departing, climbing, diverging
        "initial_altitude": {"range": (0, 5000)},
        "initial_speed": {"range": (0, 250)},
        "flight_phase": "departure",
        "clearance": "climb_via_sid",
    },
    "mixed_ops": {
        # Arrivals and departures interleaved — most realistic
        # 60% arrivals, 40% departures typically
        "arrival_pct": 0.60,
    },
    "en_route_sector": {
        # All aircraft at cruise, crossing sector
        # Conflicts arise from crossing traffic at similar altitudes
        "initial_altitude": {"range": (28000, 41000)},  # flight levels
        "initial_speed": {"range": (440, 490)},  # KTAS / Mach
        "flight_phase": "cruise",
        "clearance": "maintain_fl",
    },
    "holding_pattern": {
        # Weather delay — multiple aircraft stacked in holds
        # Stacked at different altitudes (1000 ft apart)
        "initial_altitude": {"stacked": True, "base": 7000, "increment": 1000},
        "flight_phase": "holding",
        "clearance": "hold",
    },
    "emergency_scenario": {
        # Normal ops disrupted by one aircraft declaring emergency
        # Others rerouted/held/delayed
        "emergency_aircraft": 1,  # one aircraft declares
        "others": "normal_arrival_or_departure",
    },
}
```

### 3.3 Event Types

The ATC narrative is a sequence of controller-pilot interactions and radar observations. Each event updates 1-2 aircraft states.

```
EVENT TYPES:
├── altitude_assignment     — "descend and maintain FL240" / "climb and maintain FL350"
├── speed_assignment        — "reduce speed to 250 knots" / "maintain 310 knots"
├── heading_assignment      — "fly heading 270" / "turn left heading 180"
├── approach_clearance      — "cleared ILS runway 27L approach" / "cleared visual approach"
├── landing_clearance       — "runway 27L, cleared to land"
├── takeoff_clearance       — "runway 27R, cleared for takeoff"
├── frequency_change        — "contact approach on 124.3" / "contact center 132.45"
├── traffic_advisory        — "traffic 2 o'clock, 5 miles, same altitude, Boeing 737"
├── position_report         — radar or pilot report of current position/altitude/speed
├── hold_instruction        — "hold east of MERIT, right turns, expect further clearance 1430Z"
├── go_around               — "go around, fly heading 270, climb 3000"
├── emergency_declaration   — "declaring emergency, engine failure" / "minimum fuel"
├── weather_update          — ATIS change, wind shift, visibility change
├── runway_change           — "new ATIS, landing runway changed to 28L"
├── handoff                 — aircraft transfers between sectors/facilities
└── taxi_instruction        — "taxi to runway 27L via alpha, bravo"
```

### 3.4 Event Generation (the state machine)

```python
def generate_next_event(state, scenario_time, scenario_type, rng):
    """Generate next ATC event based on current traffic state."""

    if scenario_type == "arrival_rush":
        weights = {
            "altitude_assignment": 25,      # constant descent management
            "speed_assignment": 15,          # sequencing requires speed control
            "heading_assignment": 12,        # vectors for spacing
            "approach_clearance": 10,        # when close enough
            "landing_clearance": 5,          # final
            "frequency_change": 8,           # center → approach → tower
            "traffic_advisory": 8,           # separation awareness
            "position_report": 10,           # radar updates
            "hold_instruction": 3,           # if backed up
            "go_around": 2,                  # rare but happens
            "weather_update": 2,             # ATIS changes
        }
    elif scenario_type == "departure_push":
        weights = {
            "altitude_assignment": 20,       # climb clearances
            "speed_assignment": 10,
            "heading_assignment": 15,        # departure headings for separation
            "takeoff_clearance": 15,         # the core event
            "frequency_change": 10,          # tower → departure → center
            "traffic_advisory": 5,
            "position_report": 8,
            "taxi_instruction": 10,          # ground movement
            "weather_update": 2,
            "handoff": 5,
        }
    elif scenario_type == "en_route_sector":
        weights = {
            "altitude_assignment": 25,       # step climbs + crossing traffic + fuel-burn step climbs (merged)
            "speed_assignment": 8,           # Mach number adjustments
            "heading_assignment": 5,         # rare in cruise
            "frequency_change": 15,          # sector handoffs
            "traffic_advisory": 12,          # crossing traffic awareness
            "position_report": 15,           # periodic updates
            "handoff": 15,                   # sector transitions
            "weather_update": 5,             # ride reports, turbulence
        }
    elif scenario_type == "mixed_ops":
        weights = {
            "altitude_assignment": 20,
            "speed_assignment": 12,
            "heading_assignment": 12,
            "approach_clearance": 8,
            "landing_clearance": 5,
            "takeoff_clearance": 8,
            "frequency_change": 8,
            "traffic_advisory": 8,
            "position_report": 8,
            "hold_instruction": 2,
            "go_around": 1,
            "weather_update": 3,
            "taxi_instruction": 5,
        }
    elif scenario_type == "holding_pattern":
        weights = {
            "hold_instruction": 20,          # assign/update holds
            "altitude_assignment": 15,       # stack management
            "position_report": 15,           # "how long in hold"
            "approach_clearance": 10,        # releasing from hold
            "speed_assignment": 5,
            "frequency_change": 5,
            "weather_update": 15,            # weather is WHY they're holding
            "traffic_advisory": 5,
            "emergency_declaration": 5,      # fuel becomes issue
            "landing_clearance": 5,
        }

    elif scenario_type == "emergency_scenario":
        # Emergency aircraft gets rapid altitude/speed/heading changes (priority vectors).
        # Non-emergency aircraft get holds and traffic advisories as they are displaced.
        weights = {
            "altitude_assignment": 22,       # emergency descent or climb; others re-leveled
            "speed_assignment": 12,          # emergency aircraft speed adjustments
            "heading_assignment": 15,        # emergency vectors to runway
            "approach_clearance": 8,         # emergency cleared straight-in
            "landing_clearance": 5,          # emergency lands first
            "frequency_change": 3,           # minimal — emergency stays on freq
            "traffic_advisory": 10,          # all aircraft advised of emergency traffic
            "position_report": 5,            # confirming emergency aircraft position
            "hold_instruction": 12,          # other aircraft held while emergency proceeds
            "go_around": 3,                  # aircraft on approach sent around for emergency
            "emergency_declaration": 2,      # the initial declaration event (once per scenario)
            "weather_update": 1,             # low priority during emergency
            "handoff": 2,                    # minimal — emergency traffic stays with one controller
        }

    # Filter impossible events:
    # Can't clear to land if not on approach
    # Can't issue takeoff if not at runway
    # Can't issue approach clearance if too far out
    # Landed aircraft generate no more events
    # Can't assign altitude above aircraft's service ceiling
    # Speed assignments must respect regulatory limits (250 below 10,000)

    event_type = weighted_random_choice(weights, rng)
    event = create_event(event_type, state, scenario_time, rng)
    return event
```

### 3.5 Event Handlers (Constraints)

Each event handler enforces ATC rules and physics:

```
ALTITUDE_ASSIGNMENT:
  - Pick an aircraft that needs altitude change
  - Issue: "descend and maintain [altitude]" or "climb and maintain [altitude]"
  - Update: assigned_altitude = new altitude
  - Aircraft transitions: current_altitude drifts toward assigned at climb/descent rate
  - Constraints:
    - Vertical separation: ≥1,000 ft between aircraft (below FL410)
    - Above FL410: ≥2,000 ft separation
    - Can't assign altitude above service ceiling
    - Flight levels follow semicircular rule:
      - Eastbound (0-179°): odd FL (FL290, FL310, FL330...)
      - Westbound (180-359°): even FL (FL300, FL320, FL340...)
    - Below 18,000: assign feet, not flight levels
    - Descent rate: 1,500-3,000 ft/min typical
    - Climb rate: 2,000-3,500 ft/min initial, decreasing with altitude
    - "Cross [fix] at [altitude]" — altitude must be reached by that point
    - "Descend via STAR" — follow all published restrictions

SPEED_ASSIGNMENT:
  - Pick an aircraft needing speed adjustment (sequencing, separation)
  - Issue: "reduce speed to [speed] knots" or "maintain [speed] knots"
  - Update: assigned_speed = new speed, aircraft transitions toward it
  - Constraints:
    - Below 10,000 ft MSL: max 250 KIAS (regulatory, ATC cannot waive)
    - Within Class B below/at 2,500 ft AGL within 4nm: max 200 KIAS
    - Minimum speed for jets: ~150 KIAS (unless on final)
    - Minimum speed for turboprops: ~120 KIAS
    - Speed changes of 10-30 knots per assignment (incremental)
    - On approach: progressive deceleration (250 → 210 → 180 → 170 → approach speed)

HEADING_ASSIGNMENT:
  - Pick an aircraft needing heading change (vectoring for spacing, sequencing)
  - Issue: "fly heading [degrees]" or "turn left/right heading [degrees]"
  - Update: assigned_heading = new heading
  - Constraints:
    - Heading in degrees (000-360)
    - Standard rate turn: 3°/second
    - Turn direction specified (left/right) or shortest path
    - On approach: heading assignments vector toward final approach course
    - Intercept angle to final: typically 20-45 degrees

APPROACH_CLEARANCE:
  - Pick an aircraft within 15-30 nm, properly sequenced
  - Issue: "cleared [approach_type] runway [runway] approach"
  - Update: clearance_status = "cleared_approach", approach_type, runway_assigned
  - Constraints:
    - Must be at or below intermediate altitude for approach
    - Must be on or approaching the approach course
    - Separation from preceding aircraft on approach (RECAT-based):
      - Use RECAT_SEPARATION[leading_cat][trailing_cat] from Section 2.2
      - Example: B behind E (Upper Heavy leading, Lower Large trailing): 6 nm
      - Example: E behind E (Lower Large behind Lower Large): 3 nm
      - B757 exception: when B757 leads, use row C (not row D)
    - Can't clear two aircraft for approach to same runway without adequate spacing
    - ILS requires specific intercept conditions (at or below glideslope, on localizer)

LANDING_CLEARANCE:
  - Pick aircraft on final approach, within ~3 nm of runway
  - Issue: "runway [number], cleared to land, wind [direction] at [speed]"
  - Update: clearance_status = "cleared_to_land"
  - Constraints:
    - Preceding aircraft must have cleared the runway
    - Wind reported with clearance
    - If runway not clear: "continue approach" or "go around"

TAKEOFF_CLEARANCE:
  - Pick aircraft holding short of runway
  - Issue: "runway [number], cleared for takeoff, wind [direction] at [speed]"
  - Update: flight_phase = "departure", clearance_status = "cleared_takeoff"
  - Constraints:
    - Runway must be clear (no aircraft on it)
    - Wake turbulence departure separation: use RECAT_DEPARTURE_SEPARATION (Section 2.2)
      - B757 exception: when B757 just departed, use row C separation for trailing aircraft
    - Same runway: preceding departure must be airborne and beyond departure end
    - Intersecting runways: departing aircraft must not conflict

FREQUENCY_CHANGE:
  - Pick aircraft transitioning between facilities
  - Issue: "contact [facility] on [frequency]"
  - Update: current_frequency = new frequency
  - Constraints:
    - Sequence: clearance → ground → tower → departure → center → approach → tower → ground
    - Each transition has a characteristic altitude/distance/phase trigger
    - Aircraft must be in correct phase for the transition

TRAFFIC_ADVISORY:
  - Pick an aircraft with nearby traffic
  - Issue: "traffic [clock position], [distance] miles, [direction], [aircraft type], [altitude]"
  - Update: no state change (informational only)
  - Constraints:
    - Clock position relative to aircraft's heading
    - Only advisory for traffic within ~10 nm and ±2,000 ft
    - Include type if relevant (wake turbulence awareness)
    - Pilot response: "traffic in sight" or "looking for traffic" or "negative contact"

POSITION_REPORT:
  - Pick an aircraft
  - Document current observed state: altitude, speed, heading, position
  - Update: confirms aircraft is where expected (or flags deviation)
  - Used to refresh tracked attribute values in the narrative

HOLD_INSTRUCTION:
  - Pick an aircraft that needs to hold (sequencing delay, weather, emergency ahead)
  - Issue: "hold [direction] of [fix], [turns], expect further clearance [time]"
  - Update: flight_phase = "holding", clearance_status = "hold"
  - Constraints:
    - Holding altitudes stacked 1,000 ft apart
    - Speed limits in hold (200/230/265 KIAS by altitude)
    - EFC time must be provided
    - Fuel monitoring critical (holding burns 20-30% more than cruise)

GO_AROUND:
  - Pick an aircraft on final approach
  - Issue: "go around, fly heading [heading], climb and maintain [altitude]"
  - Update: flight_phase = "missed_approach", altitude climbing, heading changing
  - Causes: runway not clear, spacing too tight, windshear, preceding aircraft slow to exit
  - Constraints:
    - Must assign heading and altitude for separation
    - Aircraft will re-enter sequence (10-20 min delay)
    - Increases workload significantly

EMERGENCY_DECLARATION:
  - Pick one aircraft (rare event, max 1 per scenario)
  - Update: special_status = "emergency", squawk = 7700
  - Priority handling: other aircraft delayed/rerouted
  - Types: engine failure, fuel emergency, medical, pressurization
  - Constraints:
    - Emergency aircraft gets priority (shortest path to runway)
    - Other aircraft may be held, rerouted, or delayed
    - ARFF standby activated

WEATHER_UPDATE:
  - Affects all aircraft (global event)
  - Update: ATIS changes — wind shift, visibility change, ceiling change
  - May trigger: runway change, approach type change, holding
  - Constraints:
    - Wind shift > 30° or speed change > 5kt → new ATIS
    - Visibility dropping below minimums → approach type upgrade (VOR → ILS) or hold
    - Ceiling dropping → category change (CAT I → CAT II or holding)

HANDOFF:
  - Pick aircraft transitioning between sectors
  - Issue: "contact [next sector] on [frequency]"
  - Update: current_sector changes, frequency changes
  - Constraints: aircraft must be near sector boundary
```

### 3.6 Attribute Trajectory Constraints (Critical for Realism)

**Altitude trajectories must follow physics and ATC rules:**

```
ARRIVAL (TRACON, narrowbody jet, B738):
  120 nm out: FL370 (cruise altitude)
  100 nm:     FL310 (top of descent, descending via STAR)
  80 nm:      FL240 ("descend and maintain FL240")
  60 nm:      FL180 (transition to altimeter, "descend and maintain 18,000")
  45 nm:      12,000 ft ("cross FIXXX at 12,000")
  30 nm:      8,000 ft ("descend and maintain 8,000")
  20 nm:      5,000 ft (vectors for ILS)
  15 nm:      4,000 ft (intercept glideslope)
  10 nm:      3,000 ft (on ILS glideslope, 3° = ~300 ft/nm)
  5 nm:       1,500 ft
  0 nm:       50 ft (threshold crossing height)

  Key: altitude decreases ~280 ft per nm on 3° glideslope
  Descent rate: 1,500-2,500 ft/min en route, 700-800 ft/min on approach

DEPARTURE (narrowbody jet):
  0 nm:       0 ft (runway)
  1 nm:       800 ft (airborne, initial climb)
  3 nm:       2,500 ft ("climb and maintain 5,000")
  5 nm:       5,000 ft
  10 nm:      10,000 ft ("contact departure, climb FL240")
  15 nm:      15,000 ft (accelerating above 250 KIAS)
  25 nm:      24,000 ft
  40 nm:      33,000 ft ("climb and maintain FL370")
  60 nm:      37,000 ft (cruise altitude)

  Key: climb rate 2,500-3,500 ft/min initially, decreasing above FL300

EN ROUTE (cruise, B777):
  Enters sector: FL350
  Step climb (as fuel burns): FL350 → FL370 → FL390
  Step climb interval: ~60-90 min between requests
  Speed: M0.84 (constant)
  Exits sector: FL370 or FL390

HOLDING STACK:
  Aircraft 1: 7,000 ft (lowest, first to be released)
  Aircraft 2: 8,000 ft
  Aircraft 3: 9,000 ft
  Aircraft 4: 10,000 ft
  Aircraft 5: 11,000 ft (highest, last released)

  As each is released: remaining aircraft descend 1,000 ft each
  Hold time: 4-6 min per circuit, 10-30 min typical total hold
```

**Speed trajectories for arrival:**
```
En route:    M0.78-0.85 (Mach number, high altitude)
Transition:  310 KIAS (~FL280, switching from Mach to KIAS)
Crossing 10,000: 250 KIAS (regulatory max)
Initial approach: 210 KIAS
Base/downwind: 180 KIAS
Final (10nm): 170 KIAS
Final (5nm):  160 KIAS
Short final:  Vref + 5-10 = 135-150 KIAS (aircraft dependent)
```

## 3.7 Operational Coherence Engine (ATC Correlation Rules)

This is the ATC equivalent of the ICU physiological correlation engine and the wildlife seasonal coherence engine. Every ATC state change must be operationally coherent — weather drives runway and approach selection, aircraft performance limits constrain what ATC can assign, flight phase constrains what clearances are valid, and separation rules constrain what multiple aircraft can do simultaneously.

### 3.7.1 Weather → Operations Cascade

Weather is the master driver. A single weather state determines what operations are possible. The state machine must pick a weather condition at initialization and enforce ALL downstream consequences.

```
WEATHER STATE TABLE:

VFR_CLEAR (ceiling >3000ft BKN/OVC, visibility >5 SM):
  - Approaches available: ALL (ILS, RNAV, VOR, visual, contact)
  - Visual approaches: YES (most common in good weather — faster throughput)
  - Runway throughput: maximum (30-36 landings/hr single runway)
  - Holding: none (no weather reason)
  - Go-arounds: rare (<1%)
  - Wind: variable, typically <15 kt
  - ATIS remark: "Visual approaches in use"

MVFR (ceiling 1000-3000ft, visibility 3-5 SM):
  - Approaches available: ILS, RNAV_LPV, RNAV_LNAV_VNAV, visual (if ceiling permits — pilot must have airport at MDA/DH)
  - Visual approaches: YES but less common (pilot must see airport)
  - Runway throughput: slightly reduced (28-34/hr)
  - Holding: possible for sequencing
  - Go-arounds: 2-5%
  - ATIS remark: "ILS approaches in use, visual approaches available"

IFR_LOW (ceiling 500-1000ft, visibility 1-3 SM):
  - Approaches available: ILS_CAT_I, RNAV_LPV only (others below minimums)
  - Visual approaches: NO (ceiling too low)
  - Runway throughput: reduced (24-30/hr — increased spacing, no visual separation)
  - Holding: likely (flow control, ground delays)
  - Go-arounds: 5-10%
  - Speed restrictions: more conservative (slower approach speeds for stability)
  - ATIS remark: "ILS approaches in use, ceiling 800 broken, visibility 2"

IFR_CAT_II (ceiling 200-500ft, visibility 1/2-1 SM, RVR 1200-2400ft):
  - Approaches available: ILS_CAT_II only (CAT I below minimums)
  - Visual approaches: NO
  - Runway throughput: severely reduced (20-26/hr)
  - Holding: very likely (many aircraft holding, ground stops at departing airports)
  - Go-arounds: 10-20% (missed approach at DH)
  - Crew qualification required: not all crews CAT II certified
  - ATIS remark: "ILS CAT II approaches in use, RVR runway 27L two thousand"

IFR_CAT_III (ceiling <200ft or zero, visibility <1/2 SM, RVR <1200ft):
  - Approaches available: ILS_CAT_IIIa or IIIb only
  - Visual approaches: NO
  - Runway throughput: minimal (15-22/hr — maximum spacing, single runway often)
  - Holding: extensive (stacks of 5-15 aircraft)
  - Go-arounds: 15-25%
  - Aircraft qualification: must be autoland capable — many aircraft diverted
  - Ground operations: low-visibility taxi procedures, follow-me vehicles
  - ATIS remark: "CAT III ILS approaches in use, RVR runway 28R six hundred"

THUNDERSTORM (CB in vicinity or on field):
  - Approaches available: depends on storm location; may be ILS with deviations
  - Visual approaches: NO (convective weather)
  - Runway throughput: drastically reduced or suspended
  - Holding: extensive (all arrivals holding until storm passes)
  - Go-arounds: 20-40% (windshear, microburst alerts)
  - Ground stop: possible (no departures)
  - Deviations: aircraft requesting deviations around cells, heading changes
  - ATIS remark: "Thunderstorms in vicinity, wind shear advisories in effect"
```

### 3.7.2 Weather ↔ Runway Configuration

Wind determines runway direction. Weather determines which approaches are available. The combination determines the full runway config.

```
RULE: WIND → RUNWAY DIRECTION
  - Aircraft land INTO the wind (headwind component maximized)
  - Wind 240-300° at ORD → west flow (runways 27L, 28C, 28R for landing)
  - Wind 060-120° at ORD → east flow (runways 10L, 10C, 10R for landing)
  - Wind shift >30° or speed change >5kt → possible runway change → new ATIS

RULE: WIND LIMITS
  - Maximum crosswind: 25-35 kt (aircraft-type dependent)
    - B737: 33 kt demonstrated crosswind
    - A320: 38 kt demonstrated crosswind
    - CRJ-200: 25 kt
    - Dash 8: 30 kt
  - Maximum tailwind: 10-15 kt (most airlines limit to 10 kt)
  - If crosswind exceeds limit for aircraft type → that aircraft cannot use that runway
    → may need to hold or divert

RULE: VISIBILITY → APPROACH TYPE AVAILABLE
  - Visibility >3 SM + ceiling >1000ft → visual approach permitted
  - RVR >2400ft → ILS CAT I permitted
  - RVR >1200ft → ILS CAT II permitted (if equipped)
  - RVR >700ft → ILS CAT IIIa permitted (if equipped)
  - RVR >150ft → ILS CAT IIIb permitted (if equipped)
  - If visibility drops mid-scenario: aircraft already on approach may continue
    (they're committed), but NEW clearances must use the upgraded approach type

RULE: RUNWAY CHANGE PROCEDURE
  When wind shifts enough to warrant runway change:
  1. New ATIS broadcast with new information letter
  2. Aircraft already on approach: continue current runway (no change mid-approach)
  3. Aircraft in hold or en route: re-sequenced for new runway
  4. Departures: switch immediately
  5. Transition period: 10-20 minutes of chaos as traffic re-sequences
  This is a MAJOR narrative event — affects all aircraft
```

### 3.7.3 Aircraft Performance ↔ ATC Constraints

ATC cannot assign values outside an aircraft's performance envelope. The state machine must check every assignment against aircraft type.

```
RULE: ALTITUDE LIMITS BY AIRCRAFT TYPE
  - B737/A320 family: max FL410
  - B777/B787/A350: max FL430-FL450
  - CRJ-200: max FL410
  - Dash 8-Q400: max FL270 (turboprop!)
  - ATR 72: max FL250
  - Cessna Citation: max FL450-FL510
  - Cessna 172 (GA): max 14,000 ft

  VIOLATION: Can't assign FL350 to a Dash 8 (ceiling is FL270)

RULE: SPEED LIMITS BY PHASE AND REGULATION
  - Below 10,000 ft MSL: 250 KIAS max (FAR 91.117a — ATC CANNOT waive)
  - Below 2,500 ft AGL within 4nm of Class C/D: 200 KIAS max
  - In holding: 200 KIAS below 6,000ft, 230 KIAS 6,001-14,000ft, 265 KIAS above 14,000ft
  - Minimum for jets: ~150 KIAS (below this → stall risk)
  - Minimum for turboprops: ~120 KIAS

  VIOLATION: Can't assign 280 knots to aircraft at 8,000 ft (above 250 limit)
  VIOLATION: Can't assign 130 knots to B737 (below minimum safe speed en route)

RULE: CLIMB/DESCENT RATE LIMITS
  - Narrowbody jet initial climb: 2,500-3,500 ft/min
  - Narrowbody at FL350: 500-1,500 ft/min (thinning air)
  - Widebody heavy: initial 2,000-3,000 ft/min
  - Turboprop: 1,200-2,200 ft/min
  - Descent rate: 1,500-3,000 ft/min typical; >3,500 uncommon
  - On ILS glideslope: ~700-800 ft/min (fixed by 3° angle)

  VIOLATION: Can't have B737 climbing 3,000 ft/min through FL380 (would be ~500-800)

RULE: APPROACH SPEED BY AIRCRAFT TYPE (Vref range)
  - B737-800: 135-142 KIAS
  - A320: 131-137 KIAS
  - B777-200: 140-150 KIAS
  - B787-9: 140-148 KIAS
  - CRJ-900: 132-138 KIAS
  - E175: 126-132 KIAS
  - Dash 8-Q400: 110-120 KIAS
  - A380: 145-155 KIAS

  On final approach, speed MUST be Vref + 5-10 knots (+ gust factor)
  Can't have B737 at 200 knots on 2-mile final (way too fast)
```

### 3.7.4 Flight Phase ↔ Clearance Sequence Rules

Clearances must follow a strict order. You can't skip steps.

```
DEPARTURE SEQUENCE (must follow this order):
  1. clearance_received_at_gate (IFR clearance from clearance delivery)
  2. pushback_approved (from ramp/ground)
  3. taxi_to_runway (from ground control, with specific route)
  4. holding_short_of_runway (ground hands off to tower)
  5. line_up_and_wait (tower, if traffic on runway or preceding departure not yet airborne)
  6. cleared_for_takeoff (tower)
  7. airborne_initial_climb (tower observes)
  8. contact_departure (tower hands off to TRACON departure)
  9. climbing_via_SID (departure control)
  10. en_route_level_flight (departure hands off to center)

  VIOLATION: Can't get "cleared for takeoff" before "taxi to runway"
  VIOLATION: Can't "contact departure" while still on the ground
  VIOLATION: Can't be "climbing via SID" before being airborne

ARRIVAL SEQUENCE (must follow this order):
  1. en_route_level_flight (center)
  2. descending_via_STAR (center, "descend via STAR")
  3. radar_vectors_for_sequencing (approach control, after center handoff)
  4. cleared_approach (approach, specific type: ILS/RNAV/visual)
  5. on_final (established on approach course)
  6. cleared_to_land (tower, when on short final ~3nm)
  7. landing_roll_and_exit (tower observes)
  8. taxi_to_gate (ground control)

  Can insert HOLD at any point between steps 1-4:
  1a. holding_at_fix (if delays exist)
  1b. released_from_hold (when sequenced)

  Can insert GO_AROUND after step 5 or 6:
  5a/6a. go_around_climbing (tower issues go-around)
  → re-enters at step 3 (radar vectors for re-sequencing)

  VIOLATION: Can't "clear to land" before "cleared approach"
  VIOLATION: Can't be "on_final" at FL250 (way too high)
  VIOLATION: Can't "descend via STAR" after already cleared for approach
```

### 3.7.5 Separation Rules (Multi-Aircraft Constraints)

These constrain what MULTIPLE aircraft can do simultaneously.

```
RULE: VERTICAL SEPARATION
  - Any two aircraft at the same position: minimum 1,000 ft vertical separation below FL410
  - Above FL410: 2,000 ft minimum
  - During climb/descent transitions: separation applies to ASSIGNED altitudes,
    not necessarily current altitudes (since one is climbing through)
  - When assigning altitude to aircraft A, check that no other aircraft
    is already assigned within 1,000 ft at the same lateral position

RULE: HORIZONTAL (RADAR) SEPARATION — RECAT-based
  - TRACON: 3 nm minimum between any two aircraft (radar minimum)
  - En route (ARTCC): 5 nm minimum
  - On approach to same runway: use RECAT_SEPARATION matrix (Section 2.2)
    - Lookup: RECAT_SEPARATION[leading_aircraft_cat][trailing_aircraft_cat]
    - B757 (RECAT D) exception: when B757 is LEADING, use row C separation
    - Minimum is always >= 3.0 nm (radar floor)

  When sequencing arrivals, spacing must account for:
  - Aircraft speed differences (fast aircraft catches slow aircraft ahead)
  - Wind effect on groundspeed
  - Wake turbulence category of LEADING aircraft (RECAT A-F)

RULE: DEPARTURE SEPARATION (TIME-BASED, SAME RUNWAY) — RECAT-based
  - Use RECAT_DEPARTURE_SEPARATION matrix (Section 2.2)
    - Lookup: RECAT_DEPARTURE_SEPARATION[leading_cat][trailing_cat] → seconds
  - B757 (RECAT D) exception: when B757 is LEADING, use row C separation
  - Minimum is always >= 60 seconds

  VIOLATION: Can't clear Southwest (B737) for takeoff 90 seconds after
  an Emirates A380 departed the same runway

RULE: PARALLEL RUNWAY OPERATIONS
  - Runways >4,300 ft apart: independent simultaneous approaches
    (no coordination needed between the two approach sequences)
  - Runways 2,500-4,300 ft apart: dependent approaches
    (1.5 nm stagger required — aircraft on parallel courses must be offset)
  - Runways <2,500 ft apart: PRM/SOIA procedures required
    (special training, monitor controller, breakout procedures)

  The airport config in the scenario determines which parallel operations are valid.
```

### 3.7.6 Cross-Attribute Correlation Rules (Must Enforce)

```
RULE 1: Weather ↔ Approach Type ↔ Go-Around Rate
  - Clear weather → visual approaches → go-around rate <1%
  - IFR low → ILS CAT I only → go-around rate 5-10%
  - CAT III conditions → autoland required → go-around rate 15-25%
  - Thunderstorms → holding or diversions → go-around rate 20-40%
  - If weather degrades mid-scenario: go-around rate MUST increase for subsequent aircraft

RULE 2: Weather ↔ Holding ↔ Fuel ↔ Emergency Risk
  - Holding burns 20-30% more fuel than cruise
  - Each 30-minute hold reduces fuel endurance by ~45 min equivalent
  - After 60+ min holding: some aircraft may declare minimum fuel
  - After 90+ min: fuel emergencies likely for short-haul aircraft (CRJ, E175)
  - If weather causes holding: fuel_remaining MUST decrease for all holding aircraft
  - Can't have aircraft hold for 2 hours and still have 4 hours fuel (if they started with 5)

RULE 3: Altitude ↔ Speed ↔ Flight Phase
  - At FL350+: speed reported in Mach (M0.78-M0.85), NOT knots
  - At FL280 (transition): switching from Mach to KIAS (280-310 KIAS)
  - Crossing 10,000 ft descending: MUST decelerate to 250 KIAS before or at 10,000
  - On final approach: speed MUST be decelerating progressively toward Vref
  - In hold: speed MUST comply with hold speed limits by altitude

  VIOLATION: Aircraft at FL370 reporting "250 knots" (should be Mach ~0.80)
  VIOLATION: Aircraft at 8,000 ft at 280 knots (exceeds 250 below 10,000)
  VIOLATION: Aircraft on 3-mile final at 220 knots (way too fast, should be ~140-150)

RULE 4: Flight Phase ↔ Altitude ↔ Distance
  Arrivals must be at appropriate altitude for their distance from airport:
  - 100+ nm: FL250-FL370 (cruise or initial descent)
  - 60-100 nm: FL180-FL280 (STAR descent)
  - 30-60 nm: 10,000-FL180 (terminal area)
  - 15-30 nm: 4,000-10,000 ft (vectored for approach)
  - 5-15 nm: 1,500-4,000 ft (on approach course)
  - 0-5 nm: 0-1,500 ft (short final)

  3:1 rule: need 3 nm per 1,000 ft of altitude to lose
  Aircraft at 30,000 ft needs ~90 nm to descend (30 × 3 = 90)

  VIOLATION: Aircraft 10 nm from airport at FL280 (can't descend 28,000 ft in 10 nm)
  VIOLATION: Aircraft 80 nm out at 3,000 ft (far too low for that distance)

RULE 5: Wind ↔ Runway ↔ Approach Direction
  - Landing runway must face INTO the prevailing wind
  - Wind from 270° → use runway 27 (heading 270), NOT runway 9 (would be tailwind)
  - Crosswind component = wind_speed × sin(wind_direction - runway_heading)
  - Headwind component = wind_speed × cos(wind_direction - runway_heading)
  - If headwind is negative (tailwind) and >10-15 kt: can't use that runway

  VIOLATION: Wind 090/25 and landing runway 27L (25-knot tailwind)

RULE 6: Time ↔ Fuel ↔ Endurance
  - Fuel decreases linearly with time (cruise burn rate × time)
  - Holding increases burn rate by 20-30%
  - Fuel remaining at any point = initial_fuel - (time_elapsed × burn_rate)
  - If fuel_remaining < 45 min reserve: pilot declares "minimum fuel"
  - If fuel_remaining < 30 min: pilot declares fuel emergency

  Burn rates (approximate):
  - B737: 4,900 lbs/hr cruise, ~6,000 lbs/hr holding
  - A320: 4,800 lbs/hr cruise, ~5,800 lbs/hr holding
  - B777: 14,000 lbs/hr cruise, ~17,000 lbs/hr holding
  - CRJ-900: 3,500 lbs/hr cruise, ~4,200 lbs/hr holding
  - E175: 3,400 lbs/hr cruise, ~4,000 lbs/hr holding

RULE 7: Turbulence ↔ Altitude ↔ Ride Reports ↔ Rerouting
  - If turbulence reported at FL350: nearby aircraft may request FL370 or FL330
  - Moderate turbulence: speed reduce to turbulence penetration speed (~280 KIAS for jets)
  - Severe turbulence: pilot requests deviation, ATC clears "deviate as necessary"
  - Clear air turbulence (CAT): typically near jet stream (FL300-FL400), winter
  - Convective turbulence: near thunderstorms, any altitude
  - Pilot reports (PIREPs) propagate: once one pilot reports, controller advises others

RULE 8: Emergency ↔ All Other Aircraft
  - Emergency declared → that aircraft gets PRIORITY over all others
  - Other aircraft: may be held, given extended vectors, speed reduced, or altitude changed
  - Emergency aircraft: shortest path to runway, fire trucks standby
  - Can't have "normal sequencing" while an emergency aircraft is inbound
  - Emergency disrupts the entire flow — other aircraft's delays INCREASE

RULE 9: ATIS Letter ↔ Weather ↔ Time Progression
  - ATIS updates sequentially: Alpha → Bravo → Charlie → ... → Zulu → Alpha
  - Each update triggered by: weather change, runway change, or hourly
  - All aircraft must acknowledge current ATIS on initial contact
  - If ATIS changes mid-scenario: aircraft already on frequency may not have new info
  - Controller: "ATIS now information Golf, winds now 280 at 18 gusting 25"

RULE 10: Night Operations ↔ Traffic Mix ↔ Volume
  - 22:00-06:00 local: primarily cargo (FedEx, UPS) and red-eyes
  - Much lower traffic volume (5-10 aircraft vs 30-50 during day)
  - Noise abatement procedures may restrict certain runways
  - No student pilots (night training restrictions)
  - Can't have SWA2241 departing at 03:00 (Southwest doesn't fly red-eyes)
  - FedEx/UPS hub airports (MEM, SDF, ONT): BUSIER at night than during day
```

### 3.7.7 Scenario Progression Archetypes

Each scenario archetype determines how the operational state evolves over time:

```python
SCENARIO_PROGRESSIONS = {
    "routine_arrival_rush": {
        # Weather: stable VFR/MVFR throughout
        # Traffic: steady stream, 30-35/hr landing rate
        # Progression: aircraft arrive, descend, sequence, land — no drama
        "weather_changes": 0,
        "runway_changes": 0,
        "holding": False,
        "emergency": False,
    },
    "weather_deterioration": {
        # Weather: starts MVFR, degrades to IFR_LOW or CAT_II
        # Progression: first few aircraft get visual approaches (easy)
        #   → weather drops → switch to ILS only → spacing increases → delays build
        #   → possible holding for later aircraft → possible diversions
        "weather_changes": 2-3,  # ATIS updates
        "approach_type_changes": True,  # visual → ILS CAT I → ILS CAT II
        "holding": True,  # develops as weather worsens
        "go_around_rate_increasing": True,
    },
    "runway_change_mid_session": {
        # Weather: wind shift occurs mid-scenario
        # Progression: first half on one runway config → wind shift → new ATIS
        #   → transition chaos (10-20 min) → re-sequenced on new runways
        # Aircraft already on approach: continue current runway
        # Aircraft in hold/en route: re-sequenced
        "weather_changes": 1,  # the wind shift
        "runway_changes": 1,  # the big event
        "transition_duration_min": 15,
    },
    "holding_stack_scenario": {
        # Weather: IFR throughout, below approach minimums initially
        # Progression: aircraft stack up in holds at fix → weather improves slowly
        #   → release aircraft one by one from lowest altitude first
        #   → spacing tight as stack clears → fuel monitoring critical
        "holding": True,
        "stack_depth": 4-8,  # aircraft in hold
        "weather_improves": True,  # eventually clears enough for approaches
        "fuel_emergencies_possible": True,
    },
    "emergency_disruption": {
        # Weather: any
        # Progression: normal ops → one aircraft declares emergency
        #   → emergency gets priority → other aircraft held/rerouted/delayed
        #   → emergency lands → normal ops resume but sequence disrupted
        "emergency": True,
        "emergency_timing": "mid_scenario",  # not at start
        "disruption_duration_min": 15-30,
    },
}
```

## 4. Narrative Generation

### 4.1 Story Structure

ATC narratives have a fundamentally different structure from wildlife or ICU — they can be **transcripts** or **logs**.

```
[SCENARIO HEADER]
  - Facility, position, time, weather (ATIS), runway config

[ATC SEQUENCE]
  - Continuous stream of controller-pilot exchanges
  - Each exchange: controller instruction → pilot readback
  - Interspersed with radar observations and coordination notes

[Optional: SHIFT CHANGE / SUMMARY]
```

For arrival rush: narrative follows multiple aircraft converging on the airport, interleaving descent clearances, speed reductions, vectors, and approach clearances.

For en route: narrative follows aircraft crossing a sector, with handoffs, altitude changes, and traffic advisories.

### 4.2 Narrative Voice — 50/50

**ATC transcript voice** (verbatim controller-pilot exchanges, present tense, radio phraseology):
> "14:32:15 — 'United 1523, descend and maintain flight level two four zero.'
> 'Descend and maintain flight level two four zero, United 1523.'
> 14:32:40 — 'Delta 445, reduce speed two five zero knots.'
> 'Two five zero knots, Delta 445.'
> 14:33:05 — 'United 1523, traffic twelve o'clock, eight miles, opposite direction, heavy triple seven, flight level two six zero.'
> 'Looking for traffic, United 1523.'"

**Sector log voice** (narrative summary, past tense, controller's perspective):
> "At 14:32, United 1523 was cleared to descend from flight level two eight zero to flight level two four zero. The Boeing 737-800 was 45 miles northwest on the STAR, number four in the arrival sequence. Delta 445, trailing by 8 miles at flight level two six zero, was given a speed reduction to 250 knots for spacing. Separation between the two was 8 miles lateral, 2,000 feet vertical — well above minimums."

Both convey the same information (altitude assignments, speed, position) but with very different density and style.

### 4.3 Narrative Templates

```python
# ── SCENARIO HEADER TEMPLATES ──

HEADER_TRANSCRIPT = [
    "{facility_name} {position}, {date}, {time_utc} Zulu. "
    "ATIS information {atis_letter}: wind {wind_dir} at {wind_speed}, "
    "visibility {visibility}, ceiling {ceiling}. "
    "Landing runway {landing_rwy}, departing runway {departing_rwy}. "
    "Altimeter {altimeter}.",

    "{facility_name}, {position} sector. {date}. Current ATIS {atis_letter}. "
    "Winds {wind_dir}/{wind_speed}, vis {visibility} SM, {sky_condition}. "
    "Active runways: {rwy_config}. {traffic_note}.",
]

HEADER_LOG = [
    "Sector log, {facility_name} {position}. {date}, {start_time}-{end_time} UTC. "
    "Weather: {weather_summary}. Runway configuration: {rwy_config}. "
    "{num_aircraft} aircraft worked during the period. {volume_note}.",

    "{date} — {facility_name} {position} position. ATIS {atis_letter} current. "
    "{weather_summary}. {rwy_config}. Traffic volume: {volume_description}.",
]

# ── ALTITUDE ASSIGNMENT TEMPLATES ──

ALT_TRANSCRIPT = [
    "{time} — '{callsign}, descend and maintain {altitude}.'\n"
    "'{descend_readback}, {callsign}.'",

    "{time} — '{callsign}, climb and maintain {altitude}.'\n"
    "'{climb_readback}, {callsign}.'",

    "{time} — '{callsign}, cross {fix} at {altitude}.'\n"
    "'Cross {fix} at {alt_readback}, {callsign}.'",

    "{time} — '{callsign}, descend via the {star_name} arrival.'\n"
    "'Descend via {star_name}, {callsign}.'",
]

ALT_LOG = [
    "At {time}, {callsign} ({aircraft_type}) was cleared to descend from "
    "{old_altitude} to {altitude}. {context}.",

    "{callsign} received a climb clearance to {altitude} at {time}, "
    "passing through {current_altitude} at {climb_rate} feet per minute. "
    "{separation_note}.",

    "The {aircraft_type} ({callsign}) was issued a descent to {altitude} at {time}. "
    "{callsign} was {distance_nm} miles {direction} of {fix}, "
    "number {sequence_number} for runway {runway}.",
]

# ── SPEED ASSIGNMENT TEMPLATES ──

SPEED_TRANSCRIPT = [
    "{time} — '{callsign}, reduce speed to {speed} knots.'\n"
    "'{speed_readback}, {callsign}.'",

    "{time} — '{callsign}, maintain {speed} knots to {fix}.'\n"
    "'Maintain {speed_readback} to {fix}, {callsign}.'",

    "{time} — '{callsign}, say airspeed.'\n"
    "'{callsign}, indicating {speed} knots.'",
]

SPEED_LOG = [
    "{callsign} was given a speed reduction to {speed} knots at {time} "
    "for spacing behind {preceding_callsign}. {gap_note}.",

    "At {time}, {callsign} reported {speed} knots indicated, "
    "{distance_nm} miles from the field at {altitude}.",
]

# ── HEADING ASSIGNMENT TEMPLATES ──

HEADING_TRANSCRIPT = [
    "{time} — '{callsign}, fly heading {heading}.'\n"
    "'Heading {heading_readback}, {callsign}.'",

    "{time} — '{callsign}, turn {direction} heading {heading}, "
    "vectors {approach_type} runway {runway}.'\n"
    "'{direction_readback} heading {heading_readback}, vectors {approach_type} "
    "{runway_readback}, {callsign}.'",
]

HEADING_LOG = [
    "{callsign} was vectored to heading {heading} at {time} for "
    "{vector_reason}. {position_note}.",
]

# ── APPROACH CLEARANCE TEMPLATES ──

APPROACH_TRANSCRIPT = [
    "{time} — '{callsign}, {distance} miles from {fix}, "
    "cleared {approach_type} runway {runway} approach.'\n"
    "'Cleared {approach_type} runway {runway}, {callsign}.'",

    "{time} — '{callsign}, cleared visual approach runway {runway}, "
    "traffic to follow is {preceding_type} on a {preceding_distance} mile final.'\n"
    "'Cleared visual {runway}, traffic in sight, {callsign}.'",
]

APPROACH_LOG = [
    "At {time}, {callsign} was cleared for the {approach_type} approach to "
    "runway {runway}, {distance} miles from the threshold at {altitude}. "
    "{sequence_note}.",
]

# ── LANDING / TAKEOFF CLEARANCE TEMPLATES ──

LANDING_TRANSCRIPT = [
    "{time} — '{callsign}, runway {runway}, cleared to land, "
    "wind {wind_dir} at {wind_speed}.'\n"
    "'Cleared to land, {runway}, {callsign}.'",
]

TAKEOFF_TRANSCRIPT = [
    "{time} — '{callsign}, runway {runway}, cleared for takeoff, "
    "wind {wind_dir} at {wind_speed}.'\n"
    "'Cleared for takeoff, {runway}, {callsign}.'",

    "{time} — '{callsign}, runway {runway}, line up and wait.'\n"
    "'Line up and wait, {runway}, {callsign}.'",
]

# ── TRAFFIC ADVISORY TEMPLATES ──

TRAFFIC_TRANSCRIPT = [
    "{time} — '{callsign}, traffic {clock_position} o'clock, "
    "{distance} miles, {direction}, {traffic_type}, {traffic_altitude}.'\n"
    "'{traffic_response}, {callsign}.'",
]

TRAFFIC_LOG = [
    "Traffic advisory issued to {callsign} at {time}: "
    "{traffic_callsign} ({traffic_type}) at {traffic_altitude}, "
    "{distance} miles {clock_position} o'clock. {separation_note}.",
]

# ── HOLD INSTRUCTION TEMPLATES ──

HOLD_TRANSCRIPT = [
    "{time} — '{callsign}, hold {direction} of {fix} on the "
    "{radial} radial, {turn_direction} turns, "
    "expect further clearance {efc_time} zulu, maintain {altitude}.'\n"
    "'Hold {direction_readback} of {fix}, {radial} radial, "
    "{turn_direction_readback} turns, expect further clearance {efc_time}, "
    "maintain {altitude_readback}, {callsign}.'",
]

HOLD_LOG = [
    "{callsign} was placed in a hold at {fix} at {time}, "
    "altitude {altitude}. Expected release: {efc_time}Z. "
    "Reason: {hold_reason}.",
]

# ── GO-AROUND TEMPLATES ──

GOAROUND_TRANSCRIPT = [
    "{time} — '{callsign}, go around, fly heading {heading}, "
    "climb and maintain {altitude}.'\n"
    "'Going around, heading {heading_readback}, climb {altitude_readback}, "
    "{callsign}.'",
]

GOAROUND_LOG = [
    "{callsign} executed a go-around at {time} due to {reason}. "
    "Assigned heading {heading}, climb to {altitude}. "
    "Re-sequenced as number {new_sequence} for runway {runway}.",
]

# ── EMERGENCY TEMPLATES ──

EMERGENCY_TRANSCRIPT = [
    "{time} — '{callsign}, declaring emergency. {emergency_type}. "
    "Request {request}. {fuel_status}. {souls}.'\n"
    "'{callsign}, roger, squawk seven seven zero zero. "
    "Fly heading {heading}, vectors runway {runway}. "
    "{emergency_response}.'",
]

# ── WEATHER UPDATE TEMPLATES ──

WEATHER_TRANSCRIPT = [
    "{time} — 'All aircraft, ATIS now information {new_letter}. "
    "{weather_change}. Landing runway now {new_runway}.'",
]

# ── FREQUENCY CHANGE TEMPLATES ──

FREQ_TRANSCRIPT = [
    "{time} — '{callsign}, contact {facility} on {frequency}.'\n"
    "'{facility_readback} {frequency_readback}, {callsign}.'",
]
```

### 4.4 Airport / Facility Pools

```python
AIRPORTS = {
    "ORD": {
        "name": "Chicago O'Hare International",
        "icao": "KORD",
        "runways": ["10L/28R", "10C/28C", "10R/28L", "9L/27R", "9R/27L", "4R/22L", "4L/22R", "14R/32L", "14L/32R", "15/33"],
        "typical_configs": {
            "west_flow": {"landing": ["27L", "28C", "28R"], "departing": ["28R", "27L", "32L"]},
            "east_flow": {"landing": ["10L", "10C", "10R"], "departing": ["9R", "10C", "22L"]},
        },
        "tracon": "C90 (Chicago TRACON)",
        "center": "ZAU (Chicago Center)",
        "elevation": 672,
        "stars": ["BENKY", "ENDEE", "GIPPER", "KRENA", "PANGG", "ROYKO"],
        "sids": ["DUPAGE", "EARLS", "OHARE", "PETTY", "ACITO", "MOBLE"],
    },
    "ATL": {
        "name": "Hartsfield-Jackson Atlanta International",
        "icao": "KATL",
        "runways": ["8L/26R", "8R/26L", "9L/27R", "9R/27L", "10/28"],
        "typical_configs": {
            "west_flow": {"landing": ["26L", "27L", "28"], "departing": ["26R", "27R"]},
            "east_flow": {"landing": ["8R", "9R", "10"], "departing": ["8L", "9L"]},
        },
        "tracon": "A80 (Atlanta TRACON)",
        "center": "ZTL (Atlanta Center)",
        "elevation": 1026,
        "stars": ["PECHY", "ERLIN", "FLCON", "HONIE", "RPTOR", "SWTHR", "CANUK", "BUKHD"],
        "sids": ["DAWGS", "FNTNA", "KAJIN", "JOTVL", "ROTFL"],
    },
    "LAX": {
        "name": "Los Angeles International",
        "icao": "KLAX",
        "runways": ["6L/24R", "6R/24L", "7L/25R", "7R/25L"],
        "typical_configs": {
            "westbound": {"landing": ["24R", "25L"], "departing": ["24L", "25R"]},
        },
        "tracon": "SCT (SoCal TRACON)",
        "center": "ZLA (Los Angeles Center)",
        "elevation": 126,
        "stars": ["SADDE", "IRNMN", "RIKKN", "SEAVU", "BRUEN", "HUULL"],
        "sids": ["ORCKA", "DOTSS", "LAXXR", "HOLBK"],
    },
    "JFK": {
        "name": "John F. Kennedy International",
        "icao": "KJFK",
        "runways": ["4L/22R", "4R/22L", "13L/31R", "13R/31L"],
        "typical_configs": {
            "south_flow": {"landing": ["13L", "13R"], "departing": ["13L", "31L"]},
            "north_flow": {"landing": ["31L", "31R"], "departing": ["31L", "22R"]},
        },
        "tracon": "N90 (New York TRACON)",
        "center": "ZNY (New York Center)",
        "elevation": 13,
        "stars": ["DEEZZ", "GREKI", "LENDY", "ROBER", "CAMRN", "PARCH"],
        "sids": ["SKORR", "GREKI", "DEEZZ", "BETTE", "HAPIE"],
    },
    "DFW": {
        "name": "Dallas/Fort Worth International",
        "icao": "KDFW",
        "runways": ["13L/31R", "13R/31L", "17C/35C", "17L/35R", "17R/35L", "18L/36R", "18R/36L"],
        "typical_configs": {
            "south_flow": {"landing": ["18R", "17C", "17R", "13R"], "departing": ["18L", "17L", "36R"]},
            "north_flow": {"landing": ["35L", "36R", "35C", "31R"], "departing": ["35R", "36L"]},
        },
        "tracon": "D10 (Dallas TRACON)",
        "center": "ZFW (Fort Worth Center)",
        "elevation": 607,
        "stars": ["BRUSR", "FEVER", "GRABE", "HOWDY", "PODDE", "SEEVR", "TRISS"],
        "sids": ["AKUNA", "DALFT", "LOWGN", "TEXSS"],
    },
    "SFO": {
        "name": "San Francisco International",
        "icao": "KSFO",
        "runways": ["1L/19R", "1R/19L", "10L/28R", "10R/28L"],
        "typical_configs": {
            "west_flow": {"landing": ["28L", "28R"], "departing": ["1R", "1L"]},
        },
        "tracon": "NCT (NorCal TRACON)",
        "center": "ZOA (Oakland Center)",
        "elevation": 13,
        "stars": ["BDEGA", "DYAMD", "MODST", "SNTAA", "WESLA", "FOGGG"],
        "sids": ["SSTIK", "TRUKN", "SAHEY", "OFFSH"],
    },
    "DEN": {
        "name": "Denver International",
        "icao": "KDEN",
        "runways": ["7/25", "8/26", "16L/34R", "16R/34L", "17L/35R", "17R/35L"],
        "typical_configs": {
            "south_flow": {"landing": ["34L", "35L", "35R"], "departing": ["34R", "35R"]},
        },
        "tracon": "D01 (Denver TRACON)",
        "center": "ZDV (Denver Center)",
        "elevation": 5431,
        "stars": ["TSHNR", "FLATI", "ROKKY", "CONNR", "SUNST", "BLZZD"],
        "sids": ["PLAINS", "PIKES", "DNVR", "ELBRR"],
    },
    "generic_class_c": {
        "name": "Metro Regional Airport",
        "icao": "KXYZ",
        "runways": ["9/27", "18/36"],
        "typical_configs": {
            "west_flow": {"landing": ["27"], "departing": ["27"]},
            "east_flow": {"landing": ["9"], "departing": ["9"]},
        },
        "tracon": "Regional Approach",
        "center": "Regional Center",
        "elevation": 800,
    },
}
```

### 4.5 Filler / Context Sentences

```python
FILLER_TRANSCRIPT = [
    "'{callsign}, say altitude.' '{callsign}, level {altitude}.'",
    "'{callsign}, ident.' '{callsign}, radar contact, {distance} miles {direction} of {fix}.'",
    "'{callsign}, say flight conditions.' '{callsign}, smooth ride, {weather_note}.'",
    "'{callsign}, expect {delay_min} minute delay for sequencing.'",
    "Coordination with {adjacent_sector}: {coordination_note}.",
    "APREQ (approval request) for {callsign} through {altitude}: approved.",
    "'{callsign}, report {fix} in sight.' 'Looking, {callsign}.'",
]

FILLER_LOG = [
    "Coordination with {adjacent_facility} confirmed {callsign}'s handoff conditions.",
    "ATIS updated to information {letter} at {time}: {change_note}.",
    "Pilot of {callsign} reported {ride_report} at {altitude}.",
    "APREQ for {callsign} through {sector}: approved at {altitude}.",
    "Runway {runway} checked clear by tower at {time}.",
    "Wind check at {time}: {wind_dir} at {wind_speed}, gusting {gust}.",
    "Spacing between {callsign_1} and {callsign_2}: {distance} nm, {separation_note}.",
]
```

## 5. Controlling num_keys and num_updates

### num_keys (number of aircraft)

- num_keys=2: paired conflict scenario (two aircraft converging at same altitude)
- num_keys=3-4: small arrival sequence or departure group
- num_keys=5-8: typical arrival rush at moderate airport
- num_keys=10-12: busy TRACON period at major hub
- num_keys=15-20: full sector during peak hour (en route or major hub)

### num_updates (state changes per aircraft)

| num_updates | Updates per aircraft | ~Word count | Scenario duration |
|-------------|---------------------|-------------|-------------------|
| 3 | 3 | 300-500 | ~5-10 minutes (short sequence) |
| 5 | 5 | 500-900 | ~10-20 minutes (arrival from STAR to final) |
| 7 | 7 | 800-1300 | ~15-30 minutes (full approach from en route) |
| 10 | 10 | 1200-2000 | ~30-45 minutes (complete sector transit) |
| 15 | 15 | 2000-3500 | ~45-60 minutes (full rush period) |
| 20+ | 20+ | 3000-5000 | ~60+ minutes (extended busy period) |

### Interleaving Strategy

ATC naturally interleaves because the controller is managing multiple aircraft simultaneously:

```
14:32:15 — United 1523, descend FL240 (UAL1523 altitude: FL280 → FL240)
14:32:40 — Delta 445, reduce speed 250 knots (DAL445 speed: 310 → 250)
14:33:05 — American 2118, fly heading 270 (AAL2118 heading: 310 → 270)
14:33:30 — United 1523, traffic advisory (no state change, filler)
14:34:00 — Southwest 2241, descend FL180 (SWA2241 altitude: FL240 → FL180)
14:34:25 — Delta 445, descend FL200 (DAL445 altitude: FL260 → FL200)
14:35:00 — United 1523, reduce speed 210 knots (UAL1523 speed: 250 → 210)
```

Constraint: same aircraft should not receive two consecutive instructions (controller works the sequence round-robin). Exception: emergency — that aircraft dominates.

## 6. Distinguishing Setups

### Variation dimensions:
1. **Aircraft mix** — all narrowbodies, mixed with widebodies, heavies present, GA mixed in
2. **Airport** — ORD, ATL, LAX, JFK, DFW, SFO, DEN, generic
3. **Scenario type** — arrival rush, departure push, mixed ops, en route, holding
4. **Weather** — clear VFR, low IFR, marginal VFR (most interesting for traffic flow)
5. **Traffic volume** — light (5 aircraft), moderate (10), heavy (15+)
6. **Queried aircraft** — which aircraft the RI/PI question asks about
7. **Queried attribute** — altitude vs speed vs heading vs position
8. **Time of day** — morning push, afternoon, red-eye, cargo late night

### Scenario archetypes:

```python
SCENARIO_ARCHETYPES = [
    "routine_arrival_rush",     # standard busy arrival period, good weather
    "weather_hold",             # weather below minimums, aircraft holding
    "runway_change",            # wind shift mid-session, reconfigure runways
    "departure_ground_stop",    # GDP in effect, metered departures
    "emergency_in_sequence",    # one aircraft declares emergency, others displaced
    "vfr_to_ifr_transition",    # weather deteriorating, approaches upgrade
    "heavy_mix",                # A380/B747s mixed in, wake spacing opens gaps
    "parallel_approaches",      # simultaneous approaches to parallel runways
    "crossing_traffic",         # en route, multiple aircraft crossing at similar altitudes
    "overnight_cargo",          # light traffic, heavy aircraft (FedEx/UPS), different feel
]
```

## 7. Name Pools

### Callsign Generation

```python
AIRLINES_AND_CALLSIGNS = {
    # Major US — high frequency
    "AAL": {"radio": "American", "iata": "AA", "fleet": ["A319", "A320", "A321", "B738", "B752", "B789", "B77W"]},
    "DAL": {"radio": "Delta", "iata": "DL", "fleet": ["A320", "A321", "B738", "B739", "A339", "B763", "B764"]},
    "UAL": {"radio": "United", "iata": "UA", "fleet": ["A319", "A320", "B738", "B39M", "B752", "B789", "B77W", "B772"]},
    "SWA": {"radio": "Southwest", "iata": "WN", "fleet": ["B737", "B738", "B38M"]},

    # Major US — medium frequency
    "JBU": {"radio": "JetBlue", "iata": "B6", "fleet": ["A320", "A321", "A20N", "A21N"]},
    "ASA": {"radio": "Alaska", "iata": "AS", "fleet": ["B738", "B739", "B39M", "E75L"]},
    "NKS": {"radio": "Spirit Wings", "iata": "NK", "fleet": ["A320", "A321", "A20N"]},
    "FFT": {"radio": "Frontier", "iata": "F9", "fleet": ["A320", "A321", "A20N"]},

    # Regional — high frequency
    "SKW": {"radio": "SkyWest", "iata": "OO", "fleet": ["CRJ7", "CRJ9", "E75L"]},
    "RPA": {"radio": "Brickyard", "iata": "YX", "fleet": ["E170", "E75L"]},
    "EDV": {"radio": "Endeavor", "iata": "9E", "fleet": ["CRJ9", "CRJ7"]},
    "ENY": {"radio": "Envoy", "iata": "MQ", "fleet": ["E170", "E75L", "CRJ7"]},

    # Cargo
    "FDX": {"radio": "FedEx", "iata": "FX", "fleet": ["B763", "B772", "B77F", "A306", "MD11"]},
    "UPS": {"radio": "UPS", "iata": "5X", "fleet": ["B748", "B763", "B753", "B772", "MD11", "A306"]},

    # International — lower frequency
    "BAW": {"radio": "Speedbird", "iata": "BA", "fleet": ["B772", "B789", "A388"]},
    "DLH": {"radio": "Lufthansa", "iata": "LH", "fleet": ["A346", "B748", "A359"]},
    "UAE": {"radio": "Emirates", "iata": "EK", "fleet": ["A388", "B77W"]},
    "ACA": {"radio": "Air Canada", "iata": "AC", "fleet": ["B789", "B77W", "A333", "B738", "A320"]},
    "AFR": {"radio": "Air France", "iata": "AF", "fleet": ["B772", "A359", "A388"]},
    "QTR": {"radio": "Qatari", "iata": "QR", "fleet": ["B77W", "A359", "A35K"]},
    "SIA": {"radio": "Singapore", "iata": "SQ", "fleet": ["B77W", "A359", "A35K", "A388"]},
    "KAL": {"radio": "Korean Air", "iata": "KE", "fleet": ["B77W", "B748", "A333"]},
}

# Flight numbers: 3-4 digits, realistic ranges
# AAL: 1-2999, DAL: 1-2999, UAL: 1-2999, SWA: 1-9999 (4 digits common)
# International: 1-999 typically
# Cargo: FDX 1-999, UPS 1-999

def generate_callsign(airline_code, rng):
    if airline_code in ("SWA",):
        number = rng.randint(100, 9999)
    elif airline_code in ("FDX", "UPS"):
        number = rng.randint(100, 999)
    elif airline_code in ("BAW", "DLH", "UAE", "ACA", "AFR", "QTR", "SIA", "KAL"):
        number = rng.randint(1, 999)
    else:
        number = rng.randint(100, 2999)
    return f"{airline_code}{number}"
```

### Waypoint / Fix Names

```python
# Real-world-inspired but generic to avoid confusion with actual airspace
WAYPOINTS = [
    "MERIT", "BOSTN", "GRAVY", "DIXIE", "JIMME", "SPICY", "BRGRS",
    "KRENA", "PANGG", "BENKY", "ENDEE", "TOMSN", "LEEVY", "WNDSR",
    "COMET", "BLAZE", "DRAKK", "FENLY", "GORDN", "HETTY", "INMAN",
    "JUNPR", "KYLER", "LOFTY", "MAXIM", "NOVAK", "OAKLY", "PRYNT",
    "QUAIL", "RYVEN", "STOKK", "TRIXX", "UMBER", "VEXAR", "WLKNS",
]
```

## 8. Question Templates

```python
RI_TEMPLATES = [
    "What was {callsign}'s {attribute} when first mentioned?",
    "At the earliest reference to {callsign}, what was {pronoun} {attribute}?",
    "When {callsign} first appears in this log, what was {pronoun} {attribute}?",
    "What was the initial {attribute} recorded for {callsign}?",
]

PI_TEMPLATES = [
    "What was {callsign}'s {attribute} at the most recent update?",
    "In the last recorded entry for {callsign}, what was {pronoun} {attribute}?",
    "What was {callsign}'s most recently noted {attribute}?",
    "At the final mention of {callsign}, what was {pronoun} {attribute}?",
]
```

## 9. Output Format

```json
{
    "id": "atc_001",
    "domain": "air_traffic_control",
    "num_keys": 6,
    "num_updates": 7,
    "narrative": "Chicago Approach, Sector 32. January 14, 2024, 1430Z...",
    "questions": {
        "RI": {
            "question": "What was United 1523's altitude when first mentioned?",
            "expected_answer": "FL370",
            "target_entity": "UAL1523",
            "target_attribute": "altitude_ft"
        },
        "PI": {
            "question": "What was United 1523's altitude at the most recent update?",
            "expected_answer": "4000",
            "target_entity": "UAL1523",
            "target_attribute": "altitude_ft"
        }
    },
    "entity_tracking": {
        "UAL1523 / altitude_ft": ["FL370", "FL280", "FL240", "FL180", "12000", "8000", "4000"],
        "DAL445 / altitude_ft": ["FL350", "FL260", "FL200", "FL180", "10000", "6000", "3000"],
        "AAL2118 / altitude_ft": ["FL330", "FL240", "FL180", "11000", "7000", "4000", "2500"],
        "SWA2241 / altitude_ft": ["FL310", "FL240", "FL180", "10000", "5000", "3000", "1500"],
        "JBU604 / altitude_ft": ["FL290", "FL220", "FL180", "9000", "5000", "3500", "2000"],
        "EDV3847 / altitude_ft": ["FL270", "FL200", "FL180", "8000", "5000", "3000", "1800"]
    },
    "config": {
        "facility": "ORD",
        "facility_type": "tracon",
        "scenario_type": "arrival_rush",
        "aircraft_mode": "same_type",
        "aircraft_types": ["narrowbody"],
        "tracked_attribute": "altitude_ft",
        "attribute_mode": "same",
        "filler_budget": "medium",
        "voice": "sector_log",
        "weather": "ifr_low",
        "archetype": "routine_arrival_rush",
        "seed": 42
    }
}
```

## 10. Implementation Order

1. **Aircraft database** — types, performance envelopes (speeds, altitudes, climb rates, wake category)
2. **Airport database** — runway configs, STARs, SIDs, TRACON/center assignments
3. **Airline/callsign database** — airlines, fleet compositions, callsign generation
4. **Waypoint database** — fix names, positions (relative to airport)
5. **State class** — AircraftState with all mutable attributes (altitude, speed, heading, clearances, phase)
6. **Physics engine** — altitude transitions (climb/descent rates), speed changes, turn rates
7. **Separation checker** — verify all altitude/speed/heading assignments maintain minimums
8. **Event generators** — one function per event type, each enforcing ATC rules
9. **Event scheduler** — decides which event happens next based on scenario type and traffic state
10. **Narrative renderer** — converts event sequence into prose/transcript using templates
11. **Trial generator** — orchestrates everything into a single trial output
12. **Validation** — verify all tracked values appear in narrative, separation maintained, RI/PI answers correct

## 11. Design Decisions (Resolved)

### Q1: Altitude as primary tracked attribute?
**DECISION: Randomly sample from full pool, but altitude is weighted 40%.** Altitude is the gold standard for ATC interference — constantly changing, tight ranges (all jets in FL200-FL410), updated with every clearance. Speed is weighted 25%, heading 15%, remaining distributed across others.

### Q2: Transcript or log voice?
**DECISION: Sample 50/50.** Transcript voice is unique to this domain — no other domain has standardized phraseology. It creates a very different reading experience (shorter, formulaic, more numbers per line). Log voice is closer to the other domains' narrative style. Having both tests whether the model's interference pattern is robust to narrative structure.

### Q3: How to represent altitude — feet or flight levels?
**DECISION: Follow real conventions.** Below 18,000 ft: feet (e.g., "12,000"). At/above 18,000 ft: flight level (e.g., "FL240"). This means the SAME attribute changes format during an arrival descent (FL280 → FL240 → FL180 → 12,000 → 8,000). This is a natural source of difficulty — the model must track that "FL240" and "24,000" are the same type of value.

### Q4: Separation enforcement?
**DECISION: Enforce basic separation.** The state machine checks that no two aircraft are assigned altitudes within 1,000 ft at the same time (unless one is climbing/descending through). Speed assignments must be feasible for aircraft type. This prevents generating physically impossible scenarios that would be flagged by anyone with ATC knowledge.

### Q5: How many airports?
**DECISION: Start with 7 real airports + 1 generic.** ORD, ATL, LAX, JFK, DFW, SFO, DEN cover diverse runway configs and traffic patterns. Generic Class C airport for simpler scenarios. Each with full runway configs and TRACON/center assignments.

### Q6: Do aircraft "land" and disappear?
**DECISION: Yes, same as mortality in wildlife.** When an aircraft lands or departs the sector, it stops generating events. For arrivals, this is the natural endpoint. For en route, handoff to the next sector is the endpoint. This naturally reduces active entities over time, creating asymmetric update counts.

### Q7: Emergency scenarios — how frequent?
**DECISION: ~5% of trials include one emergency aircraft.** Emergencies are rare but create dramatic narrative disruption — other aircraft get delayed/rerouted. The emergency aircraft gets many rapid updates (priority handling) while others stagnate. This tests whether the model handles uneven attention distribution.

## 12. Trial Config Schema (Final)

```python
@dataclass
class ATCTrialConfig:
    num_keys: int              # 2-20 aircraft
    num_updates: int           # 3-25 per aircraft (target)
    condition: str             # "RI" or "PI"
    seed: int                  # reproducibility
    tracked_attribute: str     # "altitude_ft" | "speed_knots" | "heading_degrees" |
                               # "assigned_altitude" | "assigned_speed" | "distance_nm" |
                               # "vertical_rate_fpm" | "fuel_remaining_hrs" |
                               # "delay_minutes" | "separation_nm" |
                               # "clearance_status" | "approach_type" |
                               # "runway_assigned" | "flight_phase" |
                               # "squawk_code" | "special_status" | "mixed"
    attribute_mode: str        # "same" (all aircraft same attr) | "mixed"
    aircraft_mode: str         # "same_type" (all narrowbody) | "mixed_type"
    airport: str               # "ORD" | "ATL" | "LAX" | "JFK" | "DFW" | "SFO" | "DEN" | "generic"
    facility_type: str         # "tracon" | "artcc"
    scenario_type: str         # "arrival_rush" | "departure_push" | "mixed_ops" |
                               # "en_route_sector" | "holding_pattern" | "emergency_scenario"
    weather: str               # "vfr_clear" | "mvfr" | "ifr_low" | "ifr_cat2" | "ifr_cat3" | "thunderstorm"
    filler_budget: str         # "minimal" | "light" | "medium" | "heavy"
    voice: str                 # "transcript" | "sector_log"
    scenario_archetype: str    # see archetypes above
    queried_aircraft_idx: int  # which aircraft to ask about (0 to num_keys-1)
    has_emergency: bool        # whether one aircraft declares emergency
```

**Scenario type → archetype mapping** (each scenario_type maps to its valid archetypes):

```python
SCENARIO_TYPE_ARCHETYPE_MAPPING = {
    "arrival_rush": [
        "routine_arrival_rush",     # standard busy arrival period, good weather
        "weather_hold",             # weather below minimums, aircraft holding
        "heavy_mix",                # A380/B747s mixed in, wake spacing opens gaps
        "parallel_approaches",      # simultaneous approaches to parallel runways
        "vfr_to_ifr_transition",    # weather deteriorating during arrivals
    ],
    "departure_push": [
        "departure_ground_stop",    # GDP in effect, metered departures
        "runway_change",            # wind shift mid-session, reconfigure runways
        "overnight_cargo",          # light traffic, heavy aircraft (FedEx/UPS)
    ],
    "mixed_ops": [
        "routine_arrival_rush",     # standard mixed, mostly routine
        "runway_change",            # wind shift affects both arrivals and departures
        "heavy_mix",                # wake turbulence management in mixed flow
        "emergency_in_sequence",    # emergency disrupts normal mixed ops
        "vfr_to_ifr_transition",    # weather deterioration during mixed ops
    ],
    "en_route_sector": [
        "crossing_traffic",         # multiple aircraft crossing at similar altitudes
        "overnight_cargo",          # heavy cargo transits through sector at night
    ],
    "holding_pattern": [
        "weather_hold",             # holding due to destination weather
        "vfr_to_ifr_transition",    # weather deteriorating, stacking builds
    ],
    "emergency_scenario": [
        "emergency_in_sequence",    # emergency aircraft in normal traffic flow
    ],
}
```

When `seed` is provided, all other fields can be auto-sampled deterministically:
```python
def auto_config(num_keys, num_updates, condition, seed):
    rng = Random(seed)
    aircraft_mode = rng.choices(["same_type", "mixed_type"], weights=[55, 45])[0]
    facility_type = rng.choices(["tracon", "artcc"], weights=[60, 40])[0]

    # Sample scenario_type first, then archetype from valid set
    scenario_type = rng.choices(
        ["arrival_rush", "departure_push", "mixed_ops",
         "en_route_sector", "holding_pattern", "emergency_scenario"],
        weights=[30, 12, 22, 13, 10, 13]
    )[0]
    scenario_archetype = rng.choice(SCENARIO_TYPE_ARCHETYPE_MAPPING[scenario_type])

    # Airport selection: ARTCC uses "N/A" (no specific airport)
    if facility_type == "artcc":
        airport = "N/A"
    else:
        airport = rng.choice(["ORD", "ATL", "LAX", "JFK", "DFW", "SFO", "DEN", "generic"])

    # Weather sampling: all 6 categories reachable
    weather = rng.choices(
        ["vfr_clear", "mvfr", "ifr_low", "ifr_cat2", "ifr_cat3", "thunderstorm"],
        weights=[35, 25, 18, 10, 5, 7]
    )[0]

    # Emergency flag: emergency_scenario always has emergency;
    # other types have 5% chance
    has_emergency = (scenario_type == "emergency_scenario") or (rng.random() < 0.05)

    return ATCTrialConfig(
        num_keys=num_keys,
        num_updates=num_updates,
        condition=condition,
        seed=seed,
        tracked_attribute=rng.choices(
            ["altitude_ft", "speed_knots", "heading_degrees",
             "assigned_altitude", "distance_nm", "clearance_status",
             "approach_type", "flight_phase"],
            weights=[40, 25, 15, 5, 5, 3, 3, 4]
        )[0],
        attribute_mode=rng.choice(["same", "mixed"]),
        aircraft_mode=aircraft_mode,
        airport=airport,
        facility_type=facility_type,
        scenario_type=scenario_type,
        weather=weather,
        filler_budget=rng.choice(["minimal", "light", "medium", "heavy"]),
        voice=rng.choice(["transcript", "sector_log"]),
        scenario_archetype=scenario_archetype,
        queried_aircraft_idx=rng.randint(0, num_keys - 1),
        has_emergency=has_emergency,
    )
```

**ARTCC facility handling**: When `facility_type == "artcc"`, the airport field is `"N/A"` and the scenario uses generic center sector naming:

```python
ARTCC_SECTORS = {
    "ZAU": {"name": "Chicago Center", "sectors": ["Sector 38", "Sector 45", "Sector 22", "Sector 91"]},
    "ZTL": {"name": "Atlanta Center", "sectors": ["Sector 17", "Sector 30", "Sector 56", "Sector 81"]},
    "ZLA": {"name": "Los Angeles Center", "sectors": ["Sector 12", "Sector 28", "Sector 63", "Sector 75"]},
    "ZNY": {"name": "New York Center", "sectors": ["Sector 05", "Sector 19", "Sector 42", "Sector 66"]},
    "ZFW": {"name": "Fort Worth Center", "sectors": ["Sector 33", "Sector 47", "Sector 58", "Sector 72"]},
    "ZOA": {"name": "Oakland Center", "sectors": ["Sector 14", "Sector 36", "Sector 51", "Sector 88"]},
    "ZDV": {"name": "Denver Center", "sectors": ["Sector 09", "Sector 24", "Sector 40", "Sector 67"]},
    "ZMP": {"name": "Minneapolis Center", "sectors": ["Sector 11", "Sector 26", "Sector 44", "Sector 78"]},
    "ZKC": {"name": "Kansas City Center", "sectors": ["Sector 07", "Sector 31", "Sector 52", "Sector 69"]},
    "ZJX": {"name": "Jacksonville Center", "sectors": ["Sector 16", "Sector 35", "Sector 48", "Sector 83"]},
}

def get_artcc_facility_name(rng):
    """When facility_type is artcc, generate a sector name like 'ZAU Sector 38, Chicago Center'."""
    center_code = rng.choice(list(ARTCC_SECTORS.keys()))
    center = ARTCC_SECTORS[center_code]
    sector = rng.choice(center["sectors"])
    return f"{center_code} {sector}, {center['name']}"
```
