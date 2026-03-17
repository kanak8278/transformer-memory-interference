"""Air Traffic Control narrative interference trial generator.

Implements the full generation flow:
- Aircraft/sector/airport selection
- Event sequence generation with scenario-aware weights
- ATC state tracking for all aircraft
- Narrative rendering with transcript and log voice styles
- Question generation for RI/PI conditions
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from ..base import NarrativeTrialGenerator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Altitude step-down targets for arriving aircraft (feet)
ARRIVAL_ALTITUDE_STEPS = [37000, 28000, 24000, 18000, 12000, 8000, 5000, 3000]

# Realistic speed assignment levels by phase (KIAS)
ARRIVAL_SPEED_STEPS = [280, 250, 230, 210, 180, 160, 180, 210]

# Standard event time advances in minutes
EVENT_TIME_ADVANCE = {
    "altitude_assignment":        (1.5, 4.0),
    "speed_assignment":           (1.0, 3.0),
    "heading_assignment":         (1.0, 3.0),
    "approach_clearance":         (2.0, 5.0),
    "landing_clearance":          (1.0, 2.5),
    "takeoff_clearance":          (0.5, 2.0),
    "frequency_change":           (2.0, 5.0),
    "traffic_advisory":           (1.0, 2.5),
    "position_report":            (2.0, 5.0),
    "hold_instruction":           (3.0, 8.0),
    "go_around":                  (1.5, 4.0),
    "emergency":                  (2.0, 6.0),
    "weather_update":             (3.0, 7.0),
}

# Scenario type -> compatible archetypes
SCENARIO_ARCHETYPE_MAP = {
    "arrival_rush":      ["routine_arrival_rush", "weather_hold", "heavy_mix"],
    "departure_push":    ["departure_ground_stop", "runway_change"],
    "mixed_ops":         ["emergency_in_sequence", "parallel_approaches", "vfr_to_ifr"],
    "en_route_sector":   ["crossing_traffic", "overnight_cargo"],
    "holding_pattern":   ["weather_hold"],
}

# Event weights by scenario type
SCENARIO_EVENT_WEIGHTS = {
    "arrival_rush": {
        "altitude_assignment": 35,
        "speed_assignment": 20,
        "heading_assignment": 15,
        "approach_clearance": 12,
        "landing_clearance": 8,
        "takeoff_clearance": 0,
        "frequency_change": 5,
        "traffic_advisory": 4,
        "position_report": 1,
        "hold_instruction": 0,
        "go_around": 0,
        "emergency": 0,
        "weather_update": 0,
    },
    "departure_push": {
        "altitude_assignment": 25,
        "speed_assignment": 10,
        "heading_assignment": 20,
        "approach_clearance": 0,
        "landing_clearance": 0,
        "takeoff_clearance": 30,
        "frequency_change": 10,
        "traffic_advisory": 3,
        "position_report": 2,
        "hold_instruction": 0,
        "go_around": 0,
        "emergency": 0,
        "weather_update": 0,
    },
    "mixed_ops": {
        "altitude_assignment": 25,
        "speed_assignment": 12,
        "heading_assignment": 15,
        "approach_clearance": 10,
        "landing_clearance": 8,
        "takeoff_clearance": 10,
        "frequency_change": 6,
        "traffic_advisory": 6,
        "position_report": 3,
        "hold_instruction": 2,
        "go_around": 2,
        "emergency": 1,
        "weather_update": 0,
    },
    "en_route_sector": {
        "altitude_assignment": 30,
        "speed_assignment": 15,
        "heading_assignment": 10,
        "approach_clearance": 0,
        "landing_clearance": 0,
        "takeoff_clearance": 0,
        "frequency_change": 20,
        "traffic_advisory": 10,
        "position_report": 10,
        "hold_instruction": 2,
        "go_around": 0,
        "emergency": 1,
        "weather_update": 2,
    },
    "holding_pattern": {
        "altitude_assignment": 15,
        "speed_assignment": 5,
        "heading_assignment": 5,
        "approach_clearance": 10,
        "landing_clearance": 5,
        "takeoff_clearance": 0,
        "frequency_change": 5,
        "traffic_advisory": 5,
        "position_report": 5,
        "hold_instruction": 40,
        "go_around": 3,
        "emergency": 2,
        "weather_update": 0,
    },
}

FILLER_CONFIG = {
    "minimal": {"insert_probability": 0.05, "sentences_per_insertion": 1},
    "light":   {"insert_probability": 0.25, "sentences_per_insertion": 1},
    "medium":  {"insert_probability": 0.45, "sentences_per_insertion": (1, 2)},
    "heavy":   {"insert_probability": 0.65, "sentences_per_insertion": (1, 3)},
}

SAFE_TRACKED_ATTRIBUTES = ["altitude_ft", "speed_kias", "heading_degrees"]


# ---------------------------------------------------------------------------
# Altitude formatting helpers
# ---------------------------------------------------------------------------

def format_altitude(alt_ft: int) -> str:
    """Format altitude for entity_tracking and narrative.

    Above 18000: 'FL240'. Below 18000: '12000' (feet).
    """
    if alt_ft >= 18000:
        return f"FL{alt_ft // 100}"
    else:
        return str(alt_ft)


def format_altitude_voice(alt_ft: int) -> str:
    """Format altitude for ATC transcript voice.

    FL240 -> 'flight level two four zero'.
    12000 -> 'one two thousand'.
    """
    digit_words = {
        "0": "zero", "1": "one", "2": "two", "3": "three",
        "4": "four", "5": "five", "6": "six", "7": "seven",
        "8": "eight", "9": "niner"
    }
    if alt_ft >= 18000:
        fl = alt_ft // 100
        digits = " ".join(digit_words[d] for d in str(fl))
        return f"flight level {digits}"
    else:
        thousands = alt_ft // 1000
        hundreds = (alt_ft % 1000) // 100
        if hundreds == 0:
            digits = " ".join(digit_words[d] for d in str(thousands))
            return f"{digits} thousand"
        else:
            t_digits = " ".join(digit_words[d] for d in str(thousands))
            h_digit = digit_words[str(hundreds)]
            return f"{t_digits} thousand {h_digit} hundred"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AircraftState:
    callsign: str
    aircraft_type: str
    airline_radio: str
    wake_category: str
    origin: str
    destination: str
    flight_phase: str
    current_altitude_ft: int
    assigned_altitude_ft: int
    current_speed_kias: int
    assigned_speed_kias: Optional[int]
    heading_degrees: int
    clearance_status: str
    approach_type: Optional[str]
    runway_assigned: Optional[str]
    special_status: str        # "normal", "minimum_fuel", "emergency"
    squawk: str
    fuel_remaining_hrs: float
    distance_from_airport_nm: float
    on_ground: bool = False
    landed: bool = False
    # Internal: step index into arrival altitude/speed sequences
    _altitude_step: int = 0
    _cleared_approach: bool = False
    _cleared_land: bool = False
    _departed: bool = False

    def get_attribute(self, attr_name: str) -> Any:
        if attr_name == "altitude_ft":
            return self.assigned_altitude_ft
        elif attr_name == "speed_kias":
            return self.assigned_speed_kias if self.assigned_speed_kias else self.current_speed_kias
        elif attr_name == "heading_degrees":
            return self.heading_degrees
        return getattr(self, attr_name, 0)

    def format_attribute(self, attr_name: str) -> str:
        """Return canonical string form for narrative and entity_tracking."""
        val = self.get_attribute(attr_name)
        if attr_name == "altitude_ft":
            return format_altitude(int(val))
        else:
            return str(val)


@dataclass
class ATCTrialConfig:
    num_keys: int
    num_updates: int
    seed: int
    tracked_attribute: str
    attribute_mode: str
    airport: str
    facility_type: str
    scenario_type: str
    weather: str
    filler_budget: str
    voice: str
    scenario_archetype: str
    queried_aircraft_idx: int


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _pick_landing_runway(airport_data: dict, rng: random.Random) -> str:
    """Pick a landing runway from airport data."""
    rwy_keys = [k for k in airport_data if k.startswith("runways_landing")]
    if not rwy_keys:
        return "28L"
    key = rng.choice(rwy_keys)
    return rng.choice(airport_data[key])


def _format_zulu_time(minutes: float) -> str:
    """Convert decimal minutes from scenario start into a Zulu time string.
    Base time is 0800Z; each minute is one real minute."""
    base_hour = 8
    base_min = 0
    total_min = int(base_hour * 60 + base_min + minutes)
    h = (total_min // 60) % 24
    m = total_min % 60
    return f"{h:02d}{m:02d}"


def _next_altitude_for_arrival(state: AircraftState, rng: random.Random) -> Optional[int]:
    """Return the next step-down altitude for an arriving aircraft, or None if done."""
    if state._cleared_approach or state.landed:
        return None
    idx = state._altitude_step
    if idx >= len(ARRIVAL_ALTITUDE_STEPS):
        return None
    target = ARRIVAL_ALTITUDE_STEPS[idx]
    # Don't assign if already at or below this altitude
    if state.assigned_altitude_ft <= target:
        state._altitude_step += 1
        if state._altitude_step >= len(ARRIVAL_ALTITUDE_STEPS):
            return None
        target = ARRIVAL_ALTITUDE_STEPS[state._altitude_step]
    return target


# ---------------------------------------------------------------------------
# Main Generator
# ---------------------------------------------------------------------------

class ATCTrialGenerator(NarrativeTrialGenerator):
    """Air Traffic Control narrative interference trial generator."""

    DOMAIN = "atc"

    def __init__(self):
        data_dir = Path(__file__).parent / "data"
        with open(data_dir / "atc_templates.json") as f:
            self.templates = json.load(f)
        with open(data_dir / "atc_data.json") as f:
            self.data = json.load(f)

        self.airports = self.data["airports"]
        self.airlines = self.data["airlines"]
        self.aircraft_wake = self.data["aircraft_wake"]
        self.aircraft_performance = self.data["aircraft_performance"]
        self.waypoints = self.data["waypoints"]
        self.positions = self.data["positions"]
        self.approach_types = self.data["approach_types"]
        self.emergency_types = self.data["emergency_types"]
        self.weather_conditions = self.data["weather_conditions"]
        self.atis_letters = self.data["atis_letters"]

    # -------------------------------------------------------------------
    # Auto-config
    # -------------------------------------------------------------------

    def _auto_config(self, num_keys: int, num_updates: int,
                     seed: int, rng: random.Random) -> ATCTrialConfig:
        tracked_attribute = rng.choice(SAFE_TRACKED_ATTRIBUTES)
        attribute_mode = rng.choice(["same", "mixed"])
        airport = rng.choice(list(self.airports.keys()))
        filler_budget = rng.choice(["minimal", "light", "medium", "heavy"])
        voice = rng.choice(["transcript", "log"])
        weather = rng.choice(["vmc", "imc", "marginal"])

        # facility type: artcc forces en_route_sector
        facility_type = rng.choice(["tracon", "tracon", "tracon", "artcc"])
        if facility_type == "artcc":
            scenario_type = "en_route_sector"
        else:
            scenario_type = rng.choice(list(SCENARIO_ARCHETYPE_MAP.keys()))

        # Enforce compatible archetype
        scenario_archetype = rng.choice(SCENARIO_ARCHETYPE_MAP[scenario_type])

        queried_aircraft_idx = rng.randint(0, num_keys - 1)

        return ATCTrialConfig(
            num_keys=num_keys,
            num_updates=num_updates,
            seed=seed,
            tracked_attribute=tracked_attribute,
            attribute_mode=attribute_mode,
            airport=airport,
            facility_type=facility_type,
            scenario_type=scenario_type,
            weather=weather,
            filler_budget=filler_budget,
            voice=voice,
            scenario_archetype=scenario_archetype,
            queried_aircraft_idx=queried_aircraft_idx,
        )

    # -------------------------------------------------------------------
    # Aircraft initialization
    # -------------------------------------------------------------------

    def _generate_callsign(self, rng: random.Random, used: set) -> str:
        """Generate a unique airline callsign."""
        for _ in range(50):
            airline_code = rng.choice(list(self.airlines.keys()))
            number = rng.randint(100, 9999)
            callsign = f"{airline_code}{number}"
            if callsign not in used:
                return callsign
        return f"N{rng.randint(10000, 99999)}"

    def _init_aircraft_state(self, callsign: str, scenario_type: str,
                              airport_code: str, rng: random.Random) -> AircraftState:
        """Initialize aircraft state based on scenario type."""
        airline_code = callsign[:3] if callsign[:3] in self.airlines else "AAL"
        airline_info = self.airlines.get(airline_code, self.airlines["AAL"])
        aircraft_type = rng.choice(airline_info["fleet"])
        airline_radio = airline_info["radio"]
        wake_cat = self.aircraft_wake.get(aircraft_type, "E")
        perf = self.aircraft_performance.get(aircraft_type, self.aircraft_performance["B738"])

        airport_data = self.airports.get(airport_code, self.airports["ORD"])
        rwy_keys = [k for k in airport_data if k.startswith("runways_landing")]
        if rwy_keys:
            runway = rng.choice(airport_data[rng.choice(rwy_keys)])
        else:
            runway = "28L"

        origins = ["KBOS", "KLGA", "KEWR", "KPHX", "KDEN", "KMCO", "KSEA", "KLAS", "KDTW", "KMIA"]
        origin = rng.choice([a for a in origins if a != f"K{airport_code}"])
        destination = f"K{airport_code}"

        squawk = f"{rng.randint(1, 7)}{rng.randint(0, 7)}{rng.randint(0, 7)}{rng.randint(0, 7)}"
        fuel = round(rng.uniform(2.0, 6.0), 1)

        if scenario_type == "arrival_rush":
            # Arriving aircraft: high altitude, far out
            alt_ft = rng.choice([37000, 35000, 33000, 31000, 29000, 27000])
            speed = rng.randint(270, 290)
            heading = rng.randint(0, 359)
            dist = rng.uniform(50, 120)
            phase = "descent"
            on_ground = False
        elif scenario_type == "departure_push":
            # Departing or just airborne
            if rng.random() < 0.4:
                alt_ft = 0
                speed = 0
                heading = rng.randint(0, 359)
                dist = 0
                phase = "ground"
                on_ground = True
            else:
                alt_ft = rng.choice([3000, 5000, 7000, 8000, 9000])
                speed = rng.randint(200, 250)
                heading = rng.randint(0, 359)
                dist = rng.uniform(5, 30)
                phase = "climb"
                on_ground = False
        elif scenario_type == "en_route_sector":
            alt_ft = rng.choice([31000, 33000, 35000, 37000, 39000, 41000])
            speed = rng.randint(450, 490)  # groundspeed-ish, simplified as KIAS
            heading = rng.randint(0, 359)
            dist = rng.uniform(80, 200)
            phase = "cruise"
            on_ground = False
        elif scenario_type == "holding_pattern":
            alt_ft = rng.choice([8000, 9000, 10000, 11000, 12000])
            speed = rng.randint(200, 220)
            heading = rng.randint(0, 359)
            dist = rng.uniform(20, 50)
            phase = "holding"
            on_ground = False
        else:  # mixed_ops
            if rng.random() < 0.5:
                alt_ft = rng.choice([37000, 33000, 28000, 24000])
                speed = rng.randint(260, 290)
                dist = rng.uniform(40, 100)
                phase = "descent"
            else:
                alt_ft = rng.choice([5000, 8000, 10000])
                speed = rng.randint(220, 260)
                dist = rng.uniform(10, 40)
                phase = "climb"
            heading = rng.randint(0, 359)
            on_ground = False

        return AircraftState(
            callsign=callsign,
            aircraft_type=aircraft_type,
            airline_radio=airline_radio,
            wake_category=wake_cat,
            origin=origin,
            destination=destination,
            flight_phase=phase,
            current_altitude_ft=alt_ft,
            assigned_altitude_ft=alt_ft,
            current_speed_kias=speed,
            assigned_speed_kias=speed if speed > 0 else None,
            heading_degrees=heading,
            clearance_status="cleared",
            approach_type=None,
            runway_assigned=runway,
            special_status="normal",
            squawk=squawk,
            fuel_remaining_hrs=fuel,
            distance_from_airport_nm=dist if not on_ground else 0.0,
            on_ground=on_ground,
            landed=False,
            _altitude_step=0,
            _cleared_approach=False,
            _cleared_land=False,
            _departed=False,
        )

    # -------------------------------------------------------------------
    # Event weight selection
    # -------------------------------------------------------------------

    def _get_scenario_event_weights(self, scenario_type: str) -> dict:
        """Return event weights dict for a given scenario type."""
        return dict(SCENARIO_EVENT_WEIGHTS.get(scenario_type, SCENARIO_EVENT_WEIGHTS["arrival_rush"]))

    # -------------------------------------------------------------------
    # Event handlers
    # -------------------------------------------------------------------

    def _handle_altitude_assignment(
        self, callsign: str, ac_states: Dict[str, AircraftState],
        sim_time: float, rng: random.Random,
        config: ATCTrialConfig, tracked: Dict[str, str],
        airport_code: str
    ) -> Tuple[dict, list]:
        ac = ac_states[callsign]
        attr_name = tracked[callsign]
        old_val = ac.format_attribute(attr_name)

        prev_alt = ac.assigned_altitude_ft

        # Determine new altitude
        if ac.flight_phase in ("descent", "holding") or config.scenario_type in ("arrival_rush", "holding_pattern"):
            new_alt = _next_altitude_for_arrival(ac, rng)
            if new_alt is None:
                # Staying at current or small adjustment
                new_alt = max(3000, ac.assigned_altitude_ft - rng.choice([1000, 2000, 3000]))
            ac._altitude_step += 1
        elif ac.flight_phase == "climb" or config.scenario_type == "departure_push":
            climb_targets = [5000, 8000, 10000, 15000, 18000, 24000, 28000, 35000, 39000]
            above = [t for t in climb_targets if t > ac.assigned_altitude_ft]
            new_alt = rng.choice(above) if above else ac.assigned_altitude_ft + 4000
        elif ac.flight_phase == "cruise" or config.scenario_type == "en_route_sector":
            # FL change: +/- one flight level step (1000 or 2000 ft)
            delta = rng.choice([-2000, -1000, 1000, 2000])
            new_alt = max(18000, min(43000, ac.assigned_altitude_ft + delta))
        else:
            new_alt = max(3000, ac.assigned_altitude_ft - rng.choice([2000, 3000, 5000]))

        new_alt = max(0, new_alt)
        ac.assigned_altitude_ft = new_alt
        ac.current_altitude_ft = new_alt  # simplified: assume immediate compliance

        new_val = ac.format_attribute(attr_name)

        # Ensure value changed for tracked attribute
        if attr_name == "altitude_ft" and new_val == old_val:
            new_alt = max(0, new_alt - 1000) if new_alt > 3000 else new_alt + 1000
            ac.assigned_altitude_ft = new_alt
            ac.current_altitude_ft = new_alt
            new_val = ac.format_attribute(attr_name)

        fix = rng.choice(self.waypoints)
        time_str = _format_zulu_time(sim_time)
        sep_notes = [
            "Separation maintained.",
            "5 miles in trail behind preceding traffic.",
            "Adequate spacing confirmed.",
            "Traffic advisory issued.",
        ]
        contexts = [
            "Standard step-down for sequencing.",
            "Traffic permitting, further descent expected.",
            "Per STAR profile.",
            "Coordination with adjacent sector complete.",
        ]

        event_data = {
            "callsign": callsign,
            "aircraft_type": ac.aircraft_type,
            "attr_value": format_altitude(new_alt),
            "attr_value_readback": format_altitude(new_alt),
            "prev_value": format_altitude(prev_alt),
            "time": time_str,
            "fix": fix,
            "eta": rng.randint(5, 30),
            "dist": round(ac.distance_from_airport_nm, 0),
            "separation_note": rng.choice(sep_notes),
            "context": rng.choice(contexts),
            "attr_name": "altitude",
            "attr_label": "altitude",
        }

        mentioned = (attr_name == "altitude_ft")
        state_changes = [
            (callsign, attr_name, old_val, new_val, mentioned),
        ]
        return event_data, state_changes

    def _handle_speed_assignment(
        self, callsign: str, ac_states: Dict[str, AircraftState],
        sim_time: float, rng: random.Random,
        config: ATCTrialConfig, tracked: Dict[str, str],
        airport_code: str
    ) -> Tuple[dict, list]:
        ac = ac_states[callsign]
        attr_name = tracked[callsign]
        old_val = ac.format_attribute(attr_name)

        prev_speed = ac.assigned_speed_kias or ac.current_speed_kias

        # Realistic speed targets
        if ac.flight_phase == "descent":
            speed_options = [280, 260, 250, 230, 210, 180, 160]
            below = [s for s in speed_options if s < prev_speed]
            new_speed = rng.choice(below) if below else rng.choice([180, 160, 150])
        elif ac.flight_phase == "cruise":
            speed_options = [300, 320, 340, 360, 380]
            new_speed = rng.choice(speed_options)
        elif ac.flight_phase == "holding":
            new_speed = rng.choice([210, 220, 230])
        else:
            new_speed = rng.choice([210, 230, 250, 270])

        ac.assigned_speed_kias = new_speed
        ac.current_speed_kias = new_speed

        new_val = ac.format_attribute(attr_name)

        if attr_name == "speed_kias" and new_val == old_val:
            new_speed = new_speed + rng.choice([-10, 10, 20])
            ac.assigned_speed_kias = new_speed
            ac.current_speed_kias = new_speed
            new_val = ac.format_attribute(attr_name)

        fix = rng.choice(self.waypoints)
        time_str = _format_zulu_time(sim_time)

        preceding_options = ["a B737", "an A320", "a CRJ9", "a heavy", "a B789"]
        sep_notes = [
            f"For spacing behind {rng.choice(preceding_options)}.",
            "Traffic interval established.",
            "Merging sequence maintained.",
        ]

        event_data = {
            "callsign": callsign,
            "aircraft_type": ac.aircraft_type,
            "attr_value": str(new_speed),
            "attr_value_readback": str(new_speed),
            "prev_value": str(prev_speed),
            "time": time_str,
            "fix": fix,
            "dist": round(ac.distance_from_airport_nm, 0),
            "preceding": rng.choice(preceding_options),
            "separation_note": rng.choice(sep_notes),
            "attr_name": "speed",
            "attr_label": "speed (knots)",
        }

        mentioned = (attr_name == "speed_kias")
        state_changes = [
            (callsign, attr_name, old_val, new_val, mentioned),
        ]
        return event_data, state_changes

    def _handle_heading_assignment(
        self, callsign: str, ac_states: Dict[str, AircraftState],
        sim_time: float, rng: random.Random,
        config: ATCTrialConfig, tracked: Dict[str, str],
        airport_code: str
    ) -> Tuple[dict, list]:
        ac = ac_states[callsign]
        attr_name = tracked[callsign]
        old_val = ac.format_attribute(attr_name)

        prev_hdg = ac.heading_degrees
        # Pick a heading meaningfully different from current
        delta = rng.choice([30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
        new_hdg = (prev_hdg + delta) % 360

        ac.heading_degrees = new_hdg
        new_val = ac.format_attribute(attr_name)

        if attr_name == "heading_degrees" and new_val == old_val:
            new_hdg = (new_hdg + 30) % 360
            ac.heading_degrees = new_hdg
            new_val = ac.format_attribute(attr_name)

        direction = "left" if delta > 180 else "right"
        directions_readback = {"left": "left", "right": "right"}

        fix = rng.choice(self.waypoints)
        rwy = ac.runway_assigned or "28L"
        time_str = _format_zulu_time(sim_time)

        reason_options = [
            "sequencing",
            "traffic avoidance",
            "weather deviation",
            f"ILS runway {rwy} intercept",
            "spacing behind preceding traffic",
        ]
        sep_notes = ["Separation maintained.", "Spacing adequate.", "Traffic advisory issued."]

        event_data = {
            "callsign": callsign,
            "aircraft_type": ac.aircraft_type,
            "attr_value": str(new_hdg),
            "attr_value_readback": str(new_hdg),
            "prev_value": str(prev_hdg),
            "time": time_str,
            "fix": fix,
            "rwy": rwy,
            "direction": direction,
            "direction_readback": directions_readback[direction],
            "dist": round(ac.distance_from_airport_nm, 0),
            "reason": rng.choice(reason_options),
            "separation_note": rng.choice(sep_notes),
            "alt": format_altitude(ac.assigned_altitude_ft),
            "hdg": str(new_hdg),
            "attr_name": "heading",
            "attr_label": "heading (degrees)",
        }

        mentioned = (attr_name == "heading_degrees")
        state_changes = [
            (callsign, attr_name, old_val, new_val, mentioned),
        ]
        return event_data, state_changes

    def _handle_approach_clearance(
        self, callsign: str, ac_states: Dict[str, AircraftState],
        sim_time: float, rng: random.Random,
        config: ATCTrialConfig, tracked: Dict[str, str],
        airport_code: str
    ) -> Tuple[dict, list]:
        ac = ac_states[callsign]
        attr_name = tracked[callsign]
        old_val = ac.format_attribute(attr_name)

        # Set to approach altitude and mark cleared
        approach_alt = 3000
        ac.assigned_altitude_ft = approach_alt
        ac.current_altitude_ft = approach_alt
        ac._cleared_approach = True
        ac.flight_phase = "approach"

        approach_type = rng.choice(self.approach_types[:4])  # ILS, RNAV, VOR, ILS or LOC
        ac.approach_type = approach_type

        rwy = ac.runway_assigned or "28L"
        dist = round(rng.uniform(8, 25), 0)
        fix = rng.choice(self.waypoints)
        time_str = _format_zulu_time(sim_time)

        new_val = ac.format_attribute(attr_name)

        preceding_types = ["B737", "A320", "CRJ9", "B789", "heavy"]
        sep_notes = ["ILS established.", "Gear check completed.", "Traffic in sight."]
        contexts = ["Sequenced behind preceding arrival.", "Visual contact maintained.", "Procedure turn not required."]

        event_data = {
            "callsign": callsign,
            "aircraft_type": ac.aircraft_type,
            "attr_value": format_altitude(approach_alt),
            "prev_value": old_val,
            "time": time_str,
            "fix": fix,
            "rwy": rwy,
            "dist": int(dist),
            "approach_type": approach_type,
            "hdg": str(ac.heading_degrees),
            "alt": format_altitude(approach_alt),
            "preceding_type": rng.choice(preceding_types),
            "preceding_dist": rng.randint(3, 8),
            "separation_note": rng.choice(sep_notes),
            "context": rng.choice(contexts),
            "attr_name": "altitude",
            "attr_label": "altitude",
        }

        mentioned = (attr_name == "altitude_ft")
        state_changes = [
            (callsign, attr_name, old_val, new_val, mentioned),
        ]
        return event_data, state_changes

    def _handle_landing_clearance(
        self, callsign: str, ac_states: Dict[str, AircraftState],
        sim_time: float, rng: random.Random,
        config: ATCTrialConfig, tracked: Dict[str, str],
        airport_code: str
    ) -> Tuple[dict, list]:
        ac = ac_states[callsign]
        attr_name = tracked[callsign]
        old_val = ac.format_attribute(attr_name)

        rwy = ac.runway_assigned or "28L"
        ac._cleared_land = True
        ac.flight_phase = "final"

        wc = self.weather_conditions.get(config.weather, self.weather_conditions["vmc"])
        wind_idx = rng.randint(0, len(wc["wind"]) - 1)
        wind = wc["wind"][wind_idx]
        time_str = _format_zulu_time(sim_time)

        # Speed on final
        perf = self.aircraft_performance.get(ac.aircraft_type, self.aircraft_performance["B738"])
        final_speed = perf["approach_kias"]
        ac.assigned_speed_kias = final_speed
        ac.current_speed_kias = final_speed

        new_val = ac.format_attribute(attr_name)

        preceding_types = ["B737", "A320", "CRJ9", "B789", "heavy"]
        seq = rng.randint(1, 4)

        event_data = {
            "callsign": callsign,
            "aircraft_type": ac.aircraft_type,
            "attr_value": str(final_speed),
            "prev_value": old_val,
            "time": time_str,
            "rwy": rwy,
            "wind": wind,
            "seq": seq,
            "preceding_type": rng.choice(preceding_types),
            "preceding_dist": rng.randint(2, 6),
            "attr_name": "speed",
            "attr_label": "speed (knots)",
        }

        mentioned = (attr_name == "speed_kias")
        state_changes = [
            (callsign, attr_name, old_val, new_val, mentioned),
        ]
        return event_data, state_changes

    def _handle_takeoff_clearance(
        self, callsign: str, ac_states: Dict[str, AircraftState],
        sim_time: float, rng: random.Random,
        config: ATCTrialConfig, tracked: Dict[str, str],
        airport_code: str
    ) -> Tuple[dict, list]:
        ac = ac_states[callsign]
        attr_name = tracked[callsign]
        old_val = ac.format_attribute(attr_name)

        # Depart: assign initial climb altitude
        initial_alt = rng.choice([3000, 5000, 8000])
        initial_hdg = rng.randint(0, 359)
        ac.assigned_altitude_ft = initial_alt
        ac.current_altitude_ft = 0  # rolling
        ac.heading_degrees = initial_hdg
        ac.flight_phase = "climb"
        ac.on_ground = False
        ac._departed = True

        new_val = ac.format_attribute(attr_name)

        rwy = ac.runway_assigned or "28L"
        time_str = _format_zulu_time(sim_time)
        wc = self.weather_conditions.get(config.weather, self.weather_conditions["vmc"])
        wind_idx = rng.randint(0, len(wc["wind"]) - 1)
        wind = wc["wind"][wind_idx]
        sid_list = self.airports.get(airport_code, {}).get("sids", ["OHARE8"])
        sid = rng.choice(sid_list)

        event_data = {
            "callsign": callsign,
            "aircraft_type": ac.aircraft_type,
            "attr_value": format_altitude(initial_alt),
            "prev_value": old_val,
            "time": time_str,
            "rwy": rwy,
            "wind": wind,
            "fix": sid,
            "alt": format_altitude(initial_alt),
            "hdg": str(initial_hdg),
            "attr_name": "altitude",
            "attr_label": "altitude",
        }

        mentioned = (attr_name == "altitude_ft")
        state_changes = [
            (callsign, attr_name, old_val, new_val, mentioned),
        ]
        return event_data, state_changes

    def _handle_frequency_change(
        self, callsign: str, ac_states: Dict[str, AircraftState],
        sim_time: float, rng: random.Random,
        config: ATCTrialConfig, tracked: Dict[str, str],
        airport_code: str
    ) -> Tuple[dict, list]:
        ac = ac_states[callsign]
        attr_name = tracked[callsign]
        old_val = ac.format_attribute(attr_name)
        new_val = old_val  # frequency change doesn't alter tracked attribute

        freq_options = ["119.35", "120.15", "121.25", "124.00", "125.05", "126.90",
                        "127.55", "128.20", "129.00", "132.10", "133.35", "135.00"]
        freq = rng.choice(freq_options)
        facility_options = ["Chicago Approach", "Atlanta Approach", "LA Center", "New York Center",
                            "Dallas Approach", "Chicago Center", "departure", "tower", "ground"]
        facility = rng.choice(facility_options)
        rwy = ac.runway_assigned or "28L"
        time_str = _format_zulu_time(sim_time)

        event_data = {
            "callsign": callsign,
            "aircraft_type": ac.aircraft_type,
            "attr_value": freq,
            "time": time_str,
            "freq": freq,
            "facility": facility,
            "rwy": rwy,
        }

        # Frequency changes do not constitute a tracked attribute mention
        state_changes = [
            (callsign, attr_name, old_val, new_val, False),
        ]
        return event_data, state_changes

    def _handle_traffic_advisory(
        self, callsign: str, ac_states: Dict[str, AircraftState],
        sim_time: float, rng: random.Random,
        config: ATCTrialConfig, tracked: Dict[str, str],
        all_aircraft: list,
        airport_code: str
    ) -> Tuple[dict, list]:
        ac = ac_states[callsign]
        attr_name = tracked[callsign]
        old_val = ac.format_attribute(attr_name)
        new_val = old_val  # advisory doesn't change tracked attr

        clock = rng.choice(self.templates["clock_positions"])
        dist = rng.randint(3, 15)
        traffic_alt_ft = rng.choice([5000, 8000, 10000, 12000, 15000, 18000, 24000])
        traffic_alt = format_altitude(traffic_alt_ft)
        dir_options = ["northeast", "northwest", "southeast", "southwest", "north", "south", "east", "west"]
        time_str = _format_zulu_time(sim_time)

        other_types = ["B738", "A320", "CRJ9", "B789", "E75L", "A321"]

        event_data = {
            "callsign": callsign,
            "aircraft_type": ac.aircraft_type,
            "clock": clock,
            "dist": dist,
            "traffic_alt": traffic_alt,
            "dir": rng.choice(dir_options),
            "time": time_str,
        }

        state_changes = [
            (callsign, attr_name, old_val, new_val, False),
        ]
        return event_data, state_changes

    def _handle_position_report(
        self, callsign: str, ac_states: Dict[str, AircraftState],
        sim_time: float, rng: random.Random,
        config: ATCTrialConfig, tracked: Dict[str, str],
        airport_code: str
    ) -> Tuple[dict, list]:
        ac = ac_states[callsign]
        attr_name = tracked[callsign]
        old_val = ac.format_attribute(attr_name)
        new_val = old_val  # position report confirms, doesn't change value

        fix = rng.choice(self.waypoints)
        next_fix = rng.choice([w for w in self.waypoints if w != fix])
        time_str = _format_zulu_time(sim_time)
        current_alt = format_altitude(ac.current_altitude_ft)

        event_data = {
            "callsign": callsign,
            "aircraft_type": ac.aircraft_type,
            "attr_value": current_alt,
            "current_alt": current_alt,
            "time": time_str,
            "fix": fix,
            "next_fix": next_fix,
            "dist": round(ac.distance_from_airport_nm, 0),
            "squawk": ac.squawk,
            "souls": rng.randint(80, 350),
            "fuel": round(ac.fuel_remaining_hrs, 1),
            "seq": rng.randint(1, 6),
        }

        state_changes = [
            (callsign, attr_name, old_val, new_val, False),
        ]
        return event_data, state_changes

    def _handle_hold_instruction(
        self, callsign: str, ac_states: Dict[str, AircraftState],
        sim_time: float, rng: random.Random,
        config: ATCTrialConfig, tracked: Dict[str, str],
        airport_code: str
    ) -> Tuple[dict, list]:
        ac = ac_states[callsign]
        attr_name = tracked[callsign]
        old_val = ac.format_attribute(attr_name)

        hold_alt = rng.choice([8000, 9000, 10000, 11000, 12000])
        ac.assigned_altitude_ft = hold_alt
        ac.current_altitude_ft = hold_alt
        ac.flight_phase = "holding"

        new_val = ac.format_attribute(attr_name)

        if attr_name == "altitude_ft" and new_val == old_val:
            hold_alt += 1000
            ac.assigned_altitude_ft = hold_alt
            ac.current_altitude_ft = hold_alt
            new_val = ac.format_attribute(attr_name)

        fix = rng.choice(self.waypoints)
        delay = rng.randint(10, 45)
        efc_min = sim_time + delay
        efc_str = _format_zulu_time(efc_min)
        direction_opts = ["left", "right"]
        leg_min = rng.choice([1, 2])
        time_str = _format_zulu_time(sim_time)

        reason_options = [
            "traffic congestion",
            "weather at destination",
            "runway change in progress",
            "departure flow restrictions",
            "ground stop at destination",
        ]
        sep_notes = ["Stack managed.", "Sequence maintained.", "Advisory issued to crew."]

        event_data = {
            "callsign": callsign,
            "aircraft_type": ac.aircraft_type,
            "attr_value": format_altitude(hold_alt),
            "prev_value": old_val,
            "time": time_str,
            "fix": fix,
            "delay": delay,
            "efc": efc_str,
            "direction": rng.choice(direction_opts),
            "leg_min": leg_min,
            "alt": format_altitude(hold_alt),
            "reason": rng.choice(reason_options),
            "stack_count": rng.randint(2, 6),
            "separation_note": rng.choice(sep_notes),
            "attr_name": "altitude",
            "attr_label": "altitude",
        }

        mentioned = (attr_name == "altitude_ft")
        state_changes = [
            (callsign, attr_name, old_val, new_val, mentioned),
        ]
        return event_data, state_changes

    def _handle_go_around(
        self, callsign: str, ac_states: Dict[str, AircraftState],
        sim_time: float, rng: random.Random,
        config: ATCTrialConfig, tracked: Dict[str, str],
        airport_code: str
    ) -> Tuple[dict, list]:
        ac = ac_states[callsign]
        attr_name = tracked[callsign]
        old_val = ac.format_attribute(attr_name)

        ga_alt = rng.choice([3000, 4000, 5000])
        ac.assigned_altitude_ft = ga_alt
        ac.current_altitude_ft = ga_alt
        ac._cleared_approach = False
        ac._cleared_land = False
        ac.flight_phase = "missed_approach"

        new_val = ac.format_attribute(attr_name)

        if attr_name == "altitude_ft" and new_val == old_val:
            ga_alt += 1000
            ac.assigned_altitude_ft = ga_alt
            ac.current_altitude_ft = ga_alt
            new_val = ac.format_attribute(attr_name)

        rwy = ac.runway_assigned or "28L"
        hdg = rng.randint(0, 359)
        time_str = _format_zulu_time(sim_time)

        event_data = {
            "callsign": callsign,
            "aircraft_type": ac.aircraft_type,
            "attr_value": format_altitude(ga_alt),
            "prev_value": old_val,
            "time": time_str,
            "rwy": rwy,
            "alt": format_altitude(ga_alt),
            "hdg": str(hdg),
            "attr_name": "altitude",
            "attr_label": "altitude",
        }

        mentioned = (attr_name == "altitude_ft")
        state_changes = [
            (callsign, attr_name, old_val, new_val, mentioned),
        ]
        return event_data, state_changes

    def _handle_emergency(
        self, callsign: str, ac_states: Dict[str, AircraftState],
        sim_time: float, rng: random.Random,
        config: ATCTrialConfig, tracked: Dict[str, str],
        airport_code: str
    ) -> Tuple[dict, list]:
        ac = ac_states[callsign]
        attr_name = tracked[callsign]
        old_val = ac.format_attribute(attr_name)

        # Emergency: descend direct to airport
        emergency_alt = rng.choice([5000, 8000, 10000])
        ac.assigned_altitude_ft = emergency_alt
        ac.current_altitude_ft = emergency_alt
        ac.special_status = "emergency"
        ac.flight_phase = "emergency_descent"

        new_val = ac.format_attribute(attr_name)

        if attr_name == "altitude_ft" and new_val == old_val:
            emergency_alt = max(3000, emergency_alt - 2000)
            ac.assigned_altitude_ft = emergency_alt
            ac.current_altitude_ft = emergency_alt
            new_val = ac.format_attribute(attr_name)

        rwy = ac.runway_assigned or "28L"
        emergency_type = rng.choice(self.emergency_types)
        souls = rng.randint(80, 350)
        fuel = round(ac.fuel_remaining_hrs, 1)
        time_str = _format_zulu_time(sim_time)

        event_data = {
            "callsign": callsign,
            "aircraft_type": ac.aircraft_type,
            "attr_value": format_altitude(emergency_alt),
            "prev_value": old_val,
            "time": time_str,
            "rwy": rwy,
            "alt": format_altitude(emergency_alt),
            "emergency_type": emergency_type,
            "souls": souls,
            "fuel": fuel,
            "airport": f"K{airport_code}",
            "attr_name": "altitude",
            "attr_label": "altitude",
        }

        mentioned = (attr_name == "altitude_ft")
        state_changes = [
            (callsign, attr_name, old_val, new_val, mentioned),
        ]
        return event_data, state_changes

    def _handle_weather_update(
        self, callsign: str, ac_states: Dict[str, AircraftState],
        sim_time: float, rng: random.Random,
        config: ATCTrialConfig, tracked: Dict[str, str],
        airport_code: str
    ) -> Tuple[dict, list]:
        ac = ac_states[callsign]
        attr_name = tracked[callsign]
        old_val = ac.format_attribute(attr_name)
        new_val = old_val  # weather update doesn't change tracked attr

        airport_data = self.airports.get(airport_code, self.airports["ORD"])
        wc = self.weather_conditions.get(config.weather, self.weather_conditions["vmc"])
        idx = rng.randint(0, len(wc["ceiling"]) - 1)
        ceiling = wc["ceiling"][idx]
        vis = wc["vis"][idx]
        wind = wc["wind"][idx]
        time_str = _format_zulu_time(sim_time)
        atis_letter = rng.choice(self.atis_letters)
        callsign2 = rng.choice([c for c in list(ac_states.keys()) if c != callsign] or [callsign])
        alt_str = format_altitude(ac.current_altitude_ft)
        ride = rng.choice(self.templates["ride_reports"])
        altimeter = f"{rng.randint(2942, 3030) / 100:.2f}"
        alternate_options = ["KMDW", "KPWK", "KDPA", "KGYY", "KELP", "KFLL"]
        dir_options = ["west", "east", "north", "south"]

        event_data = {
            "callsign": callsign,
            "aircraft_type": ac.aircraft_type,
            "attr_value": alt_str,
            "time": time_str,
            "airport": airport_data.get("icao", f"K{airport_code}"),
            "ceiling": ceiling,
            "vis": vis,
            "wind": wind,
            "atis_letter": atis_letter,
            "altimeter": altimeter,
            "callsign2": callsign2,
            "ride": ride,
            "alt": alt_str,
            "magnitude": rng.randint(10, 25),
            "dir": rng.choice(dir_options),
            "alternate": rng.choice(alternate_options),
        }

        state_changes = [
            (callsign, attr_name, old_val, new_val, False),
        ]
        return event_data, state_changes

    # -------------------------------------------------------------------
    # Filler rendering
    # -------------------------------------------------------------------

    def _render_filler(self, template: str, callsign: str,
                       ac_states: Dict[str, AircraftState],
                       rng: random.Random, tracked: Dict[str, str],
                       all_aircraft: list,
                       airport_code: str,
                       sim_time: float) -> str:
        ac = ac_states[callsign]

        other_callsigns = [c for c in all_aircraft if c != callsign]
        c1 = callsign
        c2 = rng.choice(other_callsigns) if other_callsigns else callsign

        fix = rng.choice(self.waypoints)
        next_fix = rng.choice([w for w in self.waypoints if w != fix])
        time_str = _format_zulu_time(sim_time)
        atis_letter = rng.choice(self.atis_letters)
        alt_str = format_altitude(ac.current_altitude_ft)
        ride = rng.choice(self.templates["ride_reports"])
        sectors = ["Chicago Center", "New York Center", "LA Center", "Atlanta Center",
                   "Denver Center", "Departure", "Tower"]
        coord_note = rng.choice(self.templates["coordination_notes"])
        dir_options = ["north", "south", "east", "west", "northeast", "southwest"]
        freq_options = ["119.35", "120.15", "121.25", "124.00", "125.05"]
        facility_options = ["Chicago Approach", "Atlanta Approach", "LA Center", "departure"]
        rwy = ac.runway_assigned or "28L"

        separation_notes = ["adequate — 4.5 nm", "tight — 3.1 nm", "good — 5.8 nm",
                            "minimal standard — 3.0 nm", "comfortable — 6.2 nm"]
        context_notes = ["traffic was building as the morning push intensified",
                         "weather holding at destination caused sequencing delays",
                         "the sector was busy with a mix of arrivals and departures",
                         "coordination with adjacent sector completed without issue",
                         "runway configuration change was in progress"]
        eta_options = ["1430Z", "1445Z", "1502Z", "1518Z", "1535Z", "1550Z", "1605Z"]

        data = {
            "callsign": callsign,
            "c1": c1,
            "c2": c2,
            "aircraft_type": ac.aircraft_type,
            "fix": fix,
            "next_fix": next_fix,
            "time": time_str,
            "atis_letter": atis_letter,
            "letter": atis_letter,
            "current_alt": alt_str,
            "alt": alt_str,
            "ride": ride,
            "dist": rng.randint(3, 25),
            "dir": rng.choice(dir_options),
            "sector": rng.choice(sectors),
            "squawk": ac.squawk,
            "souls": rng.randint(80, 350),
            "fuel": round(ac.fuel_remaining_hrs, 1),
            "freq": rng.choice(freq_options),
            "facility": rng.choice(facility_options),
            "rwy": rwy,
            "seq": rng.randint(1, 6),
            "traffic_count": rng.randint(8, 22),
            "altimeter": f"{rng.randint(2942, 3030) / 100:.2f}",
            "separation_note": rng.choice(separation_notes),
            "context": rng.choice(context_notes),
            "eta": rng.choice(eta_options),
            "preceding": rng.choice([c for c in all_aircraft if c != callsign] or [callsign]),
            "change_note": rng.choice(["winds shifted to 280 at 18", "ceiling dropped to 800 OVC",
                                       "visibility improved to 5 SM", "runway change to 28L"]),
        }

        return self.render_template(template, data)

    # -------------------------------------------------------------------
    # Header rendering
    # -------------------------------------------------------------------

    def _render_header(self, template: str, airport_code: str,
                       config: ATCTrialConfig, rng: random.Random,
                       sim_time: float) -> str:
        airport_data = self.airports.get(airport_code, self.airports["ORD"])
        wc = self.weather_conditions.get(config.weather, self.weather_conditions["vmc"])
        idx = rng.randint(0, len(wc["ceiling"]) - 1)
        ceiling = wc["ceiling"][idx]
        vis = wc["vis"][idx]
        wind = wc["wind"][idx]

        rwy_keys = [k for k in airport_data if k.startswith("runways_landing")]
        rwy = "28L"
        if rwy_keys:
            rwy = rng.choice(airport_data[rng.choice(rwy_keys)])

        atis_letter = rng.choice(self.atis_letters)
        facility = airport_data.get("tracon", "TRACON") if config.facility_type == "tracon" else airport_data.get("center", "ARTCC")
        facility_abbr = airport_data.get("tracon", "C90") if config.facility_type == "tracon" else airport_data.get("center", "ZAU")
        pos_list = self.positions.get(config.facility_type, self.positions["tracon"])
        position = rng.choice(pos_list)

        date_options = ["03/18/2026", "03/19/2026", "03/20/2026", "04/01/2026", "04/02/2026"]
        date_str = rng.choice(date_options)
        time_str = _format_zulu_time(sim_time)
        start_time = _format_zulu_time(sim_time)
        end_time = _format_zulu_time(sim_time + rng.uniform(60, 120))

        weather_summaries = {
            "vmc": "Clear, visibility 10+, light winds.",
            "imc": f"Ceiling {ceiling}, visibility {vis}, wind {wind}.",
            "marginal": f"Ceiling {ceiling}, visibility {vis}.",
        }
        weather_summary = weather_summaries.get(config.weather, "Weather nominal.")
        altimeter = f"{rng.randint(2942, 3030) / 100:.2f}"

        data = {
            "facility": facility,
            "facility_abbr": facility_abbr,
            "position": position,
            "date": date_str,
            "time": time_str,
            "start_time": start_time,
            "end_time": end_time,
            "atis_letter": atis_letter,
            "wind": wind,
            "vis": vis,
            "ceiling": ceiling,
            "rwy": rwy,
            "weather_summary": weather_summary,
            "altimeter": altimeter,
        }

        return self.render_template(template, data)

    # -------------------------------------------------------------------
    # Template selection
    # -------------------------------------------------------------------

    def _get_templates_for_event(self, event_type: str, voice: str) -> list:
        suffix = "_transcript" if voice == "transcript" else "_log"
        mapping = {
            "altitude_assignment":  "altitude" + suffix,
            "speed_assignment":     "speed" + suffix,
            "heading_assignment":   "heading" + suffix,
            "approach_clearance":   "approach_clearance" + suffix,
            "landing_clearance":    "landing_clearance_transcript",
            "takeoff_clearance":    "altitude" + suffix,
            "go_around":            "go_around_transcript",
            "hold_instruction":     "hold_instruction" + suffix,
            "emergency":            "emergency_transcript",
            "weather_update":       "weather_update_transcript",
            "traffic_advisory":     "traffic_advisory_transcript",
            "frequency_change":     "frequency_change_transcript",
            "position_report":      "filler" + suffix,
        }
        key = mapping.get(event_type, "filler" + suffix)
        templates = self.templates.get(key)
        if templates and isinstance(templates, list):
            return templates
        # Fallback
        return self.templates.get("filler" + suffix, ["[{callsign}]"])

    # -------------------------------------------------------------------
    # Main generation flow
    # -------------------------------------------------------------------

    def generate_trial(self, num_keys: int, num_updates: int,
                       condition: str, seed: int, **kwargs) -> dict:
        """Generate a single ATC narrative interference trial.

        Args:
            num_keys: number of aircraft to track (2-10)
            num_updates: target tracked-value mentions per aircraft (3-50)
            condition: "RI" or "PI" (accepted but ignored; both questions always generated)
            seed: random seed for full reproducibility
            **kwargs: optional config overrides

        Returns:
            dict matching the output JSON schema
        """
        rng = random.Random(seed)

        # Step 1: Auto-sample config, apply overrides
        config = self._auto_config(num_keys, num_updates, seed, rng)
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Ensure archetype is compatible after overrides
        if config.scenario_type in SCENARIO_ARCHETYPE_MAP:
            valid_archetypes = SCENARIO_ARCHETYPE_MAP[config.scenario_type]
            if config.scenario_archetype not in valid_archetypes:
                config.scenario_archetype = rng.choice(valid_archetypes)

        airport_code = config.airport
        airport_data = self.airports.get(airport_code, self.airports["ORD"])

        # Step 2: Generate aircraft callsigns and states
        used_callsigns: set = set()
        all_aircraft: List[str] = []
        ac_states: Dict[str, AircraftState] = {}

        for _ in range(num_keys):
            callsign = self._generate_callsign(rng, used_callsigns)
            used_callsigns.add(callsign)
            all_aircraft.append(callsign)
            ac_states[callsign] = self._init_aircraft_state(
                callsign, config.scenario_type, airport_code, rng
            )

        # Step 3: Assign tracked attributes
        tracked: Dict[str, str] = {}
        if config.attribute_mode == "same":
            for callsign in all_aircraft:
                tracked[callsign] = config.tracked_attribute
        else:
            # Mixed: cycle through attributes
            attr_cycle = ["altitude_ft", "speed_kias", "heading_degrees"]
            for i, callsign in enumerate(all_aircraft):
                tracked[callsign] = attr_cycle[i % len(attr_cycle)]

        # Step 4: Generate event sequence
        sim_time = 0.0  # minutes from scenario start
        event_log = []
        full_state_log = []
        entity_tracking = {f"{cs} / {tracked[cs]}": [] for cs in all_aircraft}
        mention_counts = {cs: 0 for cs in all_aircraft}
        total_target_mentions = num_keys * num_updates

        last_updated_callsign = None
        max_iterations = total_target_mentions * 8
        iteration = 0

        weights = self._get_scenario_event_weights(config.scenario_type)

        while sum(mention_counts.values()) < total_target_mentions and sim_time < 180.0:
            iteration += 1
            if iteration > max_iterations:
                break

            # Filter weights based on current state
            possible_weights = dict(weights)

            # Remove events that don't apply in current scenario/state
            if config.scenario_type in ("departure_push",):
                possible_weights.pop("approach_clearance", None)
                possible_weights.pop("landing_clearance", None)
            if config.scenario_type == "en_route_sector":
                possible_weights.pop("approach_clearance", None)
                possible_weights.pop("landing_clearance", None)
                possible_weights.pop("takeoff_clearance", None)
                possible_weights.pop("go_around", None)

            # Only allow takeoff if there's a ground aircraft
            ground_ac = [c for c in all_aircraft if ac_states[c].on_ground]
            if not ground_ac:
                possible_weights.pop("takeoff_clearance", None)

            # Only allow approach if someone is descending and not already cleared
            approachable = [c for c in all_aircraft
                            if not ac_states[c]._cleared_approach
                            and not ac_states[c].on_ground
                            and ac_states[c].flight_phase in ("descent", "cruise", "approach")]
            if not approachable:
                possible_weights.pop("approach_clearance", None)

            # Only allow landing clearance if someone is on approach/final
            on_final = [c for c in all_aircraft
                        if ac_states[c]._cleared_approach and not ac_states[c]._cleared_land]
            if not on_final:
                possible_weights.pop("landing_clearance", None)

            # Only allow go-around if someone is on final
            if not on_final:
                possible_weights.pop("go_around", None)

            # Remove zero-weight events
            possible_weights = {k: v for k, v in possible_weights.items() if v > 0}

            if not possible_weights:
                possible_weights = {"altitude_assignment": 1}

            event_type = self.weighted_random_choice(possible_weights, rng)

            # Select primary aircraft (not same as last, prefer fewer mentions)
            candidates = [c for c in all_aircraft if c != last_updated_callsign]
            if not candidates:
                candidates = list(all_aircraft)

            # Special event-type constraints
            if event_type == "takeoff_clearance" and ground_ac:
                filtered = [c for c in candidates if ac_states[c].on_ground]
                candidates = filtered if filtered else candidates
            elif event_type in ("approach_clearance",) and approachable:
                candidates = [c for c in candidates if c in approachable] or candidates
            elif event_type == "landing_clearance" and on_final:
                candidates = [c for c in candidates if c in on_final] or candidates
            elif event_type == "go_around" and on_final:
                candidates = [c for c in candidates if c in on_final] or candidates

            # Safety: if candidates is still empty fall back to all aircraft
            if not candidates:
                candidates = list(all_aircraft)

            # Prefer under-mentioned aircraft
            min_m = min(mention_counts[c] for c in candidates)
            urgent = [c for c in candidates if mention_counts[c] < 2]
            if urgent:
                candidates = urgent
            else:
                priority = [c for c in candidates if mention_counts[c] <= min_m + 1]
                if priority:
                    candidates = priority

            primary = rng.choice(candidates)

            # Execute event
            event_data = {}
            state_changes = []

            if event_type == "altitude_assignment":
                event_data, state_changes = self._handle_altitude_assignment(
                    primary, ac_states, sim_time, rng, config, tracked, airport_code)
            elif event_type == "speed_assignment":
                event_data, state_changes = self._handle_speed_assignment(
                    primary, ac_states, sim_time, rng, config, tracked, airport_code)
            elif event_type == "heading_assignment":
                event_data, state_changes = self._handle_heading_assignment(
                    primary, ac_states, sim_time, rng, config, tracked, airport_code)
            elif event_type == "approach_clearance":
                event_data, state_changes = self._handle_approach_clearance(
                    primary, ac_states, sim_time, rng, config, tracked, airport_code)
            elif event_type == "landing_clearance":
                event_data, state_changes = self._handle_landing_clearance(
                    primary, ac_states, sim_time, rng, config, tracked, airport_code)
            elif event_type == "takeoff_clearance":
                event_data, state_changes = self._handle_takeoff_clearance(
                    primary, ac_states, sim_time, rng, config, tracked, airport_code)
            elif event_type == "frequency_change":
                event_data, state_changes = self._handle_frequency_change(
                    primary, ac_states, sim_time, rng, config, tracked, airport_code)
            elif event_type == "traffic_advisory":
                event_data, state_changes = self._handle_traffic_advisory(
                    primary, ac_states, sim_time, rng, config, tracked, all_aircraft, airport_code)
            elif event_type == "position_report":
                event_data, state_changes = self._handle_position_report(
                    primary, ac_states, sim_time, rng, config, tracked, airport_code)
            elif event_type == "hold_instruction":
                event_data, state_changes = self._handle_hold_instruction(
                    primary, ac_states, sim_time, rng, config, tracked, airport_code)
            elif event_type == "go_around":
                event_data, state_changes = self._handle_go_around(
                    primary, ac_states, sim_time, rng, config, tracked, airport_code)
            elif event_type == "emergency":
                event_data, state_changes = self._handle_emergency(
                    primary, ac_states, sim_time, rng, config, tracked, airport_code)
            elif event_type == "weather_update":
                event_data, state_changes = self._handle_weather_update(
                    primary, ac_states, sim_time, rng, config, tracked, airport_code)
            else:
                # Fallback: altitude assignment
                event_data, state_changes = self._handle_altitude_assignment(
                    primary, ac_states, sim_time, rng, config, tracked, airport_code)

            # Record state changes
            for cs, attr, old_val, new_val, mentioned in state_changes:
                full_state_log.append({
                    "time": round(sim_time, 1),
                    "event": event_type,
                    "entity": cs,
                    "attribute": attr,
                    "old_value": str(old_val),
                    "new_value": str(new_val),
                    "mentioned_in_narrative": mentioned,
                })
                if mentioned and cs in tracked:
                    key = f"{cs} / {tracked[cs]}"
                    if key in entity_tracking:
                        if not entity_tracking[key] or entity_tracking[key][-1] != str(new_val):
                            entity_tracking[key].append(str(new_val))
                            mention_counts[cs] += 1

            event_log.append({
                "type": event_type,
                "time": round(sim_time, 1),
                "data": event_data,
            })

            last_updated_callsign = primary

            # Advance sim_time
            time_range = EVENT_TIME_ADVANCE.get(event_type, (1.5, 4.0))
            dt = rng.uniform(time_range[0], time_range[1])
            sim_time += dt
            sim_time = round(sim_time, 1)

        # Step 5: Enforce minimum 2 mentions per aircraft
        for force_pass in range(5):
            under_mentioned = [c for c in all_aircraft if mention_counts[c] < 2]
            if not under_mentioned:
                break
            for cs in under_mentioned:
                ac_states[cs].on_ground = False
                if tracked[cs] == "altitude_ft":
                    event_data, state_changes = self._handle_altitude_assignment(
                        cs, ac_states, sim_time, rng, config, tracked, airport_code)
                elif tracked[cs] == "speed_kias":
                    event_data, state_changes = self._handle_speed_assignment(
                        cs, ac_states, sim_time, rng, config, tracked, airport_code)
                else:
                    event_data, state_changes = self._handle_heading_assignment(
                        cs, ac_states, sim_time, rng, config, tracked, airport_code)

                for hn, attr, old_val, new_val, mentioned in state_changes:
                    full_state_log.append({
                        "time": round(sim_time, 1),
                        "event": "altitude_assignment",
                        "entity": hn,
                        "attribute": attr,
                        "old_value": str(old_val),
                        "new_value": str(new_val),
                        "mentioned_in_narrative": mentioned,
                    })
                    if mentioned and hn in tracked:
                        key = f"{hn} / {tracked[hn]}"
                        if key in entity_tracking:
                            if not entity_tracking[key] or entity_tracking[key][-1] != str(new_val):
                                entity_tracking[key].append(str(new_val))
                                mention_counts[hn] += 1
                event_log.append({
                    "type": "altitude_assignment",
                    "time": round(sim_time, 1),
                    "data": event_data,
                })
                sim_time += rng.uniform(1.0, 2.5)
                sim_time = round(sim_time, 1)

        # Step 6: Render narrative
        narrative_parts = []

        # Header
        voice = config.voice
        header_key = "header_transcript" if voice == "transcript" else "header_log"
        header_template = rng.choice(self.templates[header_key])
        header = self._render_header(header_template, airport_code, config, rng, 0.0)
        narrative_parts.append(header)
        narrative_parts.append("")

        filler_cfg = FILLER_CONFIG[config.filler_budget]

        # Build scenario-level defaults for variables that appear in many templates
        airport_data = self.airports.get(airport_code, self.airports.get("ORD", {}))
        default_runway = (airport_data.get("runways_landing_west", ["28L"]) +
                          airport_data.get("runways_landing_east", ["10L"]) +
                          airport_data.get("runways_landing_south", ["13L"]) +
                          airport_data.get("runways_landing_north", ["31L"]))[0]
        scenario_defaults = {
            "rwy": default_runway,
            "altimeter": f"{rng.randint(2960, 3015) / 100:.2f}",
            "atis_letter": rng.choice(self.atis_letters),
            "dir": rng.choice(["north", "south", "east", "west", "left", "right"]),
            "context": rng.choice(["traffic was moderate for the time of day",
                                   "weather holding at destination",
                                   "sequencing was straightforward",
                                   "coordination with adjacent sector was smooth"]),
            "separation_note": rng.choice(["adequate — 4.1 nm", "good — 5.2 nm", "minimal — 3.0 nm"]),
            "sector": rng.choice(["Chicago Center", "Departure", "Atlanta Approach", "LA Center"]),
            "ride": rng.choice(["smooth", "light chop", "occasional moderate"]),
            "freq": rng.choice(["119.35", "124.00", "120.15", "125.05"]),
            "facility": rng.choice(["Chicago Approach", "Atlanta Approach", "LA Center", "Departure"]),
            "eta": rng.choice(["1430Z", "1445Z", "1502Z", "1518Z"]),
            "wind": rng.choice(["270 at 12", "280 at 8", "260 at 15 gusts 22", "090 at 10"]),
            "traffic_count": rng.randint(8, 22),
            "dist": rng.randint(3, 25),
            "magnitude": rng.randint(12, 30),
            "callsign2": "N/A",  # fallback for weather PIREPs
            "preceding_type": rng.choice(["Boeing 737", "Airbus 320", "regional jet"]),
            "preceding_dist": rng.randint(2, 8),
            "hdg": rng.choice([180, 270, 90, 360, 150, 210]),
            "alt": rng.choice(["FL280", "FL240", "12000", "8000"]),
        }

        for i, event in enumerate(event_log):
            event_templates = self._get_templates_for_event(event["type"], voice)
            template = rng.choice(event_templates)
            # Merge scenario defaults under event data (event data takes priority)
            merged_data = {**scenario_defaults, **event["data"]}
            rendered = self.render_template(template, merged_data)
            narrative_parts.append(rendered)

            # Maybe insert filler
            if rng.random() < filler_cfg["insert_probability"]:
                n_sentences = filler_cfg["sentences_per_insertion"]
                if isinstance(n_sentences, tuple):
                    n_sentences = rng.randint(n_sentences[0], n_sentences[1])

                filler_key = "filler_transcript" if voice == "transcript" else "filler_log"
                filler_templates = rng.sample(
                    self.templates[filler_key],
                    min(n_sentences, len(self.templates[filler_key]))
                )
                for ft in filler_templates:
                    filler_cs = rng.choice(all_aircraft)
                    filler_rendered = self._render_filler(
                        ft, filler_cs, ac_states, rng, tracked,
                        all_aircraft, airport_code, event["time"]
                    )
                    narrative_parts.append(filler_rendered)

        # Step 7: Generate questions
        queried_idx = min(config.queried_aircraft_idx, len(all_aircraft) - 1)
        queried_cs = all_aircraft[queried_idx]
        queried_attr = tracked[queried_cs]
        tracking_key = f"{queried_cs} / {queried_attr}"
        values = entity_tracking[tracking_key]

        # Find a valid entity with distinct first/last values
        if len(values) < 2 or values[0] == values[-1]:
            for alt_cs in all_aircraft:
                alt_key = f"{alt_cs} / {tracked[alt_cs]}"
                alt_values = entity_tracking[alt_key]
                if len(alt_values) >= 2 and alt_values[0] != alt_values[-1]:
                    queried_cs = alt_cs
                    queried_attr = tracked[alt_cs]
                    tracking_key = alt_key
                    values = alt_values
                    break

        # Last resort: force a bonus event
        if len(values) < 2 or values[0] == values[-1]:
            best_cs = max(
                all_aircraft,
                key=lambda c: len(entity_tracking[f"{c} / {tracked[c]}"])
            )
            best_attr = tracked[best_cs]
            best_key = f"{best_cs} / {best_attr}"
            ac = ac_states[best_cs]

            if best_attr == "altitude_ft":
                new_alt = max(3000, ac.assigned_altitude_ft - 3000)
                ac.assigned_altitude_ft = new_alt
                ac.current_altitude_ft = new_alt
                new_formatted = format_altitude(new_alt)
            elif best_attr == "speed_kias":
                new_spd = max(150, (ac.assigned_speed_kias or 250) - 30)
                ac.assigned_speed_kias = new_spd
                new_formatted = str(new_spd)
            else:
                new_hdg = (ac.heading_degrees + 90) % 360
                ac.heading_degrees = new_hdg
                new_formatted = str(new_hdg)

            bonus_text = f" {best_cs} subsequently amended to {new_formatted}."
            narrative_parts.append(bonus_text)
            if not entity_tracking[best_key] or entity_tracking[best_key][-1] != new_formatted:
                entity_tracking[best_key].append(new_formatted)
                mention_counts[best_cs] += 1

            queried_cs = best_cs
            queried_attr = best_attr
            tracking_key = best_key
            values = entity_tracking[best_key]

        ri_answer = values[0] if values else "unknown"
        pi_answer = values[-1] if values else "unknown"

        # Human-readable attribute label for questions
        attr_labels = {
            "altitude_ft": "altitude",
            "speed_kias": "assigned airspeed",
            "heading_degrees": "assigned heading",
        }
        attr_label = attr_labels.get(queried_attr, queried_attr)

        RI_QUESTION_TEMPLATES = [
            f"What was {queried_cs}'s {attr_label} when first mentioned in this sector log?",
            f"What was {queried_cs}'s {attr_label} at the first reference in the transcript?",
            f"At {queried_cs}'s first appearance in the log, what was the {attr_label}?",
            f"What {attr_label} was assigned to {queried_cs} when first mentioned?",
        ]
        PI_QUESTION_TEMPLATES = [
            f"What was {queried_cs}'s {attr_label} at the most recent update?",
            f"In the last mention of {queried_cs}'s {attr_label}, what was the value?",
            f"What was {queried_cs}'s final recorded {attr_label}?",
            f"What is the last {attr_label} assigned to {queried_cs} in this log?",
        ]

        questions = {
            "RI": {
                "question": rng.choice(RI_QUESTION_TEMPLATES),
                "expected_answer": ri_answer,
                "target_entity": queried_cs,
                "target_attribute": queried_attr,
            },
            "PI": {
                "question": rng.choice(PI_QUESTION_TEMPLATES),
                "expected_answer": pi_answer,
                "target_entity": queried_cs,
                "target_attribute": queried_attr,
            },
        }

        # Finalize narrative
        final_parts = []
        for p in narrative_parts:
            if p == "":
                if final_parts and not final_parts[-1].endswith("\n"):
                    final_parts.append("\n")
            else:
                final_parts.append(p)
        narrative = " ".join(final_parts).replace(" \n ", "\n\n")

        return {
            "id": f"atc_{seed:06d}",
            "domain": "atc",
            "num_keys": num_keys,
            "num_updates": num_updates,
            "narrative": narrative,
            "questions": questions,
            "entity_tracking": entity_tracking,
            "full_state_log": full_state_log,
            "mention_counts": mention_counts,
            "config": {
                "seed": seed,
                "num_keys": num_keys,
                "num_updates": num_updates,
                "condition": condition,
                "tracked_attribute": config.tracked_attribute,
                "attribute_mode": config.attribute_mode,
                "airport": airport_code,
                "facility_type": config.facility_type,
                "scenario_type": config.scenario_type,
                "scenario_archetype": config.scenario_archetype,
                "weather": config.weather,
                "filler_budget": config.filler_budget,
                "voice": config.voice,
                "queried_aircraft_idx": config.queried_aircraft_idx,
                "aircraft": [
                    {
                        "callsign": cs,
                        "aircraft_type": ac_states[cs].aircraft_type,
                        "airline_radio": ac_states[cs].airline_radio,
                        "wake_category": ac_states[cs].wake_category,
                        "tracked_attribute": tracked[cs],
                    }
                    for cs in all_aircraft
                ],
            },
        }

    # -------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------

    def validate_trial(self, trial: dict) -> bool:
        narrative = trial["narrative"]
        entity_tracking = trial["entity_tracking"]
        questions = trial["questions"]

        # Check RI != PI
        ri_answer = questions["RI"]["expected_answer"]
        pi_answer = questions["PI"]["expected_answer"]
        assert ri_answer != pi_answer, (
            f"RI and PI answers must differ, got '{ri_answer}' for both"
        )

        # Check queried entity has at least 2 mentions
        target = questions["RI"]["target_entity"]
        target_attr = questions["RI"]["target_attribute"]
        tracking_key = f"{target} / {target_attr}"
        values = entity_tracking.get(tracking_key, [])
        assert len(values) >= 2, (
            f"Need at least 2 mentions for queried entity '{tracking_key}', got {len(values)}"
        )

        # Check RI answer is first value, PI answer is last
        assert values[0] == ri_answer, (
            f"RI answer should be first value '{values[0]}', got '{ri_answer}'"
        )
        assert values[-1] == pi_answer, (
            f"PI answer should be last value '{values[-1]}', got '{pi_answer}'"
        )

        return True

    # -------------------------------------------------------------------
    # Batch generation and saving
    # -------------------------------------------------------------------

    @staticmethod
    def _get_data_dir() -> Path:
        """Return data/narrative_interference/atc/ (project root relative)."""
        current = Path(__file__).resolve().parent
        for _ in range(10):
            if (current / ".git").exists() or (current / "CLAUDE.md").exists():
                break
            current = current.parent
        return current / "data" / "narrative_interference" / "atc"

    def generate_batch(
        self,
        key_levels: List[int] = None,
        update_levels: List[int] = None,
        trials_per_cell: int = 30,
        seed_start: int = 200000,
        **kwargs,
    ) -> dict:
        """Generate a full batch of trials across a grid.

        Args:
            key_levels: list of num_keys values (default: [2, 3, 5, 7, 10])
            update_levels: list of num_updates values (default: [1, 3, 5, 10, 20, 30, 40, 50])
            trials_per_cell: trials per (keys, updates, condition) cell
            seed_start: starting seed value
            **kwargs: config overrides passed to generate_trial()

        Returns:
            dict with 'metadata' and 'trials' keys
        """
        import time
        from datetime import datetime, timezone

        if key_levels is None:
            key_levels = [2, 3, 5, 7, 10]
        if update_levels is None:
            update_levels = [1, 3, 5, 10, 20, 30, 40, 50]

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        trials = []
        failures = []
        seed = seed_start
        total_cells = len(key_levels) * len(update_levels) * 2
        cell_idx = 0
        t0 = time.time()

        for nk in key_levels:
            for nu in update_levels:
                for cond in ["RI", "PI"]:
                    cell_idx += 1
                    cell_ok = 0
                    for _ in range(trials_per_cell):
                        try:
                            trial = self.generate_trial(nk, nu, cond, seed, **kwargs)
                            self.validate_trial(trial)
                            trials.append(trial)
                            cell_ok += 1
                        except Exception as e:
                            failures.append({
                                "seed": seed, "nk": nk, "nu": nu,
                                "cond": cond, "error": str(e)[:100],
                            })
                        seed += 1

                    elapsed = time.time() - t0
                    rate = len(trials) / max(elapsed, 0.1)
                    print(f"  [{cell_idx}/{total_cells}] keys={nk:>2} updates={nu:>3} "
                          f"{cond}: {cell_ok}/{trials_per_cell} ok ({rate:.0f}/sec)")

        elapsed = time.time() - t0
        print(f"\nGenerated {len(trials)} trials in {elapsed:.1f}s "
              f"({len(failures)} failures)")

        return {
            "metadata": {
                "domain": self.DOMAIN,
                "generator": self.__class__.__name__,
                "total_trials": len(trials),
                "failures": len(failures),
                "timestamp": ts,
                "grid": {
                    "num_keys": key_levels,
                    "num_updates": update_levels,
                },
                "trials_per_cell_per_condition": trials_per_cell,
                "seed_range": f"{seed_start}-{seed - 1}",
                "overrides": kwargs if kwargs else "all_random",
            },
            "trials": trials,
        }

    def save_batch(self, batch: dict, tag: str = None) -> Path:
        """Save a generated batch to data/narrative_interference/atc/.

        Args:
            batch: output from generate_batch()
            tag: optional short tag (e.g., 'altitude_same', 'v1')

        Returns:
            Path to saved file
        """
        ts = batch["metadata"]["timestamp"]
        parts = [self.DOMAIN]
        if tag:
            parts.append(tag)
        parts.append(ts)
        filename = "_".join(parts) + ".json"

        out_dir = self._get_data_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename

        with open(out_path, "w") as f:
            json.dump(batch, f, indent=2, ensure_ascii=False)

        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"Saved to: {out_path} ({size_mb:.1f} MB)")
        return out_path
