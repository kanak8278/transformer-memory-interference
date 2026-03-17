"""Hospital ICU narrative interference trial generator.

Implements the full generation flow:
- Patient selection with diagnosis and trajectory assignment
- Event sequence generation across an ICU shift
- State tracking for all patients (vital signs and lab values)
- Narrative rendering with clinical templates and filler
- Question generation for both RI and PI conditions
"""

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from ..base import NarrativeTrialGenerator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Core trackable attributes — safe candidates for the tracked attribute
CORE_TRACKED_ATTRIBUTES = [
    "heart_rate", "systolic_bp", "map", "spo2", "temperature_c",
    "lactate", "creatinine", "potassium", "hemoglobin", "wbc",
    "glucose", "troponin", "norepinephrine_dose", "fio2", "peep",
    "gcs", "ph", "paco2", "bnp", "lipase", "bun",
]

# Diagnosis-to-attribute affinity: which attributes change meaningfully for each diagnosis
DIAGNOSIS_ATTR_AFFINITY = {
    "sepsis":              ["heart_rate", "lactate", "map", "creatinine", "norepinephrine_dose", "wbc", "temperature_c"],
    "dka":                 ["glucose", "potassium", "ph", "heart_rate"],
    "ards":                ["fio2", "spo2", "peep", "heart_rate"],
    "gi_bleed":            ["hemoglobin", "heart_rate", "systolic_bp", "map"],
    "aki":                 ["creatinine", "potassium", "bun"],
    "copd_exacerbation":   ["paco2", "spo2", "ph", "heart_rate"],
    "stemi":               ["troponin", "heart_rate", "bnp", "systolic_bp"],
    "hemorrhagic_stroke":  ["gcs", "systolic_bp", "map"],
    "pancreatitis":        ["lipase", "wbc", "heart_rate", "temperature_c"],
    "post_cardiac_surgery":["troponin", "hemoglobin", "heart_rate"],
}

DIAGNOSIS_POOL = list(DIAGNOSIS_ATTR_AFFINITY.keys())

# Event types and their approximate time-advance windows (hours)
EVENT_TIME_ADVANCE = {
    "vitals_handoff":          (0.5, 1.5),
    "vitals_progress":         (1.0, 2.0),
    "lab_result":              (4.0, 8.0),
    "medication_change":       (1.0, 3.0),
    "ventilator_change":       (2.0, 4.0),
    "clinical_deterioration":  (0.5, 2.0),
    "clinical_improvement":    (1.0, 3.0),
    "shift_handoff":           (6.0, 8.0),
}

# Default event weights for a typical ICU shift
DEFAULT_EVENT_WEIGHTS = {
    "vitals_handoff":         20,
    "vitals_progress":        25,
    "lab_result":             20,
    "medication_change":      15,
    "ventilator_change":       5,
    "clinical_deterioration":  8,
    "clinical_improvement":    7,
}

FILLER_CONFIG = {
    "minimal": {"insert_probability": 0.05, "sentences_per_insertion": 1},
    "light":   {"insert_probability": 0.20, "sentences_per_insertion": 1},
    "medium":  {"insert_probability": 0.40, "sentences_per_insertion": (1, 2)},
    "heavy":   {"insert_probability": 0.60, "sentences_per_insertion": (1, 3)},
}

# Physiological bounds for clamping
PHYS_BOUNDS = {
    "heart_rate":          (30, 220),
    "systolic_bp":         (50, 250),
    "diastolic_bp":        (25, 150),
    "map":                 (30, 160),
    "respiratory_rate":    (4, 60),
    "spo2":                (50, 100),
    "temperature_c":       (33.0, 42.0),
    "potassium":           (1.5, 8.5),
    "sodium":              (110, 175),
    "creatinine":          (0.3, 20.0),
    "lactate":             (0.3, 25.0),
    "hemoglobin":          (2.0, 20.0),
    "wbc":                 (0.5, 80.0),
    "platelets":           (5, 800),
    "glucose":             (30, 1500),
    "troponin":            (0.0, 100.0),
    "inr":                 (0.5, 15.0),
    "bilirubin":           (0.1, 40.0),
    "gcs":                 (3, 15),
    "rass":                (-5, 4),
    "norepinephrine_dose": (0.0, 5.0),
    "fio2":                (21, 100),
    "peep":                (0, 30),
    "urine_output_ml_hr":  (0, 500),
    "ph":                  (6.5, 7.65),
    "hco3":                (1, 50),
    "paco2":               (15, 150),
    "bnp":                 (5, 10000),
    "lipase":              (10, 20000),
    "pf_ratio":            (20, 600),
    "anion_gap":           (3, 60),
    "bun":                 (3, 300),
    "chest_tube_output":   (0, 1000),
}

# Precision formatting for each attribute (decimal places)
ATTR_PRECISION = {
    "heart_rate": 0, "systolic_bp": 0, "diastolic_bp": 0, "map": 0,
    "respiratory_rate": 0, "spo2": 0, "temperature_c": 1,
    "potassium": 1, "sodium": 0, "creatinine": 1, "lactate": 1,
    "hemoglobin": 1, "wbc": 1, "platelets": 0, "glucose": 0,
    "troponin": 2, "inr": 1, "bilirubin": 1, "gcs": 0, "rass": 0,
    "norepinephrine_dose": 2, "fio2": 0, "peep": 0, "urine_output_ml_hr": 0,
    "ph": 2, "hco3": 0, "paco2": 0, "bnp": 0, "lipase": 0,
    "pf_ratio": 0, "anion_gap": 0, "bun": 0, "chest_tube_output": 0,
}


# ---------------------------------------------------------------------------
# Helper: format attribute value as canonical string
# ---------------------------------------------------------------------------

def format_attr_value(attr: str, value: float) -> str:
    """Format an attribute value to its canonical string representation."""
    prec = ATTR_PRECISION.get(attr, 1)
    if prec == 0:
        return str(int(round(value)))
    else:
        return f"{value:.{prec}f}"


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PatientState:
    patient_id: str          # e.g., "Bed 3"
    name: str                # e.g., "Mrs. Liu"
    age: int
    sex: str                 # "M" or "F"
    weight_kg: float
    primary_diagnosis: str   # key into diagnoses dict
    icu_day: int
    trajectory: str          # "improving", "worsening", "fluctuating"
    trajectory_step: int     # 0-based index into trajectory arrays

    # Vital signs
    heart_rate: float = 90.0
    systolic_bp: float = 120.0
    diastolic_bp: float = 70.0
    map: float = 87.0
    respiratory_rate: float = 18.0
    spo2: float = 97.0
    temperature_c: float = 37.0

    # Labs
    potassium: float = 4.0
    sodium: float = 140.0
    creatinine: float = 1.0
    lactate: float = 1.0
    hemoglobin: float = 12.0
    wbc: float = 8.0
    platelets: float = 200.0
    glucose: float = 120.0
    troponin: float = 0.01
    inr: float = 1.1
    bilirubin: float = 0.8
    gcs: float = 15.0
    rass: float = 0.0
    norepinephrine_dose: float = 0.0
    fio2: float = 21.0
    peep: float = 5.0
    urine_output_ml_hr: float = 50.0

    # Derived / extended labs
    ph: float = 7.40
    hco3: float = 24.0
    paco2: float = 40.0
    bnp: float = 100.0
    lipase: float = 50.0
    pf_ratio: float = 400.0
    anion_gap: float = 12.0
    bun: float = 15.0
    chest_tube_output: float = 0.0

    on_vasopressor: bool = False
    on_ventilator: bool = False
    alive: bool = True
    transferred: bool = False

    def get_attribute(self, attr: str) -> float:
        return getattr(self, attr, 0.0)

    def set_attribute(self, attr: str, value: float):
        setattr(self, attr, value)

    def format_attribute(self, attr: str) -> str:
        """Return canonical string for narrative text and entity_tracking."""
        val = self.get_attribute(attr)
        return format_attr_value(attr, val)


@dataclass
class ICUTrialConfig:
    num_keys: int
    num_updates: int
    seed: int
    tracked_attribute: str
    attribute_mode: str          # "same" or "mixed"
    diagnosis_mode: str          # "same" or "mixed"
    diagnoses: List[str]         # one per patient
    icu_type: str
    shift_start: str             # "07:00", "19:00", or "23:00"
    filler_budget: str
    voice: str                   # "clinical" or "narrative"
    shift_archetype: str         # "stable", "busy", "deteriorating", "recovering"
    queried_patient_idx: int
    trajectory_mix: str          # "all_improving", "all_worsening", "mixed"


# ---------------------------------------------------------------------------
# Main Generator
# ---------------------------------------------------------------------------

class ICUTrialGenerator(NarrativeTrialGenerator):

    def __init__(self):
        data_dir = Path(__file__).parent / "data"
        with open(data_dir / "icu_templates.json") as f:
            self.templates = json.load(f)
        with open(data_dir / "icu_data.json") as f:
            self.data = json.load(f)

        self.diagnoses = self.data["diagnoses"]
        self.attr_labels = self.data["attr_labels"]
        self.attr_units = self.data["attr_units"]
        self.filler_pools = self.templates["filler_variable_pools"]

    # -------------------------------------------------------------------
    # Auto-config
    # -------------------------------------------------------------------

    def _auto_config(self, num_keys: int, num_updates: int,
                     seed: int, rng: random.Random) -> ICUTrialConfig:
        tracked_attribute = rng.choice(CORE_TRACKED_ATTRIBUTES)
        attribute_mode = rng.choice(["same", "mixed"])
        diagnosis_mode = rng.choice(["same", "mixed"])

        if diagnosis_mode == "same":
            base_dx = rng.choice(DIAGNOSIS_POOL)
            diagnoses = [base_dx] * num_keys
        else:
            diagnoses = [rng.choice(DIAGNOSIS_POOL) for _ in range(num_keys)]

        icu_type = rng.choice(list(self.data["icu_types"].keys()))
        shift_start = rng.choice(["07:00", "19:00", "23:00"])
        filler_budget = rng.choice(["minimal", "light", "medium", "heavy"])
        voice = rng.choice(["clinical", "narrative"])
        shift_archetype = rng.choice(["stable", "busy", "deteriorating", "recovering"])
        queried_patient_idx = rng.randint(0, num_keys - 1)
        trajectory_mix = rng.choice(["all_improving", "all_worsening", "mixed"])

        return ICUTrialConfig(
            num_keys=num_keys,
            num_updates=num_updates,
            seed=seed,
            tracked_attribute=tracked_attribute,
            attribute_mode=attribute_mode,
            diagnosis_mode=diagnosis_mode,
            diagnoses=diagnoses,
            icu_type=icu_type,
            shift_start=shift_start,
            filler_budget=filler_budget,
            voice=voice,
            shift_archetype=shift_archetype,
            queried_patient_idx=queried_patient_idx,
            trajectory_mix=trajectory_mix,
        )

    # -------------------------------------------------------------------
    # Patient state initialization
    # -------------------------------------------------------------------

    def _assign_trajectory(self, diagnosis: str, trajectory_mix: str,
                           rng: random.Random) -> str:
        if trajectory_mix == "all_improving":
            return "improving"
        elif trajectory_mix == "all_worsening":
            return "worsening"
        elif trajectory_mix == "mixed":
            # 60% improving, 30% worsening, 10% fluctuating
            r = rng.random()
            if r < 0.60:
                return "improving"
            elif r < 0.90:
                return "worsening"
            else:
                return "fluctuating"
        return rng.choice(["improving", "worsening"])

    def _init_patient_state(self, patient_id: str, name: str, age: int,
                            sex: str, diagnosis_key: str,
                            trajectory: str, rng: random.Random) -> PatientState:
        dx_data = self.diagnoses[diagnosis_key]
        iv = dx_data.get("initial_vitals", {})

        def sample_iv(key: str, default_lo: float, default_hi: float) -> float:
            rng_range = iv.get(key, [default_lo, default_hi])
            return rng.uniform(rng_range[0], rng_range[1])

        hr = sample_iv("hr", 70, 100)
        sbp = sample_iv("sbp", 110, 140)
        dbp = sbp * rng.uniform(0.55, 0.65)
        map_val = (sbp + 2 * dbp) / 3.0
        temp = sample_iv("temperature_c", 36.8, 37.5)
        spo2 = sample_iv("spo2", 94, 99)
        rr = sample_iv("respiratory_rate", 14, 20)

        # Determine on_ventilator based on probability
        vent_prob = dx_data.get("vent_probability", 0.1)
        on_vent = rng.random() < vent_prob
        on_vaso = rng.random() < 0.3  # generic starting vasopressor probability

        weight_kg = rng.uniform(55, 110)
        icu_day = rng.randint(1, 7)

        ps = PatientState(
            patient_id=patient_id,
            name=name,
            age=age,
            sex=sex,
            weight_kg=round(weight_kg, 1),
            primary_diagnosis=diagnosis_key,
            icu_day=icu_day,
            trajectory=trajectory,
            trajectory_step=0,
            heart_rate=round(hr),
            systolic_bp=round(sbp),
            diastolic_bp=round(dbp),
            map=round(map_val),
            respiratory_rate=round(rr),
            spo2=round(spo2),
            temperature_c=round(temp, 1),
            on_ventilator=on_vent,
            on_vasopressor=on_vaso,
        )

        # Seed disease-specific labs from trajectory start
        traj_key = f"trajectory_{trajectory}" if trajectory != "fluctuating" else "trajectory_improving"
        traj = dx_data.get(traj_key, {})
        for attr, values in traj.items():
            if hasattr(ps, attr) and values:
                setattr(ps, attr, float(values[0]))

        return ps

    # -------------------------------------------------------------------
    # Trajectory advancement
    # -------------------------------------------------------------------

    def _advance_trajectory(self, ps: PatientState, rng: random.Random):
        """Advance patient one step along trajectory, with Gaussian noise."""
        dx_data = self.diagnoses[ps.primary_diagnosis]

        if ps.trajectory == "fluctuating":
            # Fluctuating: randomly pick improving or worsening step
            traj_key = rng.choice(["trajectory_improving", "trajectory_worsening"])
        else:
            traj_key = f"trajectory_{ps.trajectory}"

        traj = dx_data.get(traj_key, {})
        if not traj:
            return

        max_step = max(len(v) for v in traj.values()) - 1
        next_step = min(ps.trajectory_step + 1, max_step)

        for attr, values in traj.items():
            if not hasattr(ps, attr):
                continue
            if next_step >= len(values):
                continue

            target = float(values[next_step])
            current = getattr(ps, attr)

            # Gaussian noise: sigma = 3% of the range of this trajectory
            lo = min(values)
            hi = max(values)
            val_range = max(abs(hi - lo), 0.01)
            sigma = val_range * 0.03
            noisy = target + rng.gauss(0, sigma)

            # Clamp to physiological bounds
            bounds = PHYS_BOUNDS.get(attr, (float('-inf'), float('inf')))
            noisy = clamp(noisy, bounds[0], bounds[1])

            # Round to appropriate precision
            prec = ATTR_PRECISION.get(attr, 1)
            if prec == 0:
                noisy = round(noisy)
            else:
                noisy = round(noisy, prec)

            setattr(ps, attr, noisy)

        ps.trajectory_step = next_step
        self._apply_correlations(ps, rng)

    def _apply_correlations(self, ps: PatientState, rng: random.Random):
        """Enforce physiological correlations between attributes."""
        # Fever → tachycardia
        if ps.temperature_c > 38.5:
            hr_boost = rng.uniform(8, 15)
            ps.heart_rate = min(PHYS_BOUNDS["heart_rate"][1],
                                ps.heart_rate + hr_boost)

        # Hypotension → tachycardia
        if ps.map < 65:
            hr_boost = rng.uniform(15, 30)
            ps.heart_rate = min(PHYS_BOUNDS["heart_rate"][1],
                                ps.heart_rate + hr_boost)

        # Vasopressor → MAP response
        if ps.on_vasopressor and ps.norepinephrine_dose > 0:
            ps.map = clamp(65.0 + ps.norepinephrine_dose * 30,
                           PHYS_BOUNDS["map"][0], PHYS_BOUNDS["map"][1])

        # Recompute derived values
        # MAP from BP (if SBP/DBP changed)
        ps.map = clamp(round((ps.systolic_bp + 2 * ps.diastolic_bp) / 3.0),
                       PHYS_BOUNDS["map"][0], PHYS_BOUNDS["map"][1])

        # Clamp all tracked attributes
        for attr, (lo, hi) in PHYS_BOUNDS.items():
            if hasattr(ps, attr):
                val = getattr(ps, attr)
                clamped = clamp(val, lo, hi)
                prec = ATTR_PRECISION.get(attr, 1)
                if prec == 0:
                    clamped = round(clamped)
                else:
                    clamped = round(clamped, prec)
                setattr(ps, attr, clamped)

    # -------------------------------------------------------------------
    # Time formatting
    # -------------------------------------------------------------------

    def _minutes_to_hhmm(self, start_hhmm: str, offset_hours: float) -> str:
        """Add offset_hours to a HH:MM string, return new HH:MM (modulo 24h)."""
        h, m = int(start_hhmm[:2]), int(start_hhmm[3:])
        total_min = h * 60 + m + int(offset_hours * 60)
        total_min %= (24 * 60)
        return f"{total_min // 60:02d}:{total_min % 60:02d}"

    # -------------------------------------------------------------------
    # Event handlers
    # -------------------------------------------------------------------

    def _handle_vitals_check(self, patient: PatientState,
                              tracked_attr: str, shift_time_hours: float,
                              shift_start: str, rng: random.Random,
                              config: ICUTrialConfig) -> Tuple[dict, list]:
        """Generate a vitals check event (handoff or progress style)."""
        old_val = patient.format_attribute(tracked_attr)
        self._advance_trajectory(patient, rng)
        new_val = patient.format_attribute(tracked_attr)

        # Force a change in the tracked attribute if it didn't change
        if new_val == old_val:
            attr_val = patient.get_attribute(tracked_attr)
            bounds = PHYS_BOUNDS.get(tracked_attr, (0, 9999))
            delta = rng.uniform(0.5, 3.0)
            if rng.random() < 0.5:
                delta = -delta
            new_attr_val = clamp(attr_val + delta, bounds[0], bounds[1])
            prec = ATTR_PRECISION.get(tracked_attr, 1)
            if prec == 0:
                new_attr_val = round(new_attr_val)
            else:
                new_attr_val = round(new_attr_val, prec)
            patient.set_attribute(tracked_attr, new_attr_val)
            new_val = patient.format_attribute(tracked_attr)

        wall_time = self._minutes_to_hhmm(shift_start, shift_time_hours)

        event_data = {
            "patient_name": patient.name,
            "bed": patient.patient_id,
            "age": patient.age,
            "sex": patient.sex,
            "diagnosis": self.diagnoses[patient.primary_diagnosis]["full_name"],
            "attr_label": self.attr_labels.get(tracked_attr, tracked_attr.replace("_", " ")),
            "attr_value": new_val,
            "hr": str(int(patient.heart_rate)),
            "sbp": str(int(patient.systolic_bp)),
            "dbp": str(int(patient.diastolic_bp)),
            "map": str(int(patient.map)),
            "temp": str(patient.temperature_c),
            "spo2": str(int(patient.spo2)),
            "time": wall_time,
            "note": rng.choice(self.filler_pools["clinical_notes"]),
            "clinical_note": rng.choice(self.filler_pools["clinical_notes"]),
        }

        state_changes = [(patient.name, tracked_attr, old_val, new_val, True)]
        return event_data, state_changes

    def _handle_lab_result(self, patient: PatientState,
                           tracked_attr: str, shift_time_hours: float,
                           shift_start: str, rng: random.Random,
                           config: ICUTrialConfig) -> Tuple[dict, list]:
        """Generate a lab result event."""
        old_val = patient.format_attribute(tracked_attr)
        self._advance_trajectory(patient, rng)
        new_val = patient.format_attribute(tracked_attr)

        # Force change if stale
        if new_val == old_val:
            attr_val = patient.get_attribute(tracked_attr)
            bounds = PHYS_BOUNDS.get(tracked_attr, (0, 9999))
            delta = rng.uniform(1.0, 5.0)
            if rng.random() < 0.5:
                delta = -delta
            new_attr_val = clamp(attr_val + delta, bounds[0], bounds[1])
            prec = ATTR_PRECISION.get(tracked_attr, 1)
            if prec == 0:
                new_attr_val = round(new_attr_val)
            else:
                new_attr_val = round(new_attr_val, prec)
            patient.set_attribute(tracked_attr, new_attr_val)
            new_val = patient.format_attribute(tracked_attr)

        wall_time = self._minutes_to_hhmm(shift_start, shift_time_hours)

        event_data = {
            "patient_name": patient.name,
            "attr_label": self.attr_labels.get(tracked_attr, tracked_attr.replace("_", " ")),
            "attr_value": new_val,
            "units": self.attr_units.get(tracked_attr, ""),
            "lab_trend": rng.choice(self.filler_pools["lab_trends"]),
            "comparison": rng.choice(self.filler_pools["comparisons"]),
            "clinical_significance": rng.choice(self.filler_pools["clinical_significances"]),
            "time": wall_time,
        }

        state_changes = [(patient.name, tracked_attr, old_val, new_val, True)]
        return event_data, state_changes

    def _handle_medication_change(self, patient: PatientState,
                                  tracked_attr: str, shift_time_hours: float,
                                  shift_start: str, rng: random.Random,
                                  config: ICUTrialConfig) -> Tuple[dict, list]:
        """Generate a medication change event."""
        old_val = patient.format_attribute(tracked_attr)
        self._advance_trajectory(patient, rng)
        new_val = patient.format_attribute(tracked_attr)

        if new_val == old_val:
            attr_val = patient.get_attribute(tracked_attr)
            bounds = PHYS_BOUNDS.get(tracked_attr, (0, 9999))
            delta = rng.uniform(0.5, 2.0)
            if rng.random() < 0.5:
                delta = -delta
            new_attr_val = clamp(attr_val + delta, bounds[0], bounds[1])
            prec = ATTR_PRECISION.get(tracked_attr, 1)
            if prec == 0:
                new_attr_val = round(new_attr_val)
            else:
                new_attr_val = round(new_attr_val, prec)
            patient.set_attribute(tracked_attr, new_attr_val)
            new_val = patient.format_attribute(tracked_attr)

        dx_data = self.diagnoses[patient.primary_diagnosis]
        medications = dx_data.get("typical_medications", ["IV_fluids"])
        medication = rng.choice(medications).replace("_", " ")

        direction = rng.choice(["up", "down"])
        base_dose = rng.uniform(0.5, 2.0)
        new_dose = round(base_dose, 2)
        units = rng.choice(["mg/hr", "mcg/kg/min", "mL/hr", "mg", "units/hr"])
        physician_full = rng.choice(self.data["staff_names"]["attendings"])
        # Strip "Dr. " prefix so templates that say "Dr. {physician}" don't double up
        physician = physician_full.replace("Dr. ", "").strip()

        event_data = {
            "patient_name": patient.name,
            "physician": physician,
            "medication": medication,
            "direction": direction,
            "new_dose": str(new_dose),
            "units": units,
            "attr_label": self.attr_labels.get(tracked_attr, tracked_attr.replace("_", " ")),
            "attr_value": new_val,
        }

        state_changes = [(patient.name, tracked_attr, old_val, new_val, True)]
        return event_data, state_changes

    def _handle_ventilator_change(self, patient: PatientState,
                                  tracked_attr: str, shift_time_hours: float,
                                  shift_start: str, rng: random.Random,
                                  config: ICUTrialConfig) -> Tuple[dict, list]:
        """Generate a ventilator change event."""
        old_val = patient.format_attribute(tracked_attr)
        old_fio2 = int(patient.fio2)
        self._advance_trajectory(patient, rng)
        new_fio2 = int(patient.fio2)
        new_val = patient.format_attribute(tracked_attr)

        if new_val == old_val:
            attr_val = patient.get_attribute(tracked_attr)
            bounds = PHYS_BOUNDS.get(tracked_attr, (0, 9999))
            delta = rng.uniform(2.0, 8.0) * (1 if patient.trajectory == "worsening" else -1)
            new_attr_val = clamp(attr_val + delta, bounds[0], bounds[1])
            prec = ATTR_PRECISION.get(tracked_attr, 1)
            if prec == 0:
                new_attr_val = round(new_attr_val)
            else:
                new_attr_val = round(new_attr_val, prec)
            patient.set_attribute(tracked_attr, new_attr_val)
            new_val = patient.format_attribute(tracked_attr)

        # Ensure fio2 values differ
        if new_fio2 == old_fio2:
            new_fio2 = clamp(old_fio2 - rng.randint(5, 15), 21, 100) if patient.trajectory == "improving" else clamp(old_fio2 + rng.randint(5, 15), 21, 100)
            patient.fio2 = float(new_fio2)

        event_data = {
            "patient_name": patient.name,
            "old_fio2": str(old_fio2),
            "new_fio2": str(new_fio2),
            "peep": str(int(patient.peep)),
            "attr_label": self.attr_labels.get(tracked_attr, tracked_attr.replace("_", " ")),
            "attr_value": new_val,
        }

        state_changes = [(patient.name, tracked_attr, old_val, new_val, True)]
        return event_data, state_changes

    def _handle_clinical_event(self, patient: PatientState,
                               tracked_attr: str, shift_time_hours: float,
                               shift_start: str, rng: random.Random,
                               config: ICUTrialConfig,
                               event_type: str) -> Tuple[dict, list]:
        """Generate a clinical deterioration or improvement event."""
        old_val = patient.format_attribute(tracked_attr)
        self._advance_trajectory(patient, rng)
        new_val = patient.format_attribute(tracked_attr)

        wall_time = self._minutes_to_hhmm(shift_start, shift_time_hours)

        # Force meaningful change
        if new_val == old_val:
            attr_val = patient.get_attribute(tracked_attr)
            bounds = PHYS_BOUNDS.get(tracked_attr, (0, 9999))
            # deterioration = worse, improvement = better
            # We approximate: higher value = worse for most vitals/labs
            if event_type == "clinical_deterioration":
                delta = rng.uniform(5.0, 20.0)
            else:
                delta = -rng.uniform(5.0, 15.0)
            new_attr_val = clamp(attr_val + delta, bounds[0], bounds[1])
            prec = ATTR_PRECISION.get(tracked_attr, 1)
            if prec == 0:
                new_attr_val = round(new_attr_val)
            else:
                new_attr_val = round(new_attr_val, prec)
            patient.set_attribute(tracked_attr, new_attr_val)
            new_val = patient.format_attribute(tracked_attr)

        event_data = {
            "patient_name": patient.name,
            "time": wall_time,
            "attr_label": self.attr_labels.get(tracked_attr, tracked_attr.replace("_", " ")),
            "attr_value": new_val,
            "clinical_response": rng.choice(self.filler_pools["clinical_responses"]),
            "milestone_note": rng.choice(self.filler_pools["milestone_notes"]),
        }

        state_changes = [(patient.name, tracked_attr, old_val, new_val, True)]
        return event_data, state_changes

    def _handle_shift_handoff(self, patient: PatientState,
                              tracked_attr: str, shift_time_hours: float,
                              shift_start: str, rng: random.Random,
                              config: ICUTrialConfig) -> Tuple[dict, list]:
        """Generate a shift handoff summary event."""
        old_val = patient.format_attribute(tracked_attr)
        self._advance_trajectory(patient, rng)
        new_val = patient.format_attribute(tracked_attr)

        if new_val == old_val:
            attr_val = patient.get_attribute(tracked_attr)
            bounds = PHYS_BOUNDS.get(tracked_attr, (0, 9999))
            delta = rng.uniform(1.0, 5.0) * (1 if rng.random() < 0.5 else -1)
            new_attr_val = clamp(attr_val + delta, bounds[0], bounds[1])
            prec = ATTR_PRECISION.get(tracked_attr, 1)
            if prec == 0:
                new_attr_val = round(new_attr_val)
            else:
                new_attr_val = round(new_attr_val, prec)
            patient.set_attribute(tracked_attr, new_attr_val)
            new_val = patient.format_attribute(tracked_attr)

        event_data = {
            "patient_name": patient.name,
            "bed": patient.patient_id,
            "age": patient.age,
            "diagnosis": self.diagnoses[patient.primary_diagnosis]["full_name"],
            "icu_day": patient.icu_day,
            "overnight_summary": rng.choice(self.filler_pools["overnight_summaries"]),
            "attr_label": self.attr_labels.get(tracked_attr, tracked_attr.replace("_", " ")),
            "attr_value": new_val,
            "plan": rng.choice(self.filler_pools["plan_options"]),
        }

        state_changes = [(patient.name, tracked_attr, old_val, new_val, True)]
        return event_data, state_changes

    # -------------------------------------------------------------------
    # Filler rendering
    # -------------------------------------------------------------------

    def _render_filler(self, template: str, patient: PatientState,
                       rng: random.Random) -> str:
        import datetime
        data = {
            "patient_name": patient.name,
            "hours": rng.choice([2, 4, 6, 8, 12]),
            "urine_ml": rng.randint(50, 800),
            "urine_assessment": rng.choice(self.filler_pools["urine_assessments"]),
            "cxr_finding": rng.choice(self.filler_pools["cxr_findings"]),
            "family_note": rng.choice(self.filler_pools["family_notes"]),
            "prophylaxis": rng.choice(self.filler_pools["prophylaxis_options"]),
            "culture_date": rng.choice(["yesterday", "two days ago", "this morning"]),
            "culture_result": rng.choice(self.filler_pools["culture_results"]),
            "pain_score": rng.randint(0, 10),
            "pain_management": rng.choice(self.filler_pools["pain_managements"]),
            "nutrition_route": rng.choice(self.filler_pools["nutrition_routes"]),
            "calorie_goal": rng.choice([1400, 1600, 1800, 2000, 2200]),
            "overnight_interventions": rng.choice(self.filler_pools["overnight_interventions"]),
            "code_status": rng.choice(self.filler_pools["code_statuses"]),
            "telemetry_finding": rng.choice(self.filler_pools["telemetry_findings"]),
            "glucose_val": rng.randint(80, 250),
            "insulin_note": rng.choice(["insulin drip continued", "sliding scale ordered", "no change to insulin regimen"]),
            "glucose_target": rng.choice(["140-180", "110-150", "80-110"]),
            "intake_ml": rng.randint(500, 3000),
            "output_ml": rng.randint(500, 2500),
            "balance_note": rng.choice(["positive balance — consider diuresis", "negative balance — reassess", "roughly even"]),
        }
        return self.render_template(template, data)

    # -------------------------------------------------------------------
    # Header rendering
    # -------------------------------------------------------------------

    def _render_header(self, template: str, config: ICUTrialConfig,
                       rng: random.Random, num_patients: int) -> str:
        import datetime
        unit_name = self.data["icu_types"].get(config.icu_type, "Medical ICU")
        hospital = rng.choice(self.data["hospitals"])
        attending = rng.choice(self.data["staff_names"]["attendings"])

        # Simple date mock
        months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
        month = rng.choice(months)
        day = rng.randint(1, 28)
        year = 2025

        data = {
            "hospital": hospital,
            "unit_name": unit_name,
            "date": f"{month} {day}, {year}",
            "time": config.shift_start,
            "num_patients": num_patients,
            "attending": attending.replace("Dr. ", ""),
            "weather": rng.choice(self.data["weather_options"]),
        }
        return self.render_template(template, data)

    # -------------------------------------------------------------------
    # Template selection
    # -------------------------------------------------------------------

    def _get_templates_for_event(self, event_type: str) -> list:
        mapping = {
            "vitals_handoff":         "vitals_handoff",
            "vitals_progress":        "vitals_progress",
            "lab_result_handoff":     "lab_result_handoff",
            "lab_result_progress":    "lab_result_progress",
            "lab_result":             "lab_result_progress",
            "medication_change":      "medication_change",
            "ventilator_change":      "ventilator_change",
            "clinical_deterioration": "clinical_deterioration",
            "clinical_improvement":   "clinical_improvement",
            "shift_handoff":          "shift_handoff_summary",
        }
        key = mapping.get(event_type, "vitals_progress")
        return self.templates[key]

    # -------------------------------------------------------------------
    # Main generation flow
    # -------------------------------------------------------------------

    def generate_trial(self, num_keys: int, num_updates: int,
                       condition: str = None, seed: int = 0, **kwargs) -> dict:
        """Generate a single ICU narrative interference trial.

        Args:
            num_keys:    number of patients to track (2-12)
            num_updates: target tracked-value mentions per patient (3-30)
            condition:   ignored — both RI and PI questions are always generated
            seed:        random seed for full reproducibility
            **kwargs:    optional config overrides

        Returns:
            dict matching the output JSON schema with both RI and PI questions
        """
        rng = random.Random(seed)

        # Build config
        config = self._auto_config(num_keys, num_updates, seed, rng)
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Assign patients
        all_names = self.data["patient_names"]
        used_names: set = set()
        patients: List[PatientState] = []

        for i in range(num_keys):
            # Pick sex
            sex = rng.choice(["M", "F"])
            name_pool = all_names["male"] if sex == "M" else all_names["female"]
            available = [n for n in name_pool if n not in used_names]
            if not available:
                available = name_pool
            name = rng.choice(available)
            used_names.add(name)

            age = rng.randint(35, 88)
            diagnosis_key = config.diagnoses[i]
            trajectory = self._assign_trajectory(diagnosis_key, config.trajectory_mix, rng)
            bed_num = i + 1
            patient_id = str(bed_num)

            ps = self._init_patient_state(
                patient_id=patient_id,
                name=name,
                age=age,
                sex=sex,
                diagnosis_key=diagnosis_key,
                trajectory=trajectory,
                rng=rng,
            )
            patients.append(ps)

        # Assign tracked attributes per patient
        tracked: Dict[str, str] = {}
        if config.attribute_mode == "same":
            for ps in patients:
                tracked[ps.name] = config.tracked_attribute
        else:
            for ps in patients:
                dx = ps.primary_diagnosis
                affinity = DIAGNOSIS_ATTR_AFFINITY.get(dx, CORE_TRACKED_ATTRIBUTES)
                tracked[ps.name] = rng.choice(affinity)

        # Initialize entity tracking and mention counts
        entity_tracking: Dict[str, List[str]] = {
            f"{ps.name} / {tracked[ps.name]}": [] for ps in patients
        }
        mention_counts: Dict[str, int] = {ps.name: 0 for ps in patients}

        # Event generation loop
        shift_time_hours = 0.0
        event_log = []
        full_state_log = []
        total_target_mentions = num_keys * num_updates
        last_updated_patient = None
        max_iterations = total_target_mentions * 8

        iteration = 0
        while sum(mention_counts.values()) < total_target_mentions and shift_time_hours < 12.0:
            iteration += 1
            if iteration > max_iterations:
                break

            # Build event weights (suppress vent events for non-vented patients if primary target)
            weights = dict(DEFAULT_EVENT_WEIGHTS)

            # Pick a patient (prefer under-mentioned)
            candidates = [ps for ps in patients if ps.name != last_updated_patient]
            if not candidates:
                candidates = patients[:]

            min_m = min(mention_counts[ps.name] for ps in candidates)
            urgent = [ps for ps in candidates if mention_counts[ps.name] < 2]
            if urgent:
                candidates = urgent
            else:
                priority = [ps for ps in candidates if mention_counts[ps.name] <= min_m + 1]
                if priority:
                    candidates = priority

            patient = rng.choice(candidates)
            attr = tracked[patient.name]

            # Suppress ventilator events for non-vented patients
            event_weights = dict(weights)
            if not patient.on_ventilator:
                event_weights.pop("ventilator_change", None)
            # Also remove shift_handoff from inline weights (handled specially)
            event_weights.pop("shift_handoff", None)
            event_weights = {k: v for k, v in event_weights.items() if v > 0}

            event_type = self.weighted_random_choice(event_weights, rng)

            # Execute event
            state_changes = []
            event_data = {}

            if event_type in ("vitals_handoff", "vitals_progress"):
                event_data, state_changes = self._handle_vitals_check(
                    patient, attr, shift_time_hours, config.shift_start, rng, config)
            elif event_type == "lab_result":
                event_data, state_changes = self._handle_lab_result(
                    patient, attr, shift_time_hours, config.shift_start, rng, config)
            elif event_type == "medication_change":
                event_data, state_changes = self._handle_medication_change(
                    patient, attr, shift_time_hours, config.shift_start, rng, config)
            elif event_type == "ventilator_change":
                event_data, state_changes = self._handle_ventilator_change(
                    patient, attr, shift_time_hours, config.shift_start, rng, config)
            elif event_type in ("clinical_deterioration", "clinical_improvement"):
                event_data, state_changes = self._handle_clinical_event(
                    patient, attr, shift_time_hours, config.shift_start, rng, config, event_type)
            else:
                event_data, state_changes = self._handle_vitals_check(
                    patient, attr, shift_time_hours, config.shift_start, rng, config)

            # Record state changes
            for patient_name, a, old_v, new_v, mentioned in state_changes:
                full_state_log.append({
                    "time_hours": round(shift_time_hours, 2),
                    "event": event_type,
                    "entity": patient_name,
                    "attribute": a,
                    "old_value": str(old_v),
                    "new_value": str(new_v),
                    "mentioned_in_narrative": mentioned,
                })
                if mentioned:
                    key = f"{patient_name} / {tracked.get(patient_name, a)}"
                    if key in entity_tracking:
                        # Only record if value actually changed from last entry
                        if not entity_tracking[key] or entity_tracking[key][-1] != str(new_v):
                            entity_tracking[key].append(str(new_v))
                            mention_counts[patient_name] += 1

            event_log.append({
                "type": event_type,
                "time_hours": round(shift_time_hours, 2),
                "data": event_data,
            })
            last_updated_patient = patient.name

            # Advance time
            dt_range = EVENT_TIME_ADVANCE.get(event_type, (0.5, 2.0))
            dt = rng.uniform(dt_range[0], dt_range[1])
            shift_time_hours += dt
            shift_time_hours = round(shift_time_hours, 2)

        # Enforce minimum 2 mentions per entity
        for force_pass in range(5):
            under = [ps for ps in patients if mention_counts[ps.name] < 2]
            if not under:
                break
            for ps in under:
                attr = tracked[ps.name]
                event_data, state_changes = self._handle_vitals_check(
                    ps, attr, shift_time_hours, config.shift_start, rng, config)
                for pn, a, ov, nv, mentioned in state_changes:
                    full_state_log.append({
                        "time_hours": round(shift_time_hours, 2),
                        "event": "vitals_progress",
                        "entity": pn,
                        "attribute": a,
                        "old_value": str(ov),
                        "new_value": str(nv),
                        "mentioned_in_narrative": mentioned,
                    })
                    if mentioned:
                        key = f"{pn} / {tracked.get(pn, a)}"
                        if key in entity_tracking:
                            if not entity_tracking[key] or entity_tracking[key][-1] != str(nv):
                                entity_tracking[key].append(str(nv))
                                mention_counts[pn] += 1
                event_log.append({
                    "type": "vitals_progress",
                    "time_hours": round(shift_time_hours, 2),
                    "data": event_data,
                })
                shift_time_hours += rng.uniform(0.5, 1.5)
                shift_time_hours = round(shift_time_hours, 2)

        # Render narrative
        narrative_parts = []

        # Header
        header_template = rng.choice(self.templates["header_templates"])
        header = self._render_header(header_template, config, rng, num_keys)
        narrative_parts.append(header)
        narrative_parts.append("")

        filler_cfg = FILLER_CONFIG[config.filler_budget]

        for event in event_log:
            tmpl_list = self._get_templates_for_event(event["type"])
            template = rng.choice(tmpl_list)
            rendered = self.render_template(template, event["data"])
            narrative_parts.append(rendered)

            # Maybe insert filler
            if rng.random() < filler_cfg["insert_probability"]:
                n_sentences = filler_cfg["sentences_per_insertion"]
                if isinstance(n_sentences, tuple):
                    n_sentences = rng.randint(n_sentences[0], n_sentences[1])
                filler_tmpl_list = rng.sample(
                    self.templates["filler_templates"],
                    min(n_sentences, len(self.templates["filler_templates"]))
                )
                filler_patient = rng.choice(patients)
                for ft in filler_tmpl_list:
                    filler_rendered = self._render_filler(ft, filler_patient, rng)
                    narrative_parts.append(filler_rendered)

        # Question generation
        queried_idx = min(config.queried_patient_idx, len(patients) - 1)
        queried_patient = patients[queried_idx]
        queried_attr = tracked[queried_patient.name]
        tracking_key = f"{queried_patient.name} / {queried_attr}"
        values = entity_tracking[tracking_key]

        # Find a patient with distinct first/last values
        if len(values) < 2 or values[0] == values[-1]:
            for alt in patients:
                alt_key = f"{alt.name} / {tracked[alt.name]}"
                alt_values = entity_tracking[alt_key]
                if len(alt_values) >= 2 and alt_values[0] != alt_values[-1]:
                    queried_patient = alt
                    queried_attr = tracked[alt.name]
                    tracking_key = alt_key
                    values = alt_values
                    break

        # Last resort: force a bonus event on best candidate
        if len(values) < 2 or values[0] == values[-1]:
            best_patient = max(
                patients,
                key=lambda p: len(entity_tracking[f"{p.name} / {tracked[p.name]}"])
            )
            best_attr = tracked[best_patient.name]
            best_key = f"{best_patient.name} / {best_attr}"

            old_formatted = best_patient.format_attribute(best_attr)
            bounds = PHYS_BOUNDS.get(best_attr, (0, 9999))
            delta = rng.uniform(3.0, 12.0) * (1 if rng.random() < 0.5 else -1)
            new_raw = clamp(best_patient.get_attribute(best_attr) + delta, bounds[0], bounds[1])
            prec = ATTR_PRECISION.get(best_attr, 1)
            if prec == 0:
                new_raw = round(new_raw)
            else:
                new_raw = round(new_raw, prec)
            best_patient.set_attribute(best_attr, new_raw)
            new_formatted = best_patient.format_attribute(best_attr)

            attr_label = self.attr_labels.get(best_attr, best_attr.replace("_", " "))
            bonus_text = (f"At end of shift, {best_patient.name}'s {attr_label} "
                         f"was recorded as {new_formatted}.")
            narrative_parts.append(bonus_text)

            if not entity_tracking[best_key] or entity_tracking[best_key][-1] != new_formatted:
                entity_tracking[best_key].append(new_formatted)
                mention_counts[best_patient.name] += 1

            queried_patient = best_patient
            queried_attr = best_attr
            tracking_key = best_key
            values = entity_tracking[best_key]

        ri_answer = values[0] if values else "unknown"
        pi_answer = values[-1] if values else "unknown"

        attr_label = self.attr_labels.get(queried_attr, queried_attr.replace("_", " "))
        units_str = self.attr_units.get(queried_attr, "")
        units_phrase = f" ({units_str})" if units_str else ""

        RI_QUESTION_TEMPLATES = [
            f"What was {queried_patient.name}'s {attr_label}{units_phrase} when first recorded in this shift?",
            f"At {queried_patient.name}'s first mention in the narrative, what was the {attr_label}?",
            f"What value was initially documented for {queried_patient.name}'s {attr_label}?",
        ]
        PI_QUESTION_TEMPLATES = [
            f"What was {queried_patient.name}'s most recently recorded {attr_label}{units_phrase}?",
            f"In the last update for {queried_patient.name}, what was the {attr_label}?",
            f"What was {queried_patient.name}'s {attr_label} at the final mention in this narrative?",
        ]

        questions = {
            "RI": {
                "question": rng.choice(RI_QUESTION_TEMPLATES),
                "expected_answer": ri_answer,
                "target_entity": queried_patient.name,
                "target_attribute": queried_attr,
            },
            "PI": {
                "question": rng.choice(PI_QUESTION_TEMPLATES),
                "expected_answer": pi_answer,
                "target_entity": queried_patient.name,
                "target_attribute": queried_attr,
            },
        }

        # Assemble narrative
        final_parts = []
        for p in narrative_parts:
            if p == "":
                if final_parts and not final_parts[-1].endswith("\n"):
                    final_parts.append("\n")
            else:
                final_parts.append(p)
        narrative = " ".join(final_parts).replace(" \n ", "\n\n")

        return {
            "id": f"icu_{seed:06d}",
            "domain": "icu",
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
                "tracked_attribute": config.tracked_attribute,
                "attribute_mode": config.attribute_mode,
                "diagnosis_mode": config.diagnosis_mode,
                "icu_type": config.icu_type,
                "shift_start": config.shift_start,
                "filler_budget": config.filler_budget,
                "voice": config.voice,
                "shift_archetype": config.shift_archetype,
                "trajectory_mix": config.trajectory_mix,
                "patients": [
                    {
                        "name": ps.name,
                        "bed": ps.patient_id,
                        "age": ps.age,
                        "sex": ps.sex,
                        "diagnosis": ps.primary_diagnosis,
                        "trajectory": ps.trajectory,
                        "tracked_attribute": tracked[ps.name],
                    }
                    for ps in patients
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

        # Check that all tracked values appear in narrative
        for key, vals in entity_tracking.items():
            for val in vals:
                assert val in narrative, (
                    f"Value '{val}' for '{key}' not found in narrative text"
                )

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

        # Check RI = first, PI = last
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

    DOMAIN = "icu"

    @staticmethod
    def _get_data_dir() -> Path:
        """Return data/narrative_interference/icu/ (project root relative)."""
        current = Path(__file__).resolve().parent
        for _ in range(10):
            if (current / ".git").exists() or (current / "CLAUDE.md").exists():
                break
            current = current.parent
        return current / "data" / "narrative_interference" / "icu"

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
            key_levels:       list of num_keys values (default: [2,3,5,7,10])
            update_levels:    list of num_updates values (default: [1,3,5,10,20,30])
            trials_per_cell:  trials per (keys, updates) cell
            seed_start:       starting seed value
            **kwargs:         config overrides passed to generate_trial()

        Returns:
            dict with 'metadata' and 'trials' keys
        """
        import time
        from datetime import datetime, timezone

        if key_levels is None:
            key_levels = [2, 3, 5, 7, 10]
        if update_levels is None:
            update_levels = [1, 3, 5, 10, 20, 30]

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        trials = []
        failures = []
        seed = seed_start
        total_cells = len(key_levels) * len(update_levels)
        cell_idx = 0
        t0 = time.time()

        for nk in key_levels:
            for nu in update_levels:
                cell_idx += 1
                cell_ok = 0
                for _ in range(trials_per_cell):
                    try:
                        trial = self.generate_trial(nk, nu, None, seed, **kwargs)
                        self.validate_trial(trial)
                        trials.append(trial)
                        cell_ok += 1
                    except Exception as e:
                        failures.append({
                            "seed": seed, "nk": nk, "nu": nu,
                            "error": str(e)[:100],
                        })
                    seed += 1

                elapsed = time.time() - t0
                rate = len(trials) / max(elapsed, 0.1)
                print(f"  [{cell_idx}/{total_cells}] keys={nk:>2} updates={nu:>3}: "
                      f"{cell_ok}/{trials_per_cell} ok ({rate:.0f}/sec)")

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
                "trials_per_cell": trials_per_cell,
                "seed_range": f"{seed_start}-{seed - 1}",
                "overrides": kwargs if kwargs else "all_random",
            },
            "trials": trials,
        }

    def save_batch(self, batch: dict, tag: str = None) -> Path:
        """Save a generated batch to data/narrative_interference/icu/.

        Args:
            batch: output from generate_batch()
            tag:   optional short tag (e.g., 'hr_same', 'v1')

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
