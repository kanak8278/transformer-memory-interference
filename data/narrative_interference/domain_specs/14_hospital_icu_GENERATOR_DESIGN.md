# Hospital ICU Narrative Interference Generator — System Design

## 1. What We're Building

A function: `generate_icu_trial(num_keys, num_updates, seed) → dict`

Where:
- `num_keys` = number of patients tracked in the narrative (2-12)
- `num_updates` = number of times each patient's tracked attribute changes across observations (3-30)
- `seed` = random seed for reproducibility

Returns a dict containing:
```python
{
    "narrative": str,           # The full ICU narrative text
    "questions": {
        "RI": {
            "question": str,        # Question about the FIRST value
            "expected_answer": str,  # The first recorded value
            "target_entity": str,
            "target_attribute": str,
        },
        "PI": {
            "question": str,        # Question about the LAST value
            "expected_answer": str,  # The most recent value
            "target_entity": str,
            "target_attribute": str,
        },
    },
    "entity_tracking": dict,    # Full value history per entity
    "config": dict,             # Trial configuration metadata
}
```

**Note:** Both RI and PI questions are generated for every trial. The `condition` parameter is removed from the generator signature — the caller decides which question to pose to the model.

## 2. Core Design Decisions

### What is the "tracked attribute"?

The ICU is uniquely suited for interference because vital signs naturally create confusable values — every adult patient has heart rate in roughly 50-150 bpm, systolic BP in 80-200 mmHg, SpO2 in 85-100%, temperature in 35-40°C. When 5 patients all have HR in the 90-120 range, the model must track WHICH patient had WHICH heart rate at WHICH timepoint.

**Tracked attribute candidates (pool of 25+):**

**Vital signs (numeric, continuous, frequent updates):**
- **heart_rate** — 50-180 bpm. Updated q1h. Very confusable range across patients. Best primary candidate.
- **systolic_bp** — 70-200 mmHg. Updated q1h. Wide range but clusters by diagnosis.
- **diastolic_bp** — 40-120 mmHg. Updated q1h.
- **map** — 50-130 mmHg. Updated q1h. Clinical target is ≥65.
- **respiratory_rate** — 8-40 breaths/min. Updated q1h.
- **spo2** — 80-100%. Updated continuously but documented q1h. Tight range = very confusable.
- **temperature_c** — 35.0-41.0°C. Updated q4h. Very tight range (6 degrees total).

**Lab values (numeric, less frequent, high precision):**
- **potassium** — 2.5-6.5 mEq/L. Updated q6-24h. Narrow range with decimal precision.
- **sodium** — 120-160 mEq/L. Updated q24h.
- **creatinine** — 0.5-10.0 mg/dL. Updated q24h. Decimal precision.
- **lactate** — 0.5-15.0 mmol/L. Updated q2-6h in sepsis. Key prognostic marker.
- **hemoglobin** — 5.0-16.0 g/dL. Updated q6-24h.
- **wbc** — 1.0-40.0 thousand/uL. Updated q24h.
- **platelets** — 10-400 thousand/uL. Updated q24h.
- **bilirubin** — 0.1-25.0 mg/dL. Updated q24h.
- **troponin** — 0.01-50.0 ng/mL. Updated q3-6h.
- **glucose** — 40-600 mg/dL. Updated q1-6h.
- **inr** — 0.8-8.0. Updated q6-24h.
- **procalcitonin** — 0.01-100 ng/mL. Updated q24-48h.

**Clinical scores (ordinal/integer):**
- **gcs** — 3-15. Updated q1-4h.
- **rass** — -5 to +4. Updated q1-2h.
- **pain_score** — 0-10. Updated q4h.
- **sofa_score** — 0-24. Updated q24h.

**Treatment parameters (numeric, changes with interventions):**
- **norepinephrine_dose** — 0-3.0 mcg/kg/min. Updated with titrations.
- **fio2** — 21-100%. Updated with ventilator changes.
- **peep** — 0-24 cmH2O. Updated with ventilator changes.
- **urine_output_ml_hr** — 0-200 mL/hr. Updated q1h.
- **fluid_balance_ml** — -2000 to +5000 mL. Updated q12h.

**Categorical:**
- **ventilator_mode** — none / nasal_cannula / high_flow / bipap / AC_VC / AC_PC / PSV / APRV
- **code_status** — full / DNR / DNI / comfort
- **diet_status** — NPO / clear_liquid / regular / tube_feeds
- **vasopressor_type** — none / norepinephrine / vasopressin / epinephrine / phenylephrine / dopamine
- **sedation_agent** — none / propofol / dexmedetomidine / midazolam / ketamine
- **antibiotic_regimen** — none / vanc_pip_tazo / vanc_cefepime / meropenem / ceftriaxone_azithro / etc.

**Total: 30+ tracked attribute options.** The numeric vital signs and labs are the interference powerhouses — their ranges overlap massively across patients.

### 2.1 TRACKABLE_ATTRIBUTES Specification

```python
TRACKABLE_ATTRIBUTES = [
    # ── Vital Signs ──
    {
        "name": "heart_rate",
        "type": "int",
        "range": [40, 200],
        "format_str": "{value} bpm",
        "precision": 0,  # integer
        "direction": "volatile",
        "update_events": ["vital_sign_check", "clinical_deterioration", "clinical_improvement", "medication_change", "shift_handoff"],
        "interference_quality": "excellent",  # tight, confusable range across patients
        "diagnosis_relevant": ["sepsis", "stemi", "dka", "gi_bleed", "pe", "alcohol_withdrawal", "anaphylaxis", "thyroid_storm", "cardiogenic_shock"],
    },
    {
        "name": "systolic_bp",
        "type": "int",
        "range": [50, 250],
        "format_str": "{value} mmHg",
        "precision": 0,
        "direction": "volatile",
        "update_events": ["vital_sign_check", "clinical_deterioration", "clinical_improvement", "medication_change", "shift_handoff"],
        "interference_quality": "good",
        "diagnosis_relevant": ["sepsis", "gi_bleed", "hemorrhagic_stroke", "hypertensive_emergency", "cardiogenic_shock", "anaphylaxis"],
    },
    {
        "name": "diastolic_bp",
        "type": "int",
        "range": [30, 140],
        "format_str": "{value} mmHg",
        "precision": 0,
        "direction": "volatile",
        "update_events": ["vital_sign_check", "clinical_deterioration", "medication_change", "shift_handoff"],
        "interference_quality": "moderate",
        "diagnosis_relevant": ["sepsis", "hypertensive_emergency", "cardiogenic_shock"],
    },
    {
        "name": "map",
        "type": "int",
        "range": [35, 160],
        "format_str": "MAP {value}",
        "precision": 0,
        "direction": "volatile",
        "update_events": ["vital_sign_check", "medication_change", "clinical_deterioration", "clinical_improvement", "shift_handoff"],
        "interference_quality": "good",
        "diagnosis_relevant": ["sepsis", "cardiogenic_shock", "gi_bleed", "anaphylaxis", "hemorrhagic_stroke"],
    },
    {
        "name": "respiratory_rate",
        "type": "int",
        "range": [4, 45],
        "format_str": "{value} breaths/min",
        "precision": 0,
        "direction": "volatile",
        "update_events": ["vital_sign_check", "ventilator_change", "clinical_deterioration", "shift_handoff"],
        "interference_quality": "good",
        "diagnosis_relevant": ["ards", "copd_exacerbation", "dka", "pe", "drug_overdose", "pneumonia"],
    },
    {
        "name": "spo2",
        "type": "int",
        "range": [60, 100],
        "format_str": "{value}%",
        "precision": 0,
        "direction": "volatile",
        "update_events": ["vital_sign_check", "ventilator_change", "clinical_deterioration", "clinical_improvement", "shift_handoff"],
        "interference_quality": "excellent",  # very tight range = very confusable
        "diagnosis_relevant": ["ards", "copd_exacerbation", "pe", "pneumonia", "drug_overdose", "anaphylaxis"],
    },
    {
        "name": "temperature_c",
        "type": "float",
        "range": [34.0, 42.0],
        "format_str": "{value}°C",
        "precision": 1,
        "direction": "volatile",
        "update_events": ["vital_sign_check", "nursing_assessment", "shift_handoff"],
        "interference_quality": "excellent",  # 6-degree total range = very confusable
        "diagnosis_relevant": ["sepsis", "pneumonia", "alcohol_withdrawal", "meningitis", "thyroid_storm", "major_burns"],
    },
    # ── Lab Values ──
    {
        "name": "potassium",
        "type": "float",
        "range": [2.0, 7.5],
        "format_str": "{value} mEq/L",
        "precision": 1,
        "direction": "volatile",
        "update_events": ["lab_result", "shift_handoff"],
        "interference_quality": "excellent",  # narrow range, decimal precision
        "diagnosis_relevant": ["aki", "dka", "acute_liver_failure", "alcohol_withdrawal"],
    },
    {
        "name": "sodium",
        "type": "int",
        "range": [115, 165],
        "format_str": "{value} mEq/L",
        "precision": 0,
        "direction": "volatile",
        "update_events": ["lab_result", "shift_handoff"],
        "interference_quality": "good",
        "diagnosis_relevant": ["aki", "acute_liver_failure", "dka"],
    },
    {
        "name": "creatinine",
        "type": "float",
        "range": [0.3, 12.0],
        "format_str": "{value} mg/dL",
        "precision": 1,
        "direction": "volatile",
        "update_events": ["lab_result", "shift_handoff"],
        "interference_quality": "excellent",
        "diagnosis_relevant": ["aki", "sepsis", "cardiogenic_shock", "hypertensive_emergency"],
    },
    {
        "name": "lactate",
        "type": "float",
        "range": [0.3, 20.0],
        "format_str": "{value} mmol/L",
        "precision": 1,
        "direction": "volatile",
        "update_events": ["lab_result", "shift_handoff"],
        "interference_quality": "excellent",
        "diagnosis_relevant": ["sepsis", "cardiogenic_shock", "massive_transfusion", "status_epilepticus"],
    },
    {
        "name": "hemoglobin",
        "type": "float",
        "range": [4.0, 18.0],
        "format_str": "{value} g/dL",
        "precision": 1,
        "direction": "volatile",
        "update_events": ["lab_result", "transfusion", "shift_handoff"],
        "interference_quality": "excellent",
        "diagnosis_relevant": ["gi_bleed", "massive_transfusion", "post_cardiac_surgery", "major_burns"],
    },
    {
        "name": "wbc",
        "type": "float",
        "range": [0.5, 50.0],
        "format_str": "{value}k/uL",
        "precision": 1,
        "direction": "volatile",
        "update_events": ["lab_result", "shift_handoff"],
        "interference_quality": "good",
        "diagnosis_relevant": ["sepsis", "pneumonia", "pancreatitis", "meningitis"],
    },
    {
        "name": "platelets",
        "type": "int",
        "range": [5, 500],
        "format_str": "{value}k",
        "precision": 0,
        "direction": "volatile",
        "update_events": ["lab_result", "transfusion", "shift_handoff"],
        "interference_quality": "good",
        "diagnosis_relevant": ["sepsis", "acute_liver_failure", "massive_transfusion", "gi_bleed"],
    },
    {
        "name": "glucose",
        "type": "int",
        "range": [30, 900],
        "format_str": "{value} mg/dL",
        "precision": 0,
        "direction": "volatile",
        "update_events": ["lab_result", "medication_change", "shift_handoff"],
        "interference_quality": "good",
        "diagnosis_relevant": ["dka", "acute_liver_failure", "sepsis", "pancreatitis"],
    },
    # ── Additional Labs ──
    {
        "name": "troponin",
        "type": "float",
        "range": [0.01, 60.0],
        "format_str": "{value} ng/mL",
        "precision": 2,
        "direction": "volatile",
        "update_events": ["lab_result", "shift_handoff"],
        "interference_quality": "moderate",
        "diagnosis_relevant": ["stemi", "pe", "post_cardiac_surgery", "hypertensive_emergency"],
    },
    {
        "name": "bilirubin",
        "type": "float",
        "range": [0.1, 30.0],
        "format_str": "{value} mg/dL",
        "precision": 1,
        "direction": "volatile",
        "update_events": ["lab_result", "shift_handoff"],
        "interference_quality": "moderate",
        "diagnosis_relevant": ["acute_liver_failure", "sepsis"],
    },
    {
        "name": "inr",
        "type": "float",
        "range": [0.8, 10.0],
        "format_str": "INR {value}",
        "precision": 1,
        "direction": "volatile",
        "update_events": ["lab_result", "shift_handoff"],
        "interference_quality": "good",
        "diagnosis_relevant": ["acute_liver_failure", "massive_transfusion", "gi_bleed"],
    },
    {
        "name": "procalcitonin",
        "type": "float",
        "range": [0.01, 100.0],
        "format_str": "{value} ng/mL",
        "precision": 2,
        "direction": "volatile",
        "update_events": ["lab_result", "shift_handoff"],
        "interference_quality": "moderate",
        "diagnosis_relevant": ["sepsis", "pneumonia"],
    },
    # ── Clinical Scores ──
    {
        "name": "gcs",
        "type": "int",
        "range": [3, 15],
        "format_str": "GCS {value}",
        "precision": 0,
        "direction": "volatile",
        "update_events": ["nursing_assessment", "clinical_deterioration", "clinical_improvement", "shift_handoff"],
        "interference_quality": "good",
        "diagnosis_relevant": ["hemorrhagic_stroke", "meningitis", "acute_liver_failure", "drug_overdose", "status_epilepticus"],
    },
    {
        "name": "rass",
        "type": "int",
        "range": [-5, 4],
        "format_str": "RASS {value}",
        "precision": 0,
        "direction": "volatile",
        "update_events": ["nursing_assessment", "medication_change", "shift_handoff"],
        "interference_quality": "good",
        "diagnosis_relevant": ["ards", "sepsis", "status_epilepticus", "alcohol_withdrawal"],
    },
    {
        "name": "pain_score",
        "type": "int",
        "range": [0, 10],
        "format_str": "{value}/10 pain",
        "precision": 0,
        "direction": "volatile",
        "update_events": ["nursing_assessment", "medication_change", "shift_handoff"],
        "interference_quality": "moderate",
        "diagnosis_relevant": ["pancreatitis", "stemi", "post_cardiac_surgery", "major_burns"],
    },
    {
        "name": "sofa_score",
        "type": "int",
        "range": [0, 24],
        "format_str": "SOFA {value}",
        "precision": 0,
        "direction": "volatile",
        "update_events": ["shift_handoff"],
        "interference_quality": "moderate",
        "diagnosis_relevant": ["sepsis", "cardiogenic_shock", "acute_liver_failure"],
    },
    # ── Treatment Parameters ──
    {
        "name": "norepinephrine_dose",
        "type": "float",
        "range": [0.0, 3.0],
        "format_str": "{value} mcg/kg/min",
        "precision": 2,
        "direction": "volatile",
        "update_events": ["medication_change", "clinical_deterioration", "clinical_improvement", "shift_handoff"],
        "interference_quality": "excellent",
        "diagnosis_relevant": ["sepsis", "cardiogenic_shock", "anaphylaxis", "gi_bleed"],
    },
    {
        "name": "fio2",
        "type": "int",
        "range": [21, 100],
        "format_str": "FiO2 {value}%",
        "precision": 0,
        "direction": "volatile",
        "update_events": ["ventilator_change", "clinical_deterioration", "clinical_improvement", "shift_handoff"],
        "interference_quality": "good",
        "diagnosis_relevant": ["ards", "copd_exacerbation", "pneumonia", "pe"],
    },
    {
        "name": "peep",
        "type": "int",
        "range": [0, 24],
        "format_str": "PEEP {value} cmH2O",
        "precision": 0,
        "direction": "volatile",
        "update_events": ["ventilator_change", "shift_handoff"],
        "interference_quality": "good",
        "diagnosis_relevant": ["ards", "copd_exacerbation", "pneumonia"],
    },
    {
        "name": "urine_output_ml_hr",
        "type": "int",
        "range": [0, 300],
        "format_str": "{value} mL/hr",
        "precision": 0,
        "direction": "volatile",
        "update_events": ["nursing_assessment", "shift_handoff"],
        "interference_quality": "moderate",
        "diagnosis_relevant": ["aki", "sepsis", "cardiogenic_shock", "major_burns"],
    },
    {
        "name": "fluid_balance_ml",
        "type": "int",
        "range": [-3000, 8000],
        "format_str": "{value:+,} mL",
        "precision": 0,
        "direction": "volatile",
        "update_events": ["shift_handoff", "nursing_assessment"],
        "interference_quality": "moderate",
        "diagnosis_relevant": ["aki", "sepsis", "ards", "major_burns"],
    },
]
```

### Patient homogeneity: same-diagnosis vs mixed-diagnosis

This is the entity heterogeneity question from wildlife, adapted to ICU.

**Same-diagnosis mode (~50% of trials):** All patients have the same primary diagnosis (e.g., 5 sepsis patients). Their vital signs follow similar trajectories, lab values move in similar directions → maximum interference. This is the hard condition.

Same-diagnosis options:
- **Sepsis cohort** — all septic, all with similar HR (100-130), lactate (2-8), MAP (<65 → recovering). Best interference candidate.
- **Post-cardiac surgery cohort** — all post-CABG/valve, similar hemodynamics, troponin trajectories.
- **Respiratory failure cohort** — all intubated ARDS patients, similar FiO2/PEEP/P:F trajectories.
- **DKA cohort** — all DKA, similar glucose (400→200→150), pH, anion gap trajectories.
- **GI bleed cohort** — all GI bleed, similar H/H drops, HR elevations, transfusion patterns.

**Mixed-diagnosis mode (~50% of trials):** Patients have different diagnoses. Value ranges diverge somewhat (sepsis patient HR 120 vs post-op patient HR 80 vs DKA patient HR 110), but vital signs still overlap enough to create interference. Lab values diverge more (troponin only elevated in cardiac patients, glucose only 400+ in DKA).

Typical mixes:
- 2 sepsis + 1 STEMI + 1 post-surgical + 1 COPD exacerbation
- 3 respiratory failure (different causes: ARDS, COPD, PE)
- 2 GI bleed + 2 sepsis + 1 liver failure
- 1 of each: sepsis, DKA, stroke, post-op, overdose

### How do non-tracked attributes fit in?

Same as other domains: non-tracked attributes appear as **filler/context**. They make the narrative realistic but aren't queried.

> "At the 06:00 assessment, Mrs. Liu's **heart rate had climbed to 118 bpm**, sinus tachycardia on the monitor. Blood pressure was 92/58, MAP 69 — still above the vasopressor threshold. Her temperature was 38.6°C, up from 37.9 earlier, and the nurse noted she was becoming more diaphoretic. Lactate was pending from the 05:00 draw."

Here heart_rate (bold) is the tracked attribute. BP, MAP, temperature, diaphoresis, and lactate are filler.

## 3. State Machine Design

### 3.0 DIAGNOSIS_REGISTRY

Every diagnosis referenced anywhere in this spec must appear in this registry. If a diagnosis is not here, it cannot be sampled.

```python
DIAGNOSIS_REGISTRY = {
    # ── Fully implemented (have profiles + trajectories) ──
    "sepsis": {
        "display_name": "Sepsis / Septic Shock",
        "trajectory_source": "14_hospital_icu_CLINICAL_TRAJECTORIES.md § Sepsis / Septic Shock",
        "icu_types": ["micu", "sicu", "mixed"],
        "status": "IMPLEMENTED",
    },
    "stemi": {
        "display_name": "Acute MI / STEMI",
        "trajectory_source": "14_hospital_icu_CLINICAL_TRAJECTORIES.md § Acute MI / STEMI",
        "icu_types": ["ccu", "mixed"],
        "status": "IMPLEMENTED",
    },
    "dka": {
        "display_name": "Diabetic Ketoacidosis",
        "trajectory_source": "14_hospital_icu_CLINICAL_TRAJECTORIES.md § DKA",
        "icu_types": ["micu", "mixed"],
        "status": "IMPLEMENTED",
    },
    "ards": {
        "display_name": "ARDS / Acute Respiratory Failure",
        "trajectory_source": "14_hospital_icu_CLINICAL_TRAJECTORIES.md § ARDS",
        "icu_types": ["micu", "sicu", "mixed"],
        "status": "IMPLEMENTED",
    },
    "gi_bleed": {
        "display_name": "GI Bleed",
        "trajectory_source": "14_hospital_icu_CLINICAL_TRAJECTORIES.md § GI Bleed",
        "icu_types": ["micu", "sicu", "mixed"],
        "status": "IMPLEMENTED",
    },
    "copd_exacerbation": {
        "display_name": "COPD Exacerbation",
        "trajectory_source": "14_hospital_icu_CLINICAL_TRAJECTORIES.md § COPD Exacerbation",
        "icu_types": ["micu", "mixed"],
        "status": "IMPLEMENTED",
    },
    "hemorrhagic_stroke": {
        "display_name": "Hemorrhagic Stroke / ICH",
        "trajectory_source": "14_hospital_icu_CLINICAL_TRAJECTORIES.md § Hemorrhagic Stroke / ICH",
        "icu_types": ["nicu", "mixed"],
        "status": "IMPLEMENTED",
    },
    "post_cardiac_surgery": {
        "display_name": "Post-Cardiac Surgery (CABG/Valve)",
        "trajectory_source": "14_hospital_icu_CLINICAL_TRAJECTORIES.md § Post-Cardiac Surgery",
        "icu_types": ["sicu", "ccu", "mixed"],
        "status": "IMPLEMENTED",
    },
    "acute_liver_failure": {
        "display_name": "Acute Liver Failure",
        "trajectory_source": "14_hospital_icu_CLINICAL_TRAJECTORIES.md § Acute Liver Failure",
        "icu_types": ["micu", "mixed"],
        "status": "IMPLEMENTED",
    },
    "alcohol_withdrawal": {
        "display_name": "Alcohol Withdrawal / Delirium Tremens",
        "trajectory_source": "14_hospital_icu_CLINICAL_TRAJECTORIES.md § Alcohol Withdrawal",
        "icu_types": ["micu", "mixed"],
        "status": "IMPLEMENTED",
    },
    "aki": {
        "display_name": "Acute Kidney Injury",
        "trajectory_source": "14_hospital_icu_CLINICAL_TRAJECTORIES.md § Acute Kidney Injury",
        "icu_types": ["micu", "mixed"],
        "status": "IMPLEMENTED",
    },
    "pe": {
        "display_name": "Pulmonary Embolism",
        "trajectory_source": "14_hospital_icu_CLINICAL_TRAJECTORIES.md § Pulmonary Embolism",
        "icu_types": ["micu", "ccu", "mixed"],
        "status": "IMPLEMENTED",
    },
    "pneumonia": {
        "display_name": "Pneumonia (severe, ICU-requiring)",
        "trajectory_source": "14_hospital_icu_CLINICAL_TRAJECTORIES.md § (derived from Sepsis + ARDS sections)",
        "icu_types": ["micu", "mixed"],
        "status": "IMPLEMENTED",
    },
    "drug_overdose": {
        "display_name": "Drug Overdose (Opioid)",
        "trajectory_source": "14_hospital_icu_CLINICAL_TRAJECTORIES.md § Drug Overdose (Opioid)",
        "icu_types": ["micu", "mixed"],
        "status": "IMPLEMENTED",
    },
    "pancreatitis": {
        "display_name": "Acute Pancreatitis",
        "trajectory_source": "14_hospital_icu_CLINICAL_TRAJECTORIES.md § Acute Pancreatitis",
        "icu_types": ["micu", "sicu", "mixed"],
        "status": "IMPLEMENTED",
    },
    "anaphylaxis": {
        "display_name": "Anaphylaxis",
        "trajectory_source": "14_hospital_icu_CLINICAL_TRAJECTORIES.md § Anaphylaxis",
        "icu_types": ["micu", "mixed"],
        "status": "IMPLEMENTED",
    },
    "meningitis": {
        "display_name": "Bacterial Meningitis",
        "trajectory_source": "14_hospital_icu_CLINICAL_TRAJECTORIES.md § Bacterial Meningitis",
        "icu_types": ["nicu", "micu", "mixed"],
        "status": "IMPLEMENTED",
    },
    "cardiogenic_shock": {
        "display_name": "Cardiogenic Shock",
        "trajectory_source": "14_hospital_icu_CLINICAL_TRAJECTORIES.md § Cardiogenic Shock",
        "icu_types": ["ccu", "mixed"],
        "status": "IMPLEMENTED",
    },
    "hypertensive_emergency": {
        "display_name": "Hypertensive Emergency",
        "trajectory_source": "14_hospital_icu_CLINICAL_TRAJECTORIES.md § Hypertensive Emergency",
        "icu_types": ["micu", "ccu", "nicu", "mixed"],
        "status": "IMPLEMENTED",
    },
    "major_burns": {
        "display_name": "Major Burns",
        "trajectory_source": "14_hospital_icu_CLINICAL_TRAJECTORIES.md § Major Burns",
        "icu_types": ["sicu", "mixed"],
        "status": "IMPLEMENTED",
    },
    "massive_transfusion": {
        "display_name": "Massive Transfusion (Trauma)",
        "trajectory_source": "14_hospital_icu_CLINICAL_TRAJECTORIES.md § Massive Transfusion (Trauma)",
        "icu_types": ["sicu", "mixed"],
        "status": "IMPLEMENTED",
    },
    "status_epilepticus": {
        "display_name": "Status Epilepticus",
        "trajectory_source": "14_hospital_icu_CLINICAL_TRAJECTORIES.md § Status Epilepticus",
        "icu_types": ["nicu", "micu", "mixed"],
        "status": "IMPLEMENTED",
    },
    "thyroid_storm": {
        "display_name": "Thyroid Storm",
        "trajectory_source": "14_hospital_icu_CLINICAL_TRAJECTORIES.md § Thyroid Storm",
        "icu_types": ["micu", "mixed"],
        "status": "IMPLEMENTED",
    },
    # ── NOT IMPLEMENTED — referenced in ICU_UNITS but have no trajectories ──
    # DO NOT sample these. They exist only for documentation completeness.
    "afib_rvr": {
        "display_name": "Atrial Fibrillation with RVR",
        "trajectory_source": None,
        "icu_types": ["ccu", "mixed"],
        "status": "NOT_IMPLEMENTED",
    },
    "tbi": {
        "display_name": "Traumatic Brain Injury",
        "trajectory_source": None,
        "icu_types": ["nicu", "sicu", "mixed"],
        "status": "NOT_IMPLEMENTED",
    },
    "ischemic_stroke": {
        "display_name": "Ischemic Stroke",
        "trajectory_source": None,
        "icu_types": ["nicu", "mixed"],
        "status": "NOT_IMPLEMENTED",
    },
    "decompensated_hf": {
        "display_name": "Decompensated Heart Failure",
        "trajectory_source": None,
        "icu_types": ["ccu", "mixed"],
        "status": "NOT_IMPLEMENTED",
    },
    "post_abdominal_surgery": {
        "display_name": "Post-Abdominal Surgery",
        "trajectory_source": None,
        "icu_types": ["sicu", "mixed"],
        "status": "NOT_IMPLEMENTED",
    },
    "polytrauma": {
        "display_name": "Polytrauma",
        "trajectory_source": None,
        "icu_types": ["sicu", "mixed"],
        "status": "NOT_IMPLEMENTED",
    },
}

# Helper: pool of sampleable diagnoses (status == IMPLEMENTED only)
DIAGNOSIS_POOL = [k for k, v in DIAGNOSIS_REGISTRY.items() if v["status"] == "IMPLEMENTED"]
# Total: 23 implemented diagnoses
```

### 3.0.1 DIAGNOSIS_ATTRIBUTE_AFFINITY

Maps each diagnosis to its most clinically meaningful tracked attributes, ordered by relevance. When `attribute_mode == "mixed"`, sample from this list for each patient.

```python
DIAGNOSIS_ATTRIBUTE_AFFINITY = {
    "sepsis":              ["lactate", "heart_rate", "map", "norepinephrine_dose", "temperature_c", "wbc", "creatinine", "procalcitonin"],
    "stemi":               ["troponin", "heart_rate", "systolic_bp", "map"],
    "dka":                 ["glucose", "potassium", "respiratory_rate", "heart_rate"],
    "ards":                ["spo2", "fio2", "peep", "respiratory_rate"],
    "gi_bleed":            ["hemoglobin", "heart_rate", "systolic_bp", "map"],
    "copd_exacerbation":   ["spo2", "respiratory_rate", "heart_rate"],
    "hemorrhagic_stroke":  ["systolic_bp", "gcs", "map", "heart_rate"],
    "post_cardiac_surgery":["troponin", "heart_rate", "hemoglobin", "systolic_bp", "map"],
    "acute_liver_failure": ["inr", "bilirubin", "gcs", "creatinine", "lactate"],
    "alcohol_withdrawal":  ["heart_rate", "systolic_bp", "temperature_c"],
    "aki":                 ["creatinine", "potassium", "urine_output_ml_hr"],
    "pe":                  ["heart_rate", "spo2", "troponin", "systolic_bp"],
    "pneumonia":           ["temperature_c", "wbc", "spo2", "heart_rate", "respiratory_rate", "procalcitonin"],
    "drug_overdose":       ["respiratory_rate", "spo2", "gcs", "heart_rate"],
    "pancreatitis":        ["heart_rate", "temperature_c", "wbc", "creatinine"],
    "anaphylaxis":         ["heart_rate", "systolic_bp", "map", "spo2"],
    "meningitis":          ["gcs", "temperature_c", "heart_rate", "wbc"],
    "cardiogenic_shock":   ["lactate", "map", "norepinephrine_dose", "heart_rate", "systolic_bp"],
    "hypertensive_emergency": ["systolic_bp", "diastolic_bp", "map", "heart_rate", "creatinine"],
    "major_burns":         ["heart_rate", "hemoglobin", "temperature_c", "urine_output_ml_hr"],
    "massive_transfusion": ["hemoglobin", "lactate", "platelets", "inr", "heart_rate"],
    "status_epilepticus":  ["gcs", "lactate", "heart_rate", "temperature_c"],
    "thyroid_storm":       ["heart_rate", "temperature_c", "systolic_bp"],
}
```

### 3.1 ICU Initialization

```
Input: num_keys patients, seed
Output: initial state for all patients + ICU metadata

Steps:
1. Pick diagnosis mode: same-diagnosis (50%) or mixed-diagnosis (50%)
2. If same-diagnosis: pick one diagnosis from pool, generate num_keys patients with that diagnosis
3. If mixed-diagnosis: pick 3-5 diagnoses, distribute num_keys across them
4. Assign each patient:
   - patient_id: "Bed {N}" or room number
   - name: realistic patient name (from diverse name pool)
   - age, sex, weight_kg, height_cm
   - primary_diagnosis
   - admission_severity: mild / moderate / severe / critical
   - trajectory: improving / worsening / fluctuating (weighted by severity)
   - initial_state: all vitals and labs set to diagnosis-appropriate values
5. Pick icu_type from settings pool
6. Pick shift_start (determines time framing)
7. Generate attending/resident/nurse name pools
```

### 3.1.1 Patient Demographics Generation

**NO pediatric patients.** All generated patients are adults (age >= 18).

```python
DEMOGRAPHICS_SPEC = {
    "age": {
        "range": [18, 95],
        # Weighted toward elderly — ICU skews older
        # 18-30: 8%, 31-50: 17%, 51-65: 25%, 66-80: 35%, 81-95: 15%
        "bins": [(18, 30, 0.08), (31, 50, 0.17), (51, 65, 0.25), (66, 80, 0.35), (81, 95, 0.15)],
    },
    "sex": {
        # ICU admits are ~55% male, 45% female
        "options": ["male", "female"],
        "weights": [0.55, 0.45],
    },
    "weight_kg": {
        "male":   {"mean": 85, "std": 18, "min": 50, "max": 160},
        "female": {"mean": 72, "std": 16, "min": 40, "max": 140},
    },
    "height_cm": {
        "male":   {"mean": 177, "std": 8, "min": 155, "max": 200},
        "female": {"mean": 163, "std": 7, "min": 145, "max": 185},
    },
}

def generate_demographics(rng):
    """Generate a single patient's demographics."""
    sex = rng.choices(["male", "female"], weights=[0.55, 0.45])[0]

    # Sample age from weighted bins
    bin_choice = rng.choices(
        DEMOGRAPHICS_SPEC["age"]["bins"],
        weights=[b[2] for b in DEMOGRAPHICS_SPEC["age"]["bins"]]
    )[0]
    age = rng.randint(bin_choice[0], bin_choice[1])

    wt_spec = DEMOGRAPHICS_SPEC["weight_kg"][sex]
    weight_kg = round(max(wt_spec["min"], min(wt_spec["max"],
        rng.gauss(wt_spec["mean"], wt_spec["std"]))), 1)

    ht_spec = DEMOGRAPHICS_SPEC["height_cm"][sex]
    height_cm = round(max(ht_spec["min"], min(ht_spec["max"],
        rng.gauss(ht_spec["mean"], ht_spec["std"]))))

    return {"age": age, "sex": sex, "weight_kg": weight_kg, "height_cm": height_cm}
```

### 3.2 Patient State Initialization by Diagnosis

Each diagnosis has a characteristic initial state profile. **All 15 core attributes** are specified for every diagnosis. Values reflect typical admission presentation from the CLINICAL_TRAJECTORIES reference.

```python
DIAGNOSIS_PROFILES = {
    "sepsis": {
        "heart_rate":       {"range": (100, 135)},
        "systolic_bp":      {"range": (75, 100)},
        "diastolic_bp":     {"range": (40, 60)},
        "map":              {"range": (50, 70)},
        "respiratory_rate": {"range": (22, 35)},
        "spo2":             {"range": (88, 96)},
        "temperature_c":    {"range": (38.5, 40.2)},
        "potassium":        {"range": (3.5, 5.0)},
        "sodium":           {"range": (133, 142)},
        "creatinine":       {"range": (1.2, 3.5)},
        "lactate":          {"range": (2.5, 8.0)},
        "hemoglobin":       {"range": (9.0, 13.0)},
        "wbc":              {"range": (14.0, 28.0)},
        "platelets":        {"range": (80, 250)},
        "glucose":          {"range": (100, 200)},
        # Additional diagnosis-specific
        "procalcitonin":    {"range": (2.0, 50.0)},
        "vasopressor":      "norepinephrine",
        "vasopressor_dose": {"range": (0.05, 0.30)},
        "ventilator":       {"probability": 0.4},
    },
    "stemi": {
        "heart_rate":       {"range": (60, 110)},
        "systolic_bp":      {"range": (90, 170)},
        "diastolic_bp":     {"range": (55, 95)},
        "map":              {"range": (65, 115)},
        "respiratory_rate": {"range": (14, 24)},
        "spo2":             {"range": (92, 99)},
        "temperature_c":    {"range": (36.5, 37.5)},
        "potassium":        {"range": (3.5, 4.8)},
        "sodium":           {"range": (136, 144)},
        "creatinine":       {"range": (0.8, 1.5)},
        "lactate":          {"range": (0.8, 2.5)},
        "hemoglobin":       {"range": (11.0, 15.0)},
        "wbc":              {"range": (8.0, 15.0)},
        "platelets":        {"range": (150, 350)},
        "glucose":          {"range": (110, 250)},
        # Additional
        "troponin":         {"range": (0.5, 25.0)},
        "bnp":              {"range": (200, 2000)},
    },
    "dka": {
        "heart_rate":       {"range": (100, 130)},
        "systolic_bp":      {"range": (90, 130)},
        "diastolic_bp":     {"range": (50, 75)},
        "map":              {"range": (60, 90)},
        "respiratory_rate": {"range": (28, 40)},  # Kussmaul
        "spo2":             {"range": (95, 100)},
        "temperature_c":    {"range": (36.5, 37.8)},
        "potassium":        {"range": (4.5, 6.5)},  # falsely elevated despite total body depletion
        "sodium":           {"range": (125, 138)},   # pseudohyponatremia
        "creatinine":       {"range": (1.0, 2.5)},   # prerenal AKI
        "lactate":          {"range": (1.0, 3.0)},
        "hemoglobin":       {"range": (12.0, 16.0)},  # hemoconcentrated
        "wbc":              {"range": (10.0, 20.0)},   # stress leukocytosis
        "platelets":        {"range": (200, 400)},
        "glucose":          {"range": (350, 800)},
        # Additional
        "ph":               {"range": (7.05, 7.25)},
        "hco3":             {"range": (5, 14)},
        "anion_gap":        {"range": (18, 30)},
    },
    "ards": {
        "heart_rate":       {"range": (90, 125)},
        "systolic_bp":      {"range": (95, 135)},
        "diastolic_bp":     {"range": (55, 80)},
        "map":              {"range": (65, 95)},
        "respiratory_rate": {"range": (24, 38)},
        "spo2":             {"range": (85, 94)},
        "temperature_c":    {"range": (37.0, 39.0)},
        "potassium":        {"range": (3.5, 5.0)},
        "sodium":           {"range": (135, 145)},
        "creatinine":       {"range": (0.8, 2.0)},
        "lactate":          {"range": (1.0, 4.0)},
        "hemoglobin":       {"range": (9.5, 13.5)},
        "wbc":              {"range": (8.0, 20.0)},
        "platelets":        {"range": (120, 300)},
        "glucose":          {"range": (100, 180)},
        # Additional
        "fio2":             {"range": (60, 100)},
        "peep":             {"range": (10, 20)},
        "pf_ratio":         {"range": (80, 200)},
    },
    "gi_bleed": {
        "heart_rate":       {"range": (95, 140)},
        "systolic_bp":      {"range": (75, 110)},
        "diastolic_bp":     {"range": (40, 65)},
        "map":              {"range": (50, 75)},
        "respiratory_rate": {"range": (16, 26)},
        "spo2":             {"range": (94, 99)},
        "temperature_c":    {"range": (36.2, 37.2)},
        "potassium":        {"range": (3.3, 4.5)},
        "sodium":           {"range": (136, 144)},
        "creatinine":       {"range": (0.8, 2.0)},  # prerenal from hypovolemia
        "lactate":          {"range": (1.0, 5.0)},
        "hemoglobin":       {"range": (5.5, 8.5)},
        "wbc":              {"range": (6.0, 14.0)},
        "platelets":        {"range": (100, 300)},
        "glucose":          {"range": (90, 160)},
        # Additional
        "inr":              {"range": (1.0, 2.5)},
    },
    "copd_exacerbation": {
        "heart_rate":       {"range": (90, 120)},
        "systolic_bp":      {"range": (120, 160)},
        "diastolic_bp":     {"range": (70, 95)},
        "map":              {"range": (85, 115)},
        "respiratory_rate": {"range": (22, 32)},
        "spo2":             {"range": (85, 92)},  # target 88-92 per guidelines
        "temperature_c":    {"range": (36.8, 38.5)},
        "potassium":        {"range": (3.5, 4.8)},
        "sodium":           {"range": (135, 145)},
        "creatinine":       {"range": (0.7, 1.5)},
        "lactate":          {"range": (0.8, 2.0)},
        "hemoglobin":       {"range": (12.0, 16.0)},  # often polycythemia from chronic hypoxia
        "wbc":              {"range": (8.0, 16.0)},
        "platelets":        {"range": (180, 350)},
        "glucose":          {"range": (100, 200)},  # steroid-induced hyperglycemia
        # Additional
        "paco2":            {"range": (50, 75)},
        "ph":               {"range": (7.25, 7.38)},
    },
    "hemorrhagic_stroke": {
        "heart_rate":       {"range": (60, 100)},
        "systolic_bp":      {"range": (160, 220)},
        "diastolic_bp":     {"range": (90, 120)},
        "map":              {"range": (110, 150)},
        "respiratory_rate": {"range": (14, 24)},
        "spo2":             {"range": (94, 99)},
        "temperature_c":    {"range": (36.5, 38.0)},  # central fever may develop
        "potassium":        {"range": (3.5, 4.8)},
        "sodium":           {"range": (135, 148)},     # may have cerebral salt wasting
        "creatinine":       {"range": (0.7, 1.4)},
        "lactate":          {"range": (0.8, 2.0)},
        "hemoglobin":       {"range": (11.0, 15.0)},
        "wbc":              {"range": (7.0, 15.0)},
        "platelets":        {"range": (150, 350)},
        "glucose":          {"range": (100, 200)},     # stress hyperglycemia
        # Additional
        "gcs":              {"range": (6, 13)},
        "icp":              {"range": (15, 30)},
    },
    "post_cardiac_surgery": {
        "heart_rate":       {"range": (70, 100)},
        "systolic_bp":      {"range": (95, 130)},
        "diastolic_bp":     {"range": (55, 80)},
        "map":              {"range": (65, 95)},
        "respiratory_rate": {"range": (12, 20)},  # on ventilator initially
        "spo2":             {"range": (95, 100)},
        "temperature_c":    {"range": (35.5, 37.0)},  # hypothermic from bypass
        "potassium":        {"range": (3.5, 5.0)},
        "sodium":           {"range": (135, 145)},
        "creatinine":       {"range": (0.8, 1.8)},
        "lactate":          {"range": (1.5, 4.0)},    # bypass-related
        "hemoglobin":       {"range": (8.0, 11.0)},   # hemodilution from bypass
        "wbc":              {"range": (8.0, 16.0)},    # post-surgical leukocytosis
        "platelets":        {"range": (80, 180)},      # consumed on bypass
        "glucose":          {"range": (120, 220)},     # stress response
        # Additional
        "troponin":         {"range": (1.0, 15.0)},   # expected post-op elevation
        "chest_tube_output_ml_hr": {"range": (20, 100)},
    },
    "acute_liver_failure": {
        "heart_rate":       {"range": (90, 120)},
        "systolic_bp":      {"range": (85, 120)},
        "diastolic_bp":     {"range": (45, 70)},
        "map":              {"range": (55, 85)},       # hyperdynamic/vasodilated
        "respiratory_rate": {"range": (16, 28)},
        "spo2":             {"range": (94, 99)},
        "temperature_c":    {"range": (36.5, 38.0)},
        "potassium":        {"range": (3.2, 5.0)},
        "sodium":           {"range": (128, 140)},     # dilutional hyponatremia
        "creatinine":       {"range": (1.0, 3.5)},     # hepatorenal syndrome
        "lactate":          {"range": (2.0, 8.0)},     # impaired clearance
        "hemoglobin":       {"range": (8.0, 13.0)},
        "wbc":              {"range": (6.0, 18.0)},
        "platelets":        {"range": (40, 150)},      # reduced hepatic production
        "glucose":          {"range": (40, 120)},      # hypoglycemia from liver failure
        # Additional
        "inr":              {"range": (2.0, 6.0)},
        "bilirubin":        {"range": (5.0, 25.0)},
        "ammonia":          {"range": (80, 200)},
        "gcs":              {"range": (8, 14)},
    },
    "alcohol_withdrawal": {
        "heart_rate":       {"range": (100, 150)},
        "systolic_bp":      {"range": (140, 200)},
        "diastolic_bp":     {"range": (80, 110)},
        "map":              {"range": (100, 140)},
        "respiratory_rate": {"range": (18, 28)},
        "spo2":             {"range": (94, 99)},
        "temperature_c":    {"range": (37.5, 40.5)},   # autonomic storm
        "potassium":        {"range": (3.0, 3.8)},     # total body depletion
        "sodium":           {"range": (130, 142)},
        "creatinine":       {"range": (0.8, 2.0)},
        "lactate":          {"range": (1.0, 3.0)},
        "hemoglobin":       {"range": (10.0, 14.0)},
        "wbc":              {"range": (7.0, 14.0)},
        "platelets":        {"range": (80, 200)},      # chronic alcohol effect
        "glucose":          {"range": (60, 140)},      # hypoglycemia common
        # Additional
        "ciwa_score":       {"range": (15, 30)},
        "magnesium":        {"range": (1.2, 1.8)},
    },
    "aki": {
        "heart_rate":       {"range": (75, 110)},
        "systolic_bp":      {"range": (100, 150)},
        "diastolic_bp":     {"range": (60, 90)},
        "map":              {"range": (70, 105)},
        "respiratory_rate": {"range": (16, 26)},
        "spo2":             {"range": (94, 99)},
        "temperature_c":    {"range": (36.5, 37.8)},
        "potassium":        {"range": (5.0, 6.8)},
        "sodium":           {"range": (130, 142)},
        "creatinine":       {"range": (2.0, 8.0)},
        "lactate":          {"range": (0.8, 3.0)},
        "hemoglobin":       {"range": (9.0, 13.0)},
        "wbc":              {"range": (6.0, 14.0)},
        "platelets":        {"range": (150, 350)},
        "glucose":          {"range": (90, 180)},
        # Additional
        "bun":              {"range": (40, 100)},
        "urine_output_ml_hr": {"range": (5, 25)},
        "ph":               {"range": (7.20, 7.38)},
    },
    "pe": {
        "heart_rate":       {"range": (100, 130)},
        "systolic_bp":      {"range": (85, 130)},
        "diastolic_bp":     {"range": (50, 80)},
        "map":              {"range": (60, 95)},
        "respiratory_rate": {"range": (20, 32)},
        "spo2":             {"range": (85, 95)},
        "temperature_c":    {"range": (36.5, 37.8)},
        "potassium":        {"range": (3.5, 4.8)},
        "sodium":           {"range": (136, 144)},
        "creatinine":       {"range": (0.7, 1.5)},
        "lactate":          {"range": (1.0, 4.0)},
        "hemoglobin":       {"range": (11.0, 15.0)},
        "wbc":              {"range": (7.0, 14.0)},
        "platelets":        {"range": (150, 350)},
        "glucose":          {"range": (100, 180)},
        # Additional
        "troponin":         {"range": (0.1, 2.0)},    # RV strain
        "d_dimer":          {"range": (2000, 20000)},  # ng/mL
    },
    "pneumonia": {
        "heart_rate":       {"range": (95, 125)},
        "systolic_bp":      {"range": (100, 140)},
        "diastolic_bp":     {"range": (55, 80)},
        "map":              {"range": (65, 100)},
        "respiratory_rate": {"range": (22, 32)},
        "spo2":             {"range": (88, 95)},
        "temperature_c":    {"range": (38.5, 40.0)},
        "potassium":        {"range": (3.5, 5.0)},
        "sodium":           {"range": (132, 142)},
        "creatinine":       {"range": (0.8, 2.0)},
        "lactate":          {"range": (1.0, 3.5)},
        "hemoglobin":       {"range": (10.0, 14.0)},
        "wbc":              {"range": (14.0, 25.0)},
        "platelets":        {"range": (150, 350)},
        "glucose":          {"range": (100, 200)},
        # Additional
        "procalcitonin":    {"range": (0.5, 10.0)},
    },
    "drug_overdose": {
        # Opioid profile (default)
        "heart_rate":       {"range": (50, 75)},
        "systolic_bp":      {"range": (85, 120)},
        "diastolic_bp":     {"range": (50, 75)},
        "map":              {"range": (60, 85)},
        "respiratory_rate": {"range": (4, 10)},
        "spo2":             {"range": (60, 92)},
        "temperature_c":    {"range": (35.5, 37.0)},   # hypothermia from exposure
        "potassium":        {"range": (3.5, 5.5)},
        "sodium":           {"range": (134, 144)},
        "creatinine":       {"range": (0.8, 3.0)},     # rhabdomyolysis risk
        "lactate":          {"range": (1.0, 4.0)},
        "hemoglobin":       {"range": (11.0, 15.0)},
        "wbc":              {"range": (6.0, 14.0)},
        "platelets":        {"range": (150, 350)},
        "glucose":          {"range": (70, 160)},
        # Additional
        "gcs":              {"range": (3, 10)},
    },
    "pancreatitis": {
        "heart_rate":       {"range": (100, 125)},
        "systolic_bp":      {"range": (95, 135)},
        "diastolic_bp":     {"range": (55, 80)},
        "map":              {"range": (65, 95)},
        "respiratory_rate": {"range": (18, 28)},
        "spo2":             {"range": (93, 99)},
        "temperature_c":    {"range": (38.0, 39.5)},
        "potassium":        {"range": (3.5, 4.8)},
        "sodium":           {"range": (134, 144)},
        "creatinine":       {"range": (1.0, 2.5)},
        "lactate":          {"range": (1.0, 3.0)},
        "hemoglobin":       {"range": (10.0, 14.0)},
        "wbc":              {"range": (12.0, 22.0)},
        "platelets":        {"range": (150, 350)},
        "glucose":          {"range": (100, 200)},
        # Additional
        "lipase":           {"range": (500, 5000)},
        "calcium":          {"range": (7.0, 8.8)},     # hypocalcemia in severe
    },
    "anaphylaxis": {
        "heart_rate":       {"range": (110, 150)},     # compensatory tachycardia
        "systolic_bp":      {"range": (60, 90)},       # distributive shock
        "diastolic_bp":     {"range": (30, 55)},
        "map":              {"range": (40, 65)},
        "respiratory_rate": {"range": (22, 35)},       # bronchospasm + distress
        "spo2":             {"range": (80, 94)},       # laryngeal edema/bronchospasm
        "temperature_c":    {"range": (36.5, 37.5)},
        "potassium":        {"range": (3.5, 4.8)},
        "sodium":           {"range": (136, 144)},
        "creatinine":       {"range": (0.7, 1.3)},
        "lactate":          {"range": (2.0, 6.0)},    # from shock
        "hemoglobin":       {"range": (11.0, 15.0)},
        "wbc":              {"range": (7.0, 15.0)},
        "platelets":        {"range": (150, 350)},
        "glucose":          {"range": (100, 180)},     # stress
    },
    "meningitis": {
        "heart_rate":       {"range": (95, 130)},
        "systolic_bp":      {"range": (90, 140)},
        "diastolic_bp":     {"range": (50, 80)},
        "map":              {"range": (60, 100)},
        "respiratory_rate": {"range": (18, 28)},
        "spo2":             {"range": (94, 99)},
        "temperature_c":    {"range": (38.5, 40.5)},
        "potassium":        {"range": (3.5, 4.8)},
        "sodium":           {"range": (130, 142)},     # SIADH possible
        "creatinine":       {"range": (0.7, 1.5)},
        "lactate":          {"range": (1.0, 5.0)},    # meningococcal sepsis risk
        "hemoglobin":       {"range": (11.0, 15.0)},
        "wbc":              {"range": (12.0, 25.0)},
        "platelets":        {"range": (100, 300)},     # DIC in meningococcal
        "glucose":          {"range": (80, 160)},
        # Additional
        "gcs":              {"range": (8, 14)},        # median GCS 10 at admission
    },
    "cardiogenic_shock": {
        "heart_rate":       {"range": (90, 130)},
        "systolic_bp":      {"range": (70, 90)},
        "diastolic_bp":     {"range": (40, 60)},
        "map":              {"range": (50, 68)},
        "respiratory_rate": {"range": (20, 30)},
        "spo2":             {"range": (88, 96)},
        "temperature_c":    {"range": (36.0, 37.5)},
        "potassium":        {"range": (3.5, 5.5)},
        "sodium":           {"range": (130, 142)},
        "creatinine":       {"range": (1.2, 3.5)},    # cardiorenal
        "lactate":          {"range": (2.0, 10.0)},   # SCAI stage C-D
        "hemoglobin":       {"range": (10.0, 14.0)},
        "wbc":              {"range": (7.0, 15.0)},
        "platelets":        {"range": (120, 300)},
        "glucose":          {"range": (100, 220)},
        # Additional
        "troponin":         {"range": (1.0, 30.0)},
        "bnp":              {"range": (500, 5000)},
    },
    "hypertensive_emergency": {
        "heart_rate":       {"range": (80, 120)},
        "systolic_bp":      {"range": (200, 280)},
        "diastolic_bp":     {"range": (120, 160)},
        "map":              {"range": (145, 195)},
        "respiratory_rate": {"range": (16, 26)},
        "spo2":             {"range": (94, 99)},
        "temperature_c":    {"range": (36.5, 37.5)},
        "potassium":        {"range": (3.3, 4.8)},
        "sodium":           {"range": (135, 145)},
        "creatinine":       {"range": (1.0, 3.0)},    # end-organ damage
        "lactate":          {"range": (0.8, 3.0)},
        "hemoglobin":       {"range": (10.0, 15.0)},
        "wbc":              {"range": (6.0, 14.0)},
        "platelets":        {"range": (100, 300)},     # TMA possible
        "glucose":          {"range": (100, 200)},
        # Additional
        "troponin":         {"range": (0.02, 1.0)},   # demand ischemia
    },
    "major_burns": {
        "heart_rate":       {"range": (110, 140)},     # hypovolemic shock
        "systolic_bp":      {"range": (80, 120)},
        "diastolic_bp":     {"range": (50, 75)},
        "map":              {"range": (55, 85)},
        "respiratory_rate": {"range": (18, 28)},
        "spo2":             {"range": (92, 99)},
        "temperature_c":    {"range": (35.0, 36.5)},   # hypothermic early
        "potassium":        {"range": (4.0, 6.0)},     # cell lysis
        "sodium":           {"range": (135, 148)},
        "creatinine":       {"range": (0.8, 2.5)},
        "lactate":          {"range": (2.0, 6.0)},
        "hemoglobin":       {"range": (14.0, 18.0)},   # hemoconcentrated
        "wbc":              {"range": (10.0, 25.0)},
        "platelets":        {"range": (100, 250)},
        "glucose":          {"range": (120, 250)},     # stress hyperglycemia
    },
    "massive_transfusion": {
        "heart_rate":       {"range": (110, 150)},     # hemorrhagic shock
        "systolic_bp":      {"range": (60, 90)},
        "diastolic_bp":     {"range": (30, 55)},
        "map":              {"range": (35, 65)},
        "respiratory_rate": {"range": (22, 35)},
        "spo2":             {"range": (85, 96)},
        "temperature_c":    {"range": (34.5, 36.0)},   # hypothermic (lethal triad)
        "potassium":        {"range": (3.5, 5.5)},
        "sodium":           {"range": (134, 144)},
        "creatinine":       {"range": (0.8, 2.5)},
        "lactate":          {"range": (4.0, 12.0)},
        "hemoglobin":       {"range": (5.0, 8.0)},
        "wbc":              {"range": (8.0, 18.0)},
        "platelets":        {"range": (50, 150)},      # dilutional thrombocytopenia
        "glucose":          {"range": (100, 200)},
        # Additional
        "inr":              {"range": (1.5, 3.5)},
        "fibrinogen":       {"range": (80, 200)},      # mg/dL
        "ionized_calcium":  {"range": (0.85, 1.10)},   # citrate toxicity
    },
    "status_epilepticus": {
        "heart_rate":       {"range": (110, 150)},
        "systolic_bp":      {"range": (130, 200)},
        "diastolic_bp":     {"range": (80, 110)},
        "map":              {"range": (95, 135)},
        "respiratory_rate": {"range": (16, 30)},
        "spo2":             {"range": (85, 96)},
        "temperature_c":    {"range": (37.5, 40.0)},
        "potassium":        {"range": (3.5, 5.5)},
        "sodium":           {"range": (134, 144)},
        "creatinine":       {"range": (0.8, 2.0)},
        "lactate":          {"range": (4.0, 15.0)},   # 8.7x baseline from seizure
        "hemoglobin":       {"range": (11.0, 15.0)},
        "wbc":              {"range": (8.0, 18.0)},
        "platelets":        {"range": (150, 350)},
        "glucose":          {"range": (80, 200)},
        # Additional
        "gcs":              {"range": (3, 8)},         # ictal/postictal
        "ck":               {"range": (500, 10000)},   # rhabdomyolysis risk
    },
    "thyroid_storm": {
        "heart_rate":       {"range": (140, 180)},     # classic
        "systolic_bp":      {"range": (130, 180)},     # wide pulse pressure
        "diastolic_bp":     {"range": (50, 70)},       # low diastolic
        "map":              {"range": (75, 105)},
        "respiratory_rate": {"range": (20, 30)},
        "spo2":             {"range": (94, 99)},
        "temperature_c":    {"range": (39.0, 41.0)},   # 102-106°F classic
        "potassium":        {"range": (3.0, 4.5)},
        "sodium":           {"range": (135, 145)},
        "creatinine":       {"range": (0.7, 1.5)},
        "lactate":          {"range": (1.0, 3.0)},
        "hemoglobin":       {"range": (11.0, 15.0)},
        "wbc":              {"range": (7.0, 14.0)},
        "platelets":        {"range": (150, 350)},
        "glucose":          {"range": (120, 250)},     # hyperglycemia from catecholamine surge
    },
}
```

### 3.3 Event Types

The ICU narrative is a sequence of clinical events. Each event updates 1-3 patients' states.

```
EVENT TYPES:
├── vital_sign_check       — routine q1h vital sign documentation
├── lab_result             — lab values return (BMP, CBC, ABG, lactate, etc.)
├── medication_change      — new med started, dose titrated, med discontinued
├── ventilator_change      — mode change, FiO2/PEEP adjustment, extubation
├── procedure              — central line, arterial line, chest tube, intubation, LP, etc.
├── imaging_result         — CXR, CT, echo, ultrasound findings
├── clinical_deterioration — rapid response, code blue, acute decline
├── clinical_improvement   — milestone (off vasopressors, extubated, tolerating diet)
├── consult_note           — specialist sees patient, provides recommendations
├── nursing_assessment     — focused assessment (neuro check, wound, I&O summary)
├── transfusion            — blood products given, response documented
├── code_status_change     — goals of care discussion, code status updated
├── transfer_event         — patient transferred to/from ICU
└── shift_handoff          — summary of patient status at shift change
```

### 3.3.1 Trajectory Interpolation Engine

The trajectory system converts sparse clinical waypoints (e.g., "Hour 0: lactate 6.2, Hour 12: lactate 3.2") into continuous values at any query time.

```python
# Noise sigma per attribute — physiological measurement variability
ATTRIBUTE_NOISE_SIGMA = {
    "heart_rate":           3.0,     # ±3 bpm
    "systolic_bp":          5.0,     # ±5 mmHg
    "diastolic_bp":         3.0,     # ±3 mmHg
    "map":                  3.0,     # ±3 mmHg
    "respiratory_rate":     1.5,     # ±1.5 breaths/min
    "spo2":                 1.0,     # ±1%
    "temperature_c":        0.15,    # ±0.15°C
    "potassium":            0.1,     # ±0.1 mEq/L
    "sodium":               1.0,     # ±1 mEq/L
    "creatinine":           0.1,     # ±0.1 mg/dL
    "lactate":              0.2,     # ±0.2 mmol/L
    "hemoglobin":           0.15,    # ±0.15 g/dL
    "wbc":                  0.5,     # ±0.5 k/uL
    "platelets":            8.0,     # ±8 k
    "glucose":              8.0,     # ±8 mg/dL
    "troponin":             0.05,    # ±0.05 ng/mL
    "bilirubin":            0.2,     # ±0.2 mg/dL
    "inr":                  0.1,     # ±0.1
    "procalcitonin":        0.2,     # ±0.2 ng/mL
    "gcs":                  0.0,     # integer, no noise (use ±1 step changes)
    "rass":                 0.0,     # integer, no noise
    "norepinephrine_dose":  0.01,    # ±0.01 mcg/kg/min
    "fio2":                 2.0,     # ±2%
    "peep":                 0.0,     # integer, no noise (discrete steps)
    "urine_output_ml_hr":   5.0,    # ±5 mL/hr
}


def build_trajectory_waypoints(diagnosis, trajectory_type, total_hours, rng):
    """
    Build time-value waypoints for a patient based on diagnosis and trajectory.

    Args:
        diagnosis: str — key from DIAGNOSIS_REGISTRY
        trajectory_type: "improving" | "worsening" | "fluctuating"
        total_hours: int — total narrative duration in hours
        rng: Random instance

    Returns:
        dict mapping attribute_name -> list of (hour, value) tuples
    """
    profile = DIAGNOSIS_PROFILES[diagnosis]
    waypoints = {}

    for attr_name, attr_spec in profile.items():
        if not isinstance(attr_spec, dict) or "range" not in attr_spec:
            continue  # skip non-numeric fields like "vasopressor"

        lo, hi = attr_spec["range"]

        if trajectory_type == "improving":
            # Start at the sicker end of the range, trend toward normal
            initial = rng.uniform(lo + 0.6 * (hi - lo), hi)
            final = rng.uniform(lo, lo + 0.4 * (hi - lo))
        elif trajectory_type == "worsening":
            # Start at the better end, trend toward sicker
            initial = rng.uniform(lo, lo + 0.4 * (hi - lo))
            final = rng.uniform(lo + 0.6 * (hi - lo), hi)
        else:  # fluctuating
            initial = rng.uniform(lo, hi)
            # Create a mid-point excursion then partial return
            mid_value = rng.uniform(lo, hi)
            final = rng.uniform(lo, hi)
            mid_hour = total_hours * rng.uniform(0.3, 0.6)
            waypoints[attr_name] = [
                (0, initial),
                (mid_hour, mid_value),
                (total_hours, final),
            ]
            continue

        waypoints[attr_name] = [
            (0, initial),
            (total_hours, final),
        ]

    return waypoints


def interpolate_value(waypoints_for_attr, query_hour, attr_name, rng):
    """
    Get an attribute value at a specific hour via linear interpolation + noise.

    Args:
        waypoints_for_attr: list of (hour, value) tuples, sorted by hour
        query_hour: float — the hour to query
        attr_name: str — attribute name (for noise lookup and clamping)
        rng: Random instance

    Returns:
        float or int — the interpolated value with noise applied
    """
    # Clamp to waypoint range
    if query_hour <= waypoints_for_attr[0][0]:
        base = waypoints_for_attr[0][1]
    elif query_hour >= waypoints_for_attr[-1][0]:
        base = waypoints_for_attr[-1][1]
    else:
        # Find bracketing waypoints
        for i in range(len(waypoints_for_attr) - 1):
            t0, v0 = waypoints_for_attr[i]
            t1, v1 = waypoints_for_attr[i + 1]
            if t0 <= query_hour <= t1:
                frac = (query_hour - t0) / (t1 - t0)
                base = v0 + frac * (v1 - v0)
                break

    # Apply Gaussian noise
    sigma = ATTRIBUTE_NOISE_SIGMA.get(attr_name, 0.0)
    if sigma > 0:
        base += rng.gauss(0, sigma)

    # Apply attribute-specific clamping and rounding
    attr_spec = next((a for a in TRACKABLE_ATTRIBUTES if a["name"] == attr_name), None)
    if attr_spec:
        lo, hi = attr_spec["range"]
        base = max(lo, min(hi, base))
        if attr_spec["type"] == "int":
            base = round(base)
        else:
            base = round(base, attr_spec["precision"])

    return base
```

**Trajectory switching:** A patient may switch from improving to worsening (or vice versa) mid-narrative. This is implemented by splicing two trajectory segments:

```python
def build_switching_trajectory(diagnosis, initial_type, switch_hour, total_hours, rng):
    """Build a trajectory that switches direction at switch_hour."""
    first_half = build_trajectory_waypoints(diagnosis, initial_type, switch_hour, rng)
    second_type = "worsening" if initial_type == "improving" else "improving"
    second_half = build_trajectory_waypoints(diagnosis, second_type, total_hours - switch_hour, rng)

    merged = {}
    for attr in first_half:
        # Use the end of first_half as start of second_half
        transition_value = interpolate_value(first_half[attr], switch_hour, attr, rng)
        second_waypoints = [(switch_hour, transition_value)]
        for t, v in second_half.get(attr, []):
            if t > 0:  # skip the initial point
                second_waypoints.append((switch_hour + t, v))
        merged[attr] = first_half[attr] + second_waypoints[1:]  # avoid duplicate at switch
    return merged
```

### 3.3.2 Physiological Correlation Formulas

After interpolation, apply these correlation adjustments **in priority order**. Each formula is exact and implementable. When multiple adjustments conflict, use the **maximum effect** principle: take the largest absolute adjustment.

```python
def apply_physiological_correlations(state, patient):
    """
    Mutate state in-place to enforce physiological correlations.
    Call AFTER trajectory interpolation, BEFORE rendering.

    Args:
        state: dict of current attribute values for this patient
        patient: PatientState object with medication/device info
    """
    adjustments = {"heart_rate": 0.0, "respiratory_rate": 0.0, "map": 0.0}

    # ── Heart Rate Adjustments ──

    # 1. Fever effect: +10 bpm per °C above 37.0
    if "temperature_c" in state and state["temperature_c"] > 37.0:
        adjustments["heart_rate"] += 10 * (state["temperature_c"] - 37.0)

    # 2. Hypotension compensation: if MAP < 65, compensatory tachycardia
    if "map" in state and state["map"] < 65:
        adjustments["heart_rate"] += uniform(20, 40)  # use rng.uniform

    # 3. Anemia compensation: if Hgb < 8, compensatory tachycardia
    if "hemoglobin" in state and state["hemoglobin"] < 8.0:
        adjustments["heart_rate"] += (8.0 - state["hemoglobin"]) * 8

    # 4. Beta-blocker modifier: blunts all HR adjustments
    if patient.on_beta_blocker:
        adjustments["heart_rate"] *= 0.3

    # 5. Pain effect: +5 bpm per pain point above 5
    if "pain_score" in state and state["pain_score"] > 5:
        adjustments["heart_rate"] += (state["pain_score"] - 5) * 5

    # Apply cumulative HR adjustment
    if "heart_rate" in state:
        state["heart_rate"] = round(state["heart_rate"] + adjustments["heart_rate"])
        state["heart_rate"] = max(30, min(220, state["heart_rate"]))

    # ── Blood Pressure / MAP Adjustments ──

    # 6. Vasopressor-MAP relationship
    if patient.norepinephrine_dose > 0:
        # MAP increases ~20 mmHg per 0.5 mcg/kg/min of norepinephrine
        vasopressor_map = 55 + (patient.norepinephrine_dose / 0.5) * 20
        # Blend with underlying state (vasopressor doesn't fully override physiology)
        state["map"] = round(0.4 * state.get("map", 65) + 0.6 * vasopressor_map)
        state["map"] = max(35, min(130, state["map"]))

    # 7. PEEP-hemodynamic effect: high PEEP reduces preload
    if patient.peep and patient.peep > 12:
        map_reduction = (patient.peep - 12) * 2  # -2 mmHg per cmH2O above 12
        state["map"] = max(35, state.get("map", 65) - map_reduction)

    # 8. Recalculate SBP/DBP from MAP if MAP was adjusted
    if "map" in state:
        # MAP ≈ DBP + 1/3(SBP - DBP), so if MAP changed, adjust SBP/DBP proportionally
        if "systolic_bp" in state and "diastolic_bp" in state:
            pulse_pressure = state["systolic_bp"] - state["diastolic_bp"]
            state["diastolic_bp"] = round(state["map"] - pulse_pressure / 3)
            state["systolic_bp"] = round(state["diastolic_bp"] + pulse_pressure)

    # ── Respiratory Adjustments ──

    # 9. Acidosis → Kussmaul breathing
    if "ph" in state and state.get("ph", 7.40) < 7.35:
        adjustments["respiratory_rate"] += (7.35 - state["ph"]) * 40

    # 10. Hypoxemia → tachypnea
    if "spo2" in state and state["spo2"] < 92:
        adjustments["respiratory_rate"] += (92 - state["spo2"]) * 1.5

    if "respiratory_rate" in state:
        state["respiratory_rate"] = round(state["respiratory_rate"] + adjustments["respiratory_rate"])
        state["respiratory_rate"] = max(4, min(45, state["respiratory_rate"]))

    # ── Oxygenation ──

    # 11. FiO2-SpO2 relationship
    if patient.fio2 is not None and "spo2" in state:
        # lung_function_factor: 1.0 = normal, 0.3 = severe ARDS
        lff = patient.lung_function_factor  # set from diagnosis (ARDS=0.3-0.5, normal=0.8-1.0)
        expected_spo2 = min(100, 85 + (patient.fio2 - 21) * 0.2 * lff)
        # Blend: 60% from trajectory, 40% from FiO2 model
        state["spo2"] = round(0.6 * state["spo2"] + 0.4 * expected_spo2)
        state["spo2"] = max(60, min(100, state["spo2"]))

    # ── Renal-Electrolyte Correlations ──

    # 12. Renal failure → hyperkalemia
    if "creatinine" in state and state["creatinine"] > 2.0:
        k_adjustment = (state["creatinine"] - 1.0) * 0.3
        state["potassium"] = round(state.get("potassium", 4.0) + k_adjustment, 1)
        state["potassium"] = max(2.0, min(7.5, state["potassium"]))

    # 13. Acidosis → potassium shift (every 0.1 pH drop → K+ rises ~0.5 mEq/L)
    if "ph" in state and state.get("ph", 7.40) < 7.35:
        ph_k_shift = (7.40 - state["ph"]) / 0.1 * 0.5
        state["potassium"] = round(state.get("potassium", 4.0) + ph_k_shift * 0.3, 1)
        state["potassium"] = max(2.0, min(7.5, state["potassium"]))

    # ── Hematologic Correlations ──

    # 14. Transfusion effect: each unit PRBC raises Hgb ~1 g/dL
    # (handled in transfusion event handler, not here)

    # 15. Liver failure → coagulopathy
    if "bilirubin" in state and state.get("bilirubin", 1.0) > 5.0:
        if "inr" in state:
            state["inr"] = round(max(state["inr"], 1.0 + (state["bilirubin"] - 3.0) * 0.3), 1)
        if "platelets" in state:
            state["platelets"] = min(state["platelets"], round(250 - state["bilirubin"] * 8))
```

**Conflict resolution:** When multiple formulas adjust the same attribute:
1. Compute each adjustment independently
2. Sum all additive adjustments (e.g., fever + hypotension both increase HR)
3. Apply multiplicative modifiers last (e.g., beta-blocker dampens total HR adjustment)
4. Clamp to physiological range from TRACKABLE_ATTRIBUTES

### 3.4 Event Generation (the state machine)

```python
def generate_next_event(state, icu_hour, time_of_day, rng):
    """Generate a plausible next clinical event based on current state and timing."""

    # Time of day affects event mix
    if time_of_day == "night_shift":  # 19:00-07:00
        weights = {
            "vital_sign_check": 30,        # routine checks continue
            "lab_result": 15,               # 04:00-05:00 AM labs result
            "medication_change": 10,        # fewer med changes at night
            "ventilator_change": 5,         # less tweaking at night
            "nursing_assessment": 15,       # q4h assessments
            "clinical_deterioration": 5,    # can happen anytime
            "clinical_improvement": 3,
            "shift_handoff": 2,             # 19:00 and 07:00
            # Fewer procedures, consults, imaging at night
            "procedure": 2,
            "consult_note": 1,
            "imaging_result": 3,
            "transfusion": 4,
            "transfer_event": 2,
            "code_status_change": 1,
        }
    elif time_of_day == "day_shift_am":  # 07:00-12:00
        weights = {
            "vital_sign_check": 20,
            "lab_result": 20,               # AM labs result 06:00-09:00
            "medication_change": 15,        # attending rounds → changes
            "ventilator_change": 10,        # SBT trials in morning
            "procedure": 8,                 # scheduled procedures
            "imaging_result": 10,           # morning CXRs
            "consult_note": 8,              # consults round in AM
            "nursing_assessment": 10,
            "clinical_deterioration": 3,
            "clinical_improvement": 5,
            "shift_handoff": 2,             # 07:00 handoff
            "transfusion": 3,
            "transfer_event": 3,
            "code_status_change": 2,
        }
    else:  # day_shift_pm (12:00-19:00)
        weights = {
            "vital_sign_check": 25,
            "lab_result": 10,               # repeat labs if ordered
            "medication_change": 12,
            "ventilator_change": 8,
            "procedure": 5,
            "imaging_result": 8,
            "consult_note": 5,
            "nursing_assessment": 12,
            "clinical_deterioration": 4,
            "clinical_improvement": 4,
            "transfusion": 3,
            "transfer_event": 3,
            "code_status_change": 2,
        }

    # Patient acuity modifies weights:
    # Unstable patients → more vital checks, more medication changes
    # Improving patients → more improvement milestones, weaning events
    # Worsening patients → more deterioration events, more lab checks

    # Filter impossible events:
    # Can't extubate a patient not on ventilator
    # Can't wean vasopressors if not on any
    # Transferred/deceased patients generate no events
    # Can't do shift handoff if not at shift change time

    event_type = weighted_random_choice(weights, rng)
    event = create_event(event_type, state, icu_hour, rng)
    return event
```

### 3.4.1 Time Progression Model

The ICU narrative clock advances in real-time hours. Unlike game-based domains, ICU time is continuous and event frequency varies by acuity and time of day.

```python
# Base time interval (in hours) between events, by event type
EVENT_TIME_INTERVALS = {
    "vital_sign_check":        {"base": 1.0,  "range": (0.25, 2.0)},
    "lab_result":              {"base": 4.0,  "range": (2.0, 12.0)},
    "medication_change":       {"base": 2.0,  "range": (0.5, 6.0)},
    "ventilator_change":       {"base": 3.0,  "range": (1.0, 8.0)},
    "procedure":               {"base": 4.0,  "range": (2.0, 12.0)},
    "imaging_result":          {"base": 6.0,  "range": (3.0, 12.0)},
    "clinical_deterioration":  {"base": 6.0,  "range": (0.1, 12.0)},   # can be sudden
    "clinical_improvement":    {"base": 8.0,  "range": (4.0, 24.0)},   # gradual
    "consult_note":            {"base": 6.0,  "range": (3.0, 12.0)},
    "nursing_assessment":      {"base": 4.0,  "range": (2.0, 8.0)},
    "transfusion":             {"base": 4.0,  "range": (2.0, 6.0)},
    "code_status_change":      {"base": 12.0, "range": (6.0, 48.0)},
    "transfer_event":          {"base": 12.0, "range": (6.0, 48.0)},
    "shift_handoff":           {"base": 12.0, "range": (12.0, 12.0)},  # fixed at shift boundaries
}

# Acuity modifier: unstable patients get events faster
ACUITY_TIME_MODIFIER = {
    "critical":  0.5,   # half the base interval
    "severe":    0.7,
    "moderate":  1.0,   # base interval
    "mild":      1.5,   # 1.5x base interval
    "stable":    2.0,
}

# Time-of-day effects
TIME_OF_DAY_MODIFIER = {
    # Night shift (19:00-07:00): fewer elective events, same emergency frequency
    "night_shift":    {"vital_sign_check": 1.0, "lab_result": 1.5, "procedure": 2.5,
                       "consult_note": 3.0, "imaging_result": 1.5, "medication_change": 1.3,
                       "clinical_deterioration": 1.0, "clinical_improvement": 1.5},
    # AM rounds (07:00-12:00): high activity, decisions being made
    "day_shift_am":   {"vital_sign_check": 0.8, "lab_result": 0.6, "procedure": 0.7,
                       "consult_note": 0.6, "imaging_result": 0.7, "medication_change": 0.7,
                       "clinical_deterioration": 1.0, "clinical_improvement": 0.8},
    # PM (12:00-19:00): moderate activity
    "day_shift_pm":   {"vital_sign_check": 1.0, "lab_result": 1.2, "procedure": 1.0,
                       "consult_note": 1.0, "imaging_result": 1.0, "medication_change": 1.0,
                       "clinical_deterioration": 1.0, "clinical_improvement": 1.0},
}

def compute_time_advance(event_type, acuity, time_of_day, rng):
    """
    Compute hours until next event.

    Args:
        event_type: str — the event type that just occurred
        acuity: str — patient acuity level ("critical"..."stable")
        time_of_day: str — "night_shift" | "day_shift_am" | "day_shift_pm"
        rng: Random instance

    Returns:
        float — hours to advance
    """
    spec = EVENT_TIME_INTERVALS[event_type]
    base = spec["base"]
    lo, hi = spec["range"]

    # Apply acuity modifier
    acuity_mod = ACUITY_TIME_MODIFIER.get(acuity, 1.0)

    # Apply time-of-day modifier
    tod_mods = TIME_OF_DAY_MODIFIER.get(time_of_day, {})
    tod_mod = tod_mods.get(event_type, 1.0)

    adjusted_base = base * acuity_mod * tod_mod
    adjusted_lo = lo * acuity_mod * tod_mod
    adjusted_hi = hi * acuity_mod * tod_mod

    dt = rng.uniform(adjusted_lo, adjusted_hi)
    # Minimum gap: 0.1 hours (6 minutes) between any two events
    dt = max(0.1, dt)
    return round(dt, 2)


def get_time_of_day(icu_hour):
    """Convert absolute ICU hour to time-of-day category."""
    hour_of_day = icu_hour % 24
    if 7 <= hour_of_day < 12:
        return "day_shift_am"
    elif 12 <= hour_of_day < 19:
        return "day_shift_pm"
    else:
        return "night_shift"
```

### 3.5 Event Handlers (Constraints)

Each event handler enforces clinical rules and physiological correlations:

```
VITAL_SIGN_CHECK:
  - Pick a living, non-transferred patient
  - Advance time by 30-120 min (depending on acuity — unstable q15-30min, stable q1-2h)
  - Update: heart_rate, systolic_bp, diastolic_bp, map, respiratory_rate, spo2, temperature
  - Constraints:
    - Values must follow physiological correlations:
      - Fever (+1°C) → HR increases ~10 bpm
      - Hypotension (MAP <65) → compensatory tachycardia (+20-40 bpm)
      - Unless on beta-blocker (blunts tachycardia)
    - Values drift from previous check by realistic amounts:
      - HR: ±5-15 bpm per hour (stable), ±20-40 (acute event)
      - SBP: ±5-15 mmHg per hour (stable), ±20-40 (acute event)
      - SpO2: ±1-3% per hour (stable), ±5-10 (acute event)
      - Temp: ±0.2-0.5°C per 4h
    - Follow patient trajectory (improving → vitals trend toward normal; worsening → trend away)
    - Vasopressor-dependent patients: MAP correlates with dose (titrated to target)

LAB_RESULT:
  - Pick a patient with pending labs (typically all patients get AM labs)
  - Update: relevant lab panel values (BMP, CBC, LFTs, coag, ABG)
  - Constraints:
    - Labs follow diagnosis-specific trajectories:
      - Sepsis improving: lactate 6→4→2.5→1.8 over 24-48h
      - DKA improving: glucose 600→350→200→150, pH 7.10→7.20→7.30→7.38
      - GI bleed: Hgb may drop 1-2 g/dL per 6h if actively bleeding
    - Physiological correlations:
      - Rising creatinine → rising potassium, rising BUN
      - Worsening liver → rising INR, rising bilirubin, falling albumin
      - Acidosis → low pH, low HCO3, compensatory low PaCO2
    - Lab results have realistic precision (creatinine to 0.1, potassium to 0.1, lactate to 0.1)
    - Critical values trigger immediate events (K >6.5, Hgb <7, troponin elevation)

MEDICATION_CHANGE:
  - Pick a patient
  - Update: medication doses, possibly resulting vital sign changes
  - Types:
    - Vasopressor titration: norepinephrine 0.15 → 0.20 mcg/kg/min (MAP response in next vital check)
    - Vasopressor wean: 0.10 → 0.05 → off (milestone if completely off)
    - Antibiotic change: escalate (broader spectrum) or de-escalate (narrower based on cultures)
    - Sedation change: propofol rate adjusted, sedation vacation
    - Insulin drip titration: rate adjusted based on glucose
    - New medication: diuretic started for fluid overload, anticoagulation started
  - Constraints:
    - Can't start vasopressor without hypotension indication
    - Vasopressor dose changes are incremental (0.02-0.05 mcg/kg/min per step)
    - Antibiotics need to match diagnosis
    - Sedation adjustments correlate with RASS score

VENTILATOR_CHANGE:
  - Pick an intubated patient
  - Update: ventilator settings (FiO2, PEEP, mode, tidal volume, rate)
  - Types:
    - FiO2 wean: 80→60→50→40% (target SpO2 92-96%)
    - PEEP wean: 14→12→10→8→5 cmH2O
    - Mode change: AC/VC → PSV (weaning), PSV → SBT
    - Extubation: if passing SBT criteria
    - Escalation: FiO2 increase, PEEP increase, prone positioning
  - Constraints:
    - FiO2 and PEEP follow ARDSnet table
    - Can't wean FiO2 below SpO2 92% (will cause desaturation)
    - PEEP changes affect hemodynamics (high PEEP → decreased preload → possible hypotension)
    - Extubation requires: FiO2 ≤40%, PEEP ≤8, adequate mentation, passing SBT
    - Must wean FiO2 before PEEP (per clinical practice)

PROCEDURE:
  - Pick a patient needing a procedure
  - Update: procedure_status, possibly new monitoring lines
  - Types:
    - Central line placement → now has central access
    - Arterial line → now has continuous BP monitoring
    - Chest tube → new output to monitor
    - Intubation → now on ventilator
    - Bronchoscopy → possible FiO2 change during/after
    - Dialysis initiation → electrolyte changes over hours
    - LP → CSF results pending
  - Constraints:
    - Procedures require consent (or emergency exception)
    - Post-procedure vitals q15min × 4, then q1h
    - Complications possible but rare (pneumothorax from central line, etc.)

CLINICAL_DETERIORATION:
  - Pick a patient (weighted toward worsening trajectory patients)
  - Update: acute change in vitals, lab values, or clinical status
  - Types:
    - New-onset arrhythmia (afib RVR → HR jumps to 140-170)
    - Acute desaturation (SpO2 drops 10-15% → mucus plug, pneumothorax, PE)
    - Hemodynamic collapse (MAP drops <55 → vasopressor started or escalated)
    - Acute bleeding (Hgb drop, tachycardia, hypotension)
    - Seizure (altered mentation, needs intervention)
    - Fever spike (temp jumps 1-2°C → new infection, drug reaction)
    - Code blue (cardiac arrest → CPR, ACLS protocol)
  - Constraints:
    - Deterioration events must be physiologically coherent
    - Each deterioration triggers appropriate response events
    - Code blue is rare (1-2% of ICU patients) but possible

CLINICAL_IMPROVEMENT:
  - Pick a patient (weighted toward improving trajectory patients)
  - Update: milestone achieved
  - Types:
    - Off vasopressors (norepinephrine → off, MAP maintaining ≥65)
    - Extubated (passed SBT, now on nasal cannula / room air)
    - Lactate cleared (<2.0 mmol/L)
    - Tolerating diet (NPO → clear liquids → regular)
    - Mental status improving (GCS increase, following commands)
    - Transferred to floor (leaving ICU)
  - Constraints:
    - Milestones must be clinically appropriate (can't be off pressors while still in shock)
    - Improvement is gradual (no instant recovery from multiorgan failure)

CONSULT_NOTE:
  - Pick a patient with a relevant consult
  - Update: specialist recommendation → may trigger medication or procedure changes
  - Types: cardiology, pulmonology, nephrology, surgery, neurology, infectious disease, palliative care
  - Constraints: consult type matches diagnosis (ID for sepsis, nephrology for AKI, etc.)

NURSING_ASSESSMENT:
  - Pick a patient
  - Update: focused assessment findings
  - Covers: neuro check (GCS, pupils, RASS), skin (wounds, lines), I&O summary, pain, psychosocial
  - Constraints: assessment frequency matches acuity (neuro checks q1h for neuro patients, q4h standard)

TRANSFUSION:
  - Pick a patient with Hgb <7 (or <8 if cardiac) or platelets <20k or INR >5 with bleeding
  - Update: hemoglobin increases ~1 g/dL per unit PRBC, platelets increase ~30k per unit
  - Constraints: transfusion takes 2-4h per unit, vitals monitored q15min during

SHIFT_HANDOFF:
  - Occurs at shift change times (07:00 and 19:00)
  - Summary of ALL patients — this is a natural "panoramic" event that touches many entities
  - Structured: diagnosis, current vitals, overnight events, plan for next shift
  - Constraints: only at shift boundary times

TRANSFER_EVENT:
  - Patient transfers out of ICU (to floor) or into ICU (from ER, floor, OR)
  - Update: patient leaves (no more events) or new patient arrives (initializes state)
  - Constraints: transfer out only for clinically stable patients
```

### 3.5.1 Event Handler Specifications

Every event handler returns an `EventDict`:

```python
EventDict = {
    "event_type": str,              # e.g., "vital_sign_check"
    "timestamp_hours": float,       # absolute ICU hour
    "patient_id": str,              # e.g., "Bed 3"
    "patient_name": str,            # e.g., "Mrs. Liu"
    "state_mutations": dict,        # {attr_name: new_value} — fields that changed
    "narrative_text": str,          # rendered text for this event (filled by renderer, not handler)
    "tracked_attr_mentioned": bool, # True if this event mentions the tracked attribute's new value
    "metadata": dict,               # event-type-specific extras
}
```

#### Handler 1: `vital_sign_check`

```python
def handle_vital_sign_check(patient, icu_hour, rng) -> EventDict:
    """
    Routine vital sign documentation.

    Parameters:
        patient: PatientState — the patient being checked
        icu_hour: float — current narrative hour
        rng: Random — RNG instance

    Preconditions:
        - Patient is alive and not transferred out
        - At least 0.25 hours since last vital check for this patient

    State mutations:
        - heart_rate: interpolated from trajectory + correlations
        - systolic_bp: interpolated + correlations
        - diastolic_bp: interpolated + correlations
        - map: recalculated as diastolic_bp + (systolic_bp - diastolic_bp) / 3
        - respiratory_rate: interpolated + correlations
        - spo2: interpolated + FiO2 correlation
        - temperature_c: interpolated (changes slowly, q4h significant moves)

    Returns:
        EventDict with state_mutations for all 7 vital signs
    """
    # 1. Interpolate each vital from trajectory waypoints
    new_vitals = {}
    for attr in ["heart_rate", "systolic_bp", "diastolic_bp", "map",
                 "respiratory_rate", "spo2", "temperature_c"]:
        if attr in patient.waypoints:
            new_vitals[attr] = interpolate_value(
                patient.waypoints[attr], icu_hour, attr, rng
            )

    # 2. Apply physiological correlations (mutates new_vitals in-place)
    apply_physiological_correlations(new_vitals, patient)

    # 3. Enforce realistic drift from previous check
    prev = patient.current_state
    DRIFT_LIMITS = {
        "heart_rate": 15, "systolic_bp": 15, "diastolic_bp": 10,
        "map": 10, "respiratory_rate": 5, "spo2": 3, "temperature_c": 0.5,
    }
    for attr, max_drift in DRIFT_LIMITS.items():
        if attr in new_vitals and attr in prev:
            delta = new_vitals[attr] - prev[attr]
            hours_elapsed = icu_hour - patient.last_vital_check_hour
            max_allowed = max_drift * max(1, hours_elapsed)
            if abs(delta) > max_allowed:
                new_vitals[attr] = prev[attr] + max_allowed * (1 if delta > 0 else -1)

    # 4. Recalculate MAP from SBP/DBP
    if "systolic_bp" in new_vitals and "diastolic_bp" in new_vitals:
        new_vitals["map"] = round(
            new_vitals["diastolic_bp"] + (new_vitals["systolic_bp"] - new_vitals["diastolic_bp"]) / 3
        )

    return {
        "event_type": "vital_sign_check",
        "timestamp_hours": icu_hour,
        "patient_id": patient.bed,
        "patient_name": patient.name,
        "state_mutations": new_vitals,
        "tracked_attr_mentioned": patient.tracked_attribute in new_vitals,
        "metadata": {
            "acuity": patient.acuity,
            "time_of_day": get_time_of_day(icu_hour),
        },
    }
```

#### Handler 2: `lab_result`

```python
def handle_lab_result(patient, icu_hour, lab_panel, rng) -> EventDict:
    """
    Lab values return from the lab.

    Parameters:
        patient: PatientState
        icu_hour: float
        lab_panel: str — "bmp" | "cbc" | "abg" | "lactate" | "coag" | "lft" | "troponin"
        rng: Random

    Preconditions:
        - Patient is alive
        - Appropriate labs have been ordered (AM labs for all patients, stat labs for
          specific indications)

    State mutations (by panel):
        bmp:      sodium, potassium, creatinine, glucose, (bun, hco3 as metadata)
        cbc:      hemoglobin, wbc, platelets
        abg:      (ph, paco2, pao2, hco3 as metadata — not directly tracked)
        lactate:  lactate
        coag:     inr, (ptt, fibrinogen as metadata)
        lft:      bilirubin, (ast, alt, alk_phos, albumin as metadata)
        troponin: troponin

    Returns:
        EventDict with state_mutations for the relevant lab values
    """
    PANEL_ATTRS = {
        "bmp":      ["sodium", "potassium", "creatinine", "glucose"],
        "cbc":      ["hemoglobin", "wbc", "platelets"],
        "lactate":  ["lactate"],
        "coag":     ["inr"],
        "lft":      ["bilirubin"],
        "troponin": ["troponin"],
    }

    attrs = PANEL_ATTRS.get(lab_panel, [])
    mutations = {}
    for attr in attrs:
        if attr in patient.waypoints:
            mutations[attr] = interpolate_value(
                patient.waypoints[attr], icu_hour, attr, rng
            )

    # Apply renal-electrolyte correlations
    apply_physiological_correlations(mutations, patient)

    return {
        "event_type": "lab_result",
        "timestamp_hours": icu_hour,
        "patient_id": patient.bed,
        "patient_name": patient.name,
        "state_mutations": mutations,
        "tracked_attr_mentioned": patient.tracked_attribute in mutations,
        "metadata": {
            "panel": lab_panel,
            "previous_values": {a: patient.current_state.get(a) for a in attrs},
        },
    }
```

#### Handler 3: `medication_change`

```python
def handle_medication_change(patient, icu_hour, med_type, rng) -> EventDict:
    """
    Medication started, titrated, or discontinued.

    Parameters:
        patient: PatientState
        icu_hour: float
        med_type: str — "vasopressor_titration" | "vasopressor_wean" | "antibiotic_change" |
                        "sedation_change" | "insulin_titration" | "new_medication"
        rng: Random

    Preconditions:
        - vasopressor_titration: patient.norepinephrine_dose > 0 and MAP < target
        - vasopressor_wean: patient.norepinephrine_dose > 0 and MAP > target + 5
        - insulin_titration: patient on insulin drip and glucose check done
        - sedation_change: patient on sedation and RASS assessment done

    State mutations:
        vasopressor_titration:
            norepinephrine_dose: current + uniform(0.02, 0.05)
            (MAP effect appears in NEXT vital_sign_check via correlation engine)
        vasopressor_wean:
            norepinephrine_dose: current - uniform(0.02, 0.05), min 0
        antibiotic_change:
            (no tracked attr mutations; metadata only)
        sedation_change:
            (rass may shift ±1-2 at next nursing_assessment)
        insulin_titration:
            (glucose effect at next lab; insulin_rate in metadata)

    Returns:
        EventDict
    """
    mutations = {}
    metadata = {"med_type": med_type}

    if med_type == "vasopressor_titration":
        delta = rng.uniform(0.02, 0.05)
        new_dose = round(patient.norepinephrine_dose + delta, 2)
        new_dose = min(3.0, new_dose)
        mutations["norepinephrine_dose"] = new_dose
        metadata["old_dose"] = patient.norepinephrine_dose
        metadata["new_dose"] = new_dose

    elif med_type == "vasopressor_wean":
        delta = rng.uniform(0.02, 0.05)
        new_dose = round(max(0, patient.norepinephrine_dose - delta), 2)
        mutations["norepinephrine_dose"] = new_dose
        metadata["old_dose"] = patient.norepinephrine_dose
        metadata["new_dose"] = new_dose
        if new_dose == 0:
            metadata["milestone"] = "vasopressors_off"

    elif med_type == "insulin_titration":
        old_rate = patient.insulin_rate or 0
        glucose = patient.current_state.get("glucose", 200)
        if glucose > 250:
            new_rate = old_rate + rng.choice([1, 2])
        elif glucose < 150:
            new_rate = max(0, old_rate - rng.choice([1, 2]))
        else:
            new_rate = old_rate
        metadata["old_rate"] = old_rate
        metadata["new_rate"] = new_rate
        metadata["trigger_glucose"] = glucose

    return {
        "event_type": "medication_change",
        "timestamp_hours": icu_hour,
        "patient_id": patient.bed,
        "patient_name": patient.name,
        "state_mutations": mutations,
        "tracked_attr_mentioned": patient.tracked_attribute in mutations,
        "metadata": metadata,
    }
```

#### Handler 4: `ventilator_change`

```python
def handle_ventilator_change(patient, icu_hour, vent_action, rng) -> EventDict:
    """
    Ventilator setting adjustment or mode change.

    Parameters:
        patient: PatientState
        icu_hour: float
        vent_action: str — "fio2_wean" | "fio2_increase" | "peep_wean" | "peep_increase" |
                           "mode_change" | "extubation" | "intubation"
        rng: Random

    Preconditions:
        - fio2_wean: patient.fio2 > 30 and SpO2 >= 94 for past 2 checks
        - peep_wean: patient.peep > 5 and SpO2 >= 92 on current FiO2
        - extubation: FiO2 <= 40 AND PEEP <= 8 AND GCS >= 10 AND passing SBT
        - intubation: SpO2 < 85 despite max non-invasive OR GCS < 8 OR RR < 6

    State mutations:
        fio2_wean:     fio2: current - uniform(5, 15), min 21
        fio2_increase: fio2: current + uniform(5, 20), max 100
        peep_wean:     peep: current - 2
        peep_increase: peep: current + 2
        extubation:    fio2 -> None, peep -> None, ventilator_mode -> "nasal_cannula" or "room_air"
        intubation:    fio2 -> 100, peep -> 8, ventilator_mode -> "AC_VC"

    Returns:
        EventDict
    """
    mutations = {}
    metadata = {"vent_action": vent_action}

    if vent_action == "fio2_wean":
        old_fio2 = patient.fio2
        new_fio2 = max(21, old_fio2 - rng.randint(5, 15))
        mutations["fio2"] = new_fio2
        metadata["old_fio2"] = old_fio2

    elif vent_action == "fio2_increase":
        old_fio2 = patient.fio2
        new_fio2 = min(100, old_fio2 + rng.randint(5, 20))
        mutations["fio2"] = new_fio2
        metadata["old_fio2"] = old_fio2

    elif vent_action == "peep_wean":
        old_peep = patient.peep
        new_peep = max(5, old_peep - 2)
        mutations["peep"] = new_peep
        metadata["old_peep"] = old_peep

    elif vent_action == "peep_increase":
        old_peep = patient.peep
        new_peep = min(24, old_peep + 2)
        mutations["peep"] = new_peep
        metadata["old_peep"] = old_peep

    elif vent_action == "extubation":
        mutations["fio2"] = None
        mutations["peep"] = None
        metadata["post_extubation_device"] = rng.choice(["nasal_cannula_3L", "nasal_cannula_4L", "room_air"])
        metadata["milestone"] = "extubated"

    elif vent_action == "intubation":
        mutations["fio2"] = 100
        mutations["peep"] = 8
        metadata["vent_mode"] = "AC_VC"
        metadata["indication"] = rng.choice(["respiratory_failure", "airway_protection", "status_epilepticus"])

    return {
        "event_type": "ventilator_change",
        "timestamp_hours": icu_hour,
        "patient_id": patient.bed,
        "patient_name": patient.name,
        "state_mutations": mutations,
        "tracked_attr_mentioned": patient.tracked_attribute in mutations,
        "metadata": metadata,
    }
```

#### Handler 5: `clinical_deterioration`

```python
def handle_clinical_deterioration(patient, icu_hour, deterioration_type, rng) -> EventDict:
    """
    Acute clinical decline requiring immediate intervention.

    Parameters:
        patient: PatientState
        icu_hour: float
        deterioration_type: str — "new_arrhythmia" | "acute_desaturation" |
                                   "hemodynamic_collapse" | "acute_bleeding" |
                                   "seizure" | "fever_spike" | "code_blue"
        rng: Random

    Preconditions:
        - Patient is alive and in ICU
        - Weighted toward worsening trajectory patients (3x weight vs improving)
        - code_blue: rare (1-2% of ICU patients)

    State mutations by type:
        new_arrhythmia:
            heart_rate: rng.randint(140, 180)  # jump to afib/SVT range
            systolic_bp: current - rng.randint(15, 30)  # hemodynamic impact
            map: recalculate

        acute_desaturation:
            spo2: current - rng.randint(8, 15)  # acute drop
            respiratory_rate: current + rng.randint(5, 12)
            heart_rate: current + rng.randint(10, 25)  # compensatory

        hemodynamic_collapse:
            systolic_bp: rng.randint(55, 75)
            map: rng.randint(35, 50)
            heart_rate: current + rng.randint(20, 40)
            lactate: current + rng.uniform(2.0, 5.0)
            norepinephrine_dose: increase by 0.10-0.20 or start at 0.10

        acute_bleeding:
            hemoglobin: current - rng.uniform(1.5, 3.0)
            heart_rate: current + rng.randint(15, 30)
            systolic_bp: current - rng.randint(15, 30)

        seizure:
            gcs: 3 (during ictal)
            heart_rate: rng.randint(120, 160)
            lactate: current + rng.uniform(3.0, 8.0)

        fever_spike:
            temperature_c: current + rng.uniform(1.0, 2.0)
            heart_rate: current + round(10 * temp_increase)  # per correlation

        code_blue:
            heart_rate: 0 or rng.choice([30, 0, 180])  # arrest rhythms
            systolic_bp: 0
            map: 0
            spo2: rng.randint(40, 70)

    Returns:
        EventDict — state_mutations reflect the acute change.
        Metadata includes response actions (e.g., "started norepinephrine",
        "intubated", "called code team").
    """
    mutations = {}
    metadata = {"deterioration_type": deterioration_type, "response_actions": []}

    if deterioration_type == "new_arrhythmia":
        mutations["heart_rate"] = rng.randint(140, 180)
        mutations["systolic_bp"] = max(60, patient.current_state.get("systolic_bp", 110) - rng.randint(15, 30))
        mutations["map"] = max(40, patient.current_state.get("map", 75) - rng.randint(10, 20))
        metadata["rhythm"] = rng.choice(["atrial fibrillation with RVR", "SVT", "ventricular tachycardia"])
        metadata["response_actions"].append("cardiology consulted")

    elif deterioration_type == "acute_desaturation":
        mutations["spo2"] = max(60, patient.current_state.get("spo2", 95) - rng.randint(8, 15))
        mutations["respiratory_rate"] = min(45, patient.current_state.get("respiratory_rate", 18) + rng.randint(5, 12))
        mutations["heart_rate"] = min(180, patient.current_state.get("heart_rate", 90) + rng.randint(10, 25))
        metadata["cause"] = rng.choice(["mucus plug", "pneumothorax", "flash pulmonary edema", "PE"])
        metadata["response_actions"].append("FiO2 increased to 100%")

    elif deterioration_type == "hemodynamic_collapse":
        mutations["systolic_bp"] = rng.randint(55, 75)
        mutations["diastolic_bp"] = rng.randint(30, 45)
        mutations["map"] = rng.randint(35, 50)
        mutations["heart_rate"] = min(180, patient.current_state.get("heart_rate", 90) + rng.randint(20, 40))
        mutations["lactate"] = round(patient.current_state.get("lactate", 2.0) + rng.uniform(2.0, 5.0), 1)
        new_norepi = round(patient.norepinephrine_dose + rng.uniform(0.10, 0.20), 2)
        mutations["norepinephrine_dose"] = min(3.0, new_norepi)
        metadata["response_actions"].extend(["fluid bolus 1L LR", "norepinephrine uptitrated"])

    elif deterioration_type == "acute_bleeding":
        mutations["hemoglobin"] = round(max(4.0, patient.current_state.get("hemoglobin", 10.0) - rng.uniform(1.5, 3.0)), 1)
        mutations["heart_rate"] = min(160, patient.current_state.get("heart_rate", 90) + rng.randint(15, 30))
        mutations["systolic_bp"] = max(60, patient.current_state.get("systolic_bp", 110) - rng.randint(15, 30))
        metadata["response_actions"].extend(["type and cross sent", "2 units PRBC ordered"])

    elif deterioration_type == "seizure":
        mutations["gcs"] = 3
        mutations["heart_rate"] = rng.randint(120, 160)
        mutations["lactate"] = round(patient.current_state.get("lactate", 1.5) + rng.uniform(3.0, 8.0), 1)
        metadata["response_actions"].extend(["lorazepam 4mg IV given", "neurology stat consult"])

    elif deterioration_type == "fever_spike":
        temp_increase = rng.uniform(1.0, 2.0)
        mutations["temperature_c"] = round(min(42.0, patient.current_state.get("temperature_c", 37.0) + temp_increase), 1)
        mutations["heart_rate"] = min(160, patient.current_state.get("heart_rate", 90) + round(10 * temp_increase))
        metadata["response_actions"].extend(["blood cultures drawn", "acetaminophen given"])

    elif deterioration_type == "code_blue":
        mutations["heart_rate"] = rng.choice([0, 30, 180])  # PEA, brady, VT/VF
        mutations["systolic_bp"] = 0
        mutations["map"] = 0
        mutations["spo2"] = rng.randint(40, 70)
        metadata["response_actions"].extend(["code team called", "CPR initiated", "epinephrine 1mg IV"])
        metadata["is_code_blue"] = True

    return {
        "event_type": "clinical_deterioration",
        "timestamp_hours": icu_hour,
        "patient_id": patient.bed,
        "patient_name": patient.name,
        "state_mutations": mutations,
        "tracked_attr_mentioned": patient.tracked_attribute in mutations,
        "metadata": metadata,
    }
```

### 3.6 Attribute Trajectory Constraints (Critical for Realism)

**Vital sign trajectories must follow physiological rules:**

```
SEPSIS — IMPROVING (typical 3-5 day trajectory):
  Hour 0:   HR 125, SBP 82, MAP 55, Temp 39.2, Lactate 6.2, Norepi 0.20
  Hour 6:   HR 118, SBP 90, MAP 62, Temp 39.0, Lactate 4.8, Norepi 0.15
  Hour 12:  HR 110, SBP 95, MAP 68, Temp 38.5, Lactate 3.2, Norepi 0.08
  Hour 24:  HR 102, SBP 102, MAP 72, Temp 38.0, Lactate 2.4, Norepi 0.03
  Hour 36:  HR 95, SBP 110, MAP 75, Temp 37.5, Lactate 1.8, Norepi OFF
  Hour 48:  HR 88, SBP 118, MAP 80, Temp 37.2, Lactate 1.2
  Hour 72:  HR 82, SBP 122, MAP 82, Temp 37.0, Lactate 0.9

  WBC: 22→18→14→11 over 3 days
  Creatinine: 2.8→2.5→2.0→1.5 over 3 days
  Procalcitonin: 15→8→3→0.8 over 5 days

SEPSIS — WORSENING:
  Hour 0:   HR 128, SBP 78, MAP 52, Temp 39.5, Lactate 4.5, Norepi 0.15
  Hour 6:   HR 135, SBP 72, MAP 48, Temp 40.1, Lactate 6.8, Norepi 0.30
  Hour 12:  HR 140, SBP 68, MAP 45, Temp 40.3, Lactate 9.2, Norepi 0.50, + Vasopressin 0.04
  Hour 24:  HR 145, SBP 65, MAP 42, Temp 39.8, Lactate 12.5, multiorgan failure
  (New AKI: Cr 2.0→3.5→5.0, Oliguric, K+ rising)
  (DIC: Platelets 180→90→35, INR 1.2→2.1→3.5, Fibrinogen 300→120→80)

DKA — IMPROVING (typical 12-24h):
  Hour 0:   Glucose 650, pH 7.12, HCO3 8, K+ 5.8, AG 28, HR 120, RR 35
  Hour 4:   Glucose 380, pH 7.20, HCO3 12, K+ 4.5, AG 20, HR 110, RR 28
  Hour 8:   Glucose 220, pH 7.28, HCO3 16, K+ 3.8, AG 14, HR 98, RR 22
  Hour 12:  Glucose 165, pH 7.34, HCO3 20, K+ 3.5, AG 11, HR 88, RR 18
  Hour 24:  Glucose 140, pH 7.38, HCO3 23, K+ 4.0, AG 10, HR 82, RR 16

Key constraint: Glucose drops 50-100 mg/dL/hr on insulin drip. AG closes as pH corrects.
K+ FALLS as acidosis corrects (despite initially high serum K, total body K is depleted).

GI BLEED — STABILIZING:
  Hour 0:   Hgb 6.8, HR 118, SBP 88, MAP 58 → transfuse 2 units PRBC
  Hour 4:   Hgb 8.2 (post-transfusion), HR 102, SBP 100, MAP 68
  Hour 8:   Hgb 8.0 (stable, no rebleed), HR 92, SBP 110, MAP 75
  Hour 24:  Hgb 7.8, HR 85, SBP 118, MAP 80 → trending to stability

Key constraint: Each unit PRBC raises Hgb ~1 g/dL. If Hgb keeps dropping → active bleed.
```

**Cross-attribute correlations the state machine must enforce:**
- Fever → tachycardia (~10 bpm per °C above 37)
- Hypotension → compensatory tachycardia (unless beta-blocked)
- Low Hgb → tachycardia (compensatory)
- Acidosis → hyperventilation (RR increases, Kussmaul in DKA)
- Rising creatinine → rising potassium, rising BUN
- Worsening liver function → rising INR, rising bilirubin
- PEEP increase → possible MAP decrease (reduced preload)
- Vasopressor increase → MAP increase (within 5-15 min)

## 4. Narrative Generation

### 4.1 Story Structure

A trial narrative has this structure:

```
[ICU HEADER]
  - Unit type, current census, time/date, attending

[BLOCK 1: Admission / Initial Assessment]
  - 2-5 patient introductions with admitting vitals and diagnosis

[BLOCK 2: Ongoing Management (mid-shift or next shift)]
  - 3-10 clinical events across patients (vitals, labs, interventions)

[BLOCK 3: Later Course (subsequent shift or day)]
  - 2-8 events showing evolution

[Optional: SHIFT HANDOFF SUMMARY]
```

Scales with num_updates. For num_updates=3, might be a single shift snapshot. For num_updates=20, could span 2-3 ICU days.

### 4.2 Narrative Voice — 50/50

**Shift handoff voice** (structured, terse, clinical shorthand):
> "Bed 4, Mrs. Liu, 67-year-old female, sepsis secondary to cholangitis, ICU day 2. HR 118, BP 92/58, MAP 69, temp 38.6, SpO2 94% on 4L NC. Lactate trending down, 3.2 from 6.1 yesterday. Norepi weaned to 0.08 from 0.20. Creatinine up to 2.4 from 1.8. Urine output 25 mL/hr. Pip-tazo day 2. Plan: continue vasopressor wean, repeat lactate at 12:00, hold on diuresis until off pressors."

**Progress note voice** (narrative, fuller sentences, clinical reasoning visible):
> "At the 06:00 assessment, Mrs. Liu's heart rate had climbed to 118 beats per minute, up from 105 at midnight. The team suspected her persistent tachycardia was multifactorial — her temperature had ticked up to 38.6°C, and her lactate, though improving at 3.2, was not yet cleared. Blood pressure remained tenuous at 92/58, MAP 69, still dependent on norepinephrine at 0.08 mcg/kg/min. The overnight resident had weaned the levophed cautiously from 0.20, but further de-escalation would wait for the morning lactate. Her creatinine had risen to 2.4, prompting nephrology to be consulted."

### 4.3 Narrative Templates

```python
# ── ICU HEADER TEMPLATES ──

HEADER_TEMPLATES = [
    "{unit_name}, {hospital_name}. {date}, {time}. Attending: Dr. {attending}. "
    "Current census: {census}/{capacity} beds. {census_note}.",

    "Report from the {shift_name} shift, {unit_name}. {date}. {num_patients} "
    "patients currently on service under Dr. {attending}. {unit_context}.",
]

# ── VITAL SIGN CHECK TEMPLATES ──

VITALS_HANDOFF = [
    "Bed {bed}: HR {hr}, BP {sbp}/{dbp}, MAP {map}, RR {rr}, SpO2 {spo2}% "
    "on {o2_device}, temp {temp}°C. {trend_note}.",

    "{time}: {patient_name} — vitals stable with HR {hr}, pressure {sbp}/{dbp}, "
    "sat {spo2}% on {o2_device}. {additional_note}.",
]

VITALS_PROGRESS = [
    "At the {time} check, {patient_name}'s heart rate was {hr} beats per minute, "
    "{rhythm_description}. Blood pressure read {sbp}/{dbp} with a MAP of {map}. "
    "{o2_status}. {clinical_context}.",

    "The {time} vitals for {patient_name} showed {vital_summary}. {interpretation}. "
    "{plan_note}.",

    "{patient_name}'s {time} assessment: heart rate {hr}, {hr_trend}. "
    "Blood pressure {sbp}/{dbp}, MAP {map} — {bp_interpretation}. "
    "Temperature {temp}°C. SpO2 {spo2}% on {o2_device}.",

    "Checking on {patient_name} at {time}, the nurse documented a heart rate of "
    "{hr} — {hr_change} from the previous check. Pressure was {sbp}/{dbp}, "
    "MAP {map}. {additional_vitals}.",
]

# ── LAB RESULT TEMPLATES ──

LAB_HANDOFF = [
    "AM labs: Na {sodium}, K {potassium}, Cr {creatinine}, BUN {bun}, "
    "glucose {glucose}, Hgb {hemoglobin}, WBC {wbc}, platelets {platelets}. "
    "{notable_values}.",

    "Lactate back at {lactate} — {lactate_trend}. {additional_labs}.",
]

LAB_PROGRESS = [
    "The morning labs for {patient_name} came back showing {key_lab} at {value} "
    "{units}, {comparison_to_prior}. {clinical_significance}. {action_taken}.",

    "{patient_name}'s {time} lactate returned at {lactate} mmol/L, "
    "{trend_direction} from {previous_lactate}. {interpretation}. "
    "{management_response}.",

    "Overnight labs revealed {patient_name}'s creatinine had {cr_trend} to "
    "{creatinine} mg/dL. Potassium was {potassium}. {renal_assessment}. "
    "{plan}.",

    "The CBC showed {patient_name}'s hemoglobin at {hemoglobin} g/dL, "
    "{hgb_change}. White count {wbc}k, {wbc_interpretation}. "
    "Platelets {platelets}k. {action}.",
]

# ── MEDICATION CHANGE TEMPLATES ──

MED_HANDOFF = [
    "Norepi weaned from {old_dose} to {new_dose}. MAP maintained >{map_target}.",

    "Started on {medication} {dose} for {indication}. {monitoring_plan}.",
]

MED_PROGRESS = [
    "At {time}, the team titrated {patient_name}'s norepinephrine from "
    "{old_dose} to {new_dose} mcg/kg/min. The MAP responded within minutes, "
    "settling at {map}. {assessment}.",

    "Dr. {physician} ordered {medication} {dose} {route} for {patient_name} after "
    "{clinical_trigger}. {expected_effect}. {monitoring}.",

    "{patient_name}'s insulin drip was adjusted from {old_rate} to {new_rate} "
    "units/hour after a point-of-care glucose of {glucose} mg/dL. "
    "{glucose_trend}.",
]

# ── VENTILATOR CHANGE TEMPLATES ──

VENT_HANDOFF = [
    "Vent: {mode}, FiO2 {fio2}%, PEEP {peep}, RR {set_rr}, Vt {vt}. "
    "ABG: {ph}/{paco2}/{pao2}/{hco3}. P:F {pf_ratio}.",
]

VENT_PROGRESS = [
    "Respiratory therapy weaned {patient_name}'s FiO2 from {old_fio2}% to "
    "{new_fio2}%, and the SpO2 held steady at {spo2}%. PEEP remained at "
    "{peep} cmH2O. {assessment}.",

    "{patient_name} passed the spontaneous breathing trial at {time} — "
    "RSBI {rsbi}, no distress over 30 minutes. The team proceeded with "
    "extubation at {extubation_time}. Post-extubation: {post_extubation_status}.",

    "Given {patient_name}'s worsening oxygenation (P:F {pf_ratio}), PEEP was "
    "increased from {old_peep} to {new_peep} cmH2O. FiO2 remained at {fio2}%. "
    "{response}.",
]

# ── CLINICAL DETERIORATION TEMPLATES ──

DETERIORATION_PROGRESS = [
    "At {time}, {patient_name}'s monitor alarmed — heart rate had jumped to "
    "{hr} with a new {rhythm}. Blood pressure dropped to {sbp}/{dbp}. "
    "{response_description}. {outcome}.",

    "Rapid response called to Bed {bed} at {time}. {patient_name} found "
    "{clinical_finding}. {intervention}. {post_intervention_status}.",

    "{patient_name} became acutely {symptom} at {time}. SpO2 fell to {spo2}%. "
    "The team {intervention}. {current_status}.",
]

# ── CLINICAL IMPROVEMENT TEMPLATES ──

IMPROVEMENT_PROGRESS = [
    "Good news for {patient_name}: vasopressors were successfully discontinued "
    "at {time} after {duration_on_pressors} hours. MAP has been maintaining "
    "above {map} on {pronoun} own. {plan}.",

    "{patient_name} was extubated at {time} and transitioned to {o2_device}. "
    "SpO2 {spo2}%, respiratory rate {rr}, no distress. {post_extubation_plan}.",

    "{patient_name}'s lactate finally cleared — {lactate} mmol/L at {time}, "
    "down from a peak of {peak_lactate}. {clinical_significance}. {plan}.",
]

# ── SHIFT HANDOFF SUMMARY TEMPLATES ──

HANDOFF_SUMMARY = [
    "Bed {bed}, {patient_name}, {age}yo {sex}, {diagnosis}, {icu_day_str}. "
    "Overnight: {overnight_summary}. Current: {current_vitals_summary}. "
    "Plan: {plan}.",
]

# ── TRANSFUSION TEMPLATES ──

TRANSFUSION_PROGRESS = [
    "{patient_name} received {num_units} unit(s) packed red blood cells after "
    "hemoglobin returned at {pre_hgb} g/dL. Post-transfusion Hgb: {post_hgb} g/dL. "
    "{response}.",
]
```

### 4.3.1 Template Variable Pools

Every `{placeholder}` in the templates above must draw from a defined pool. Grouped by category:

```python
# ── Vital Sign Descriptions ──
HR_TREND_POOL = [
    "up from", "down from", "unchanged from", "slightly elevated from",
    "slightly lower than", "essentially the same as", "a jump from",
    "marginally improved from",
]

BP_INTERPRETATION_POOL = [
    "still above the vasopressor threshold",
    "marginal, just above target MAP",
    "adequate for now but tenuous",
    "improved from the overnight low",
    "concerning for worsening perfusion",
    "well within the target range",
    "low but stable on current support",
    "improved since the vasopressor uptitration",
]

O2_STATUS_POOL = [
    "Saturating {spo2}% on {o2_device}",
    "SpO2 was {spo2}% on {fio2}% FiO2",
    "Oxygen saturation held at {spo2}% on {o2_device}",
    "Sat was {spo2}% — stable on {o2_device}",
    "Room air sat {spo2}%",
    "Maintaining {spo2}% on current settings",
]

RHYTHM_DESCRIPTION_POOL = [
    "sinus tachycardia on the monitor",
    "normal sinus rhythm",
    "sinus rhythm with occasional PVCs",
    "atrial fibrillation with rapid ventricular response",
    "sinus bradycardia",
    "irregularly irregular rhythm",
    "sinus rhythm, rate controlled",
    "paced rhythm",
]

# ── Lab Interpretations ──
CLINICAL_SIGNIFICANCE_POOL = [
    "concerning for worsening renal function",
    "trending in the right direction",
    "a modest improvement from yesterday",
    "essentially unchanged",
    "a new critical value requiring immediate attention",
    "consistent with the clinical trajectory",
    "suggestive of ongoing organ dysfunction",
    "the first sign of improvement in this parameter",
]

LACTATE_TREND_POOL = [
    "trending down nicely",
    "improved from the peak",
    "still elevated but clearing",
    "essentially cleared at this point",
    "stubbornly elevated despite resuscitation",
    "rising, concerning for worsening perfusion",
    "bouncing around without a clear trend",
    "down by more than 10% from last check",
]

WBC_INTERPRETATION_POOL = [
    "consistent with ongoing infection",
    "trending down on antibiotics",
    "leukocytosis persisting",
    "improving leukocytosis",
    "stress leukocytosis versus true infection",
    "now within normal range",
]

# ── Clinical Context / Assessment ──
ASSESSMENT_POOL = [
    "the team was cautiously optimistic",
    "this was attributed to the medication change",
    "the overnight course had been uneventful",
    "the attending noted this was expected for the clinical trajectory",
    "the team was concerned about the trajectory",
    "this prompted a change in the management plan",
    "the trend was reassuring",
    "further workup was ordered to clarify the picture",
]

PLAN_NOTE_POOL = [
    "Plan was to continue current management and reassess in 4 hours",
    "The team decided to hold further changes pending the next set of labs",
    "Repeat labs were ordered for 6 hours from now",
    "The attending requested a subspecialty consult",
    "The decision was made to escalate therapy",
    "Plan was to wean support as tolerated",
    "Family meeting was planned to discuss goals of care",
    "The patient was cleared for transfer to the floor pending bed availability",
]

TREND_NOTE_POOL = [
    "Vitals trending toward baseline",
    "Hemodynamically stable overnight",
    "Required one vasopressor uptitration at 03:00",
    "No acute changes from prior",
    "Improving from admission vitals",
    "Unchanged from last assessment",
    "Some improvement in hemodynamics noted",
    "Vitals consistent with clinical trajectory",
]

# ── Filler Context Variables ──
CXR_FINDING_POOL = [
    "bilateral infiltrates unchanged from prior",
    "improved aeration at the left base",
    "new right-sided pleural effusion",
    "stable cardiomegaly with vascular congestion",
    "endotracheal tube in appropriate position",
    "no new acute process",
    "worsening bilateral opacities concerning for ARDS progression",
]

FAMILY_NOTE_POOL = [
    "was asking to speak with the attending",
    "appeared to understand the clinical situation",
    "had questions about the plan of care",
    "was requesting an update from the overnight team",
    "had stepped out for a break",
]

SITE_ASSESSMENT_POOL = [
    "clean, dry, and intact without signs of infection",
    "mildly erythematous; being monitored",
    "dressing changed, no drainage noted",
    "site appeared unremarkable",
    "some tenderness on palpation but no induration",
]

RASS_DESCRIPTION_POOL = [
    "alert and calm (RASS 0)",
    "drowsy but easily rousable (RASS -1)",
    "light sedation, brief eye opening to voice (RASS -2)",
    "moderate sedation, movement to voice (RASS -3)",
    "deep sedation, no response to voice (RASS -4)",
    "unarousable (RASS -5)",
    "restless (RASS +1)",
    "agitated, frequent non-purposeful movement (RASS +2)",
]

URINE_ASSESSMENT_POOL = [
    "adequate output suggesting maintained renal perfusion",
    "marginal — being watched closely",
    "below target — nephrology aware",
    "good output post-diuretic",
    "oliguric — fluid challenge being considered",
    "trending up from earlier today",
]
```

### 4.4 Location / Unit Pools

```python
ICU_UNITS = [
    {"name": "Medical ICU", "abbrev": "MICU", "beds": 16, "typical_diagnoses": ["sepsis", "ards", "dka", "gi_bleed", "copd_exacerbation", "liver_failure", "aki", "overdose", "pancreatitis", "pneumonia"]},
    {"name": "Surgical ICU", "abbrev": "SICU", "beds": 12, "typical_diagnoses": ["post_cardiac_surgery", "post_abdominal_surgery", "polytrauma", "pancreatitis"]},
    {"name": "Cardiac ICU", "abbrev": "CCU", "beds": 10, "typical_diagnoses": ["stemi", "decompensated_hf", "afib_rvr", "post_cardiac_surgery", "pe"]},
    {"name": "Neuro ICU", "abbrev": "NICU", "beds": 10, "typical_diagnoses": ["hemorrhagic_stroke", "ischemic_stroke", "tbi", "status_epilepticus", "meningitis"]},
    {"name": "Mixed Medical-Surgical ICU", "abbrev": "ICU", "beds": 20, "typical_diagnoses": "all"},
]

HOSPITALS = [
    "University Medical Center", "St. Mary's Hospital", "Memorial Regional",
    "Riverside General", "County Medical Center", "Valley Presbyterian",
    "Metropolitan Hospital", "Sacred Heart Medical Center",
    "Lakeview Regional Medical Center", "Providence Medical Center",
    "Harbor University Hospital", "Mercy General",
]
```

### 4.5 Filler / Context Sentences

Between tracked-attribute updates, we add context with non-tracked attributes:

```python
FILLER_TEMPLATES = [
    "The patient's Foley had put out {urine_ml} mL over the past {hours} hours — {urine_assessment}.",
    "Chest X-ray from this morning showed {cxr_finding}.",
    "Family was at bedside and {family_note}.",
    "Nursing noted the {line_type} site was {site_assessment}.",
    "{patient_name} was {rass_description} on the RASS scale.",
    "Diet had been advanced to {diet_status} and {diet_tolerance}.",
    "DVT prophylaxis with {prophylaxis_type} was continued.",
    "Blood cultures from {date} had {culture_result}.",
    "The patient had been receiving {fluid_type} at {rate} mL/hr.",
    "Pharmacy recommended {pharmacy_rec} based on {rationale}.",
    "The wound on {location} was {wound_assessment}.",
    "PT/OT had evaluated and {rehab_note}.",
    "Night shift reported {overnight_event}.",
]
```

## 5. Controlling num_keys and num_updates

### num_keys (number of patients)

- num_keys=2: two patients, detailed paired comparison (e.g., two sepsis patients, one improving, one worsening)
- num_keys=3-4: small ICU team's patients for one shift
- num_keys=5-6: typical attending's census in a unit
- num_keys=8-10: busy unit, full census for handoff
- num_keys=12: large unit, panoramic view

### num_updates (state changes per patient)

| num_updates | Updates per patient | ~Word count | Time period covered |
|-------------|-------------------|-------------|---------------------|
| 3 | 3 | 400-600 | ~4-8 hours (single shift snapshot) |
| 5 | 5 | 600-1000 | ~8-16 hours (full shift + handoff) |
| 7 | 7 | 900-1400 | ~24 hours (day-night cycle) |
| 10 | 10 | 1300-2000 | ~24-48 hours |
| 15 | 15 | 2000-3500 | ~48-72 hours (full ICU course) |
| 20+ | 20+ | 3000-5000 | ~72+ hours (extended ICU stay) |

### Interleaving Strategy

Events naturally interleave across patients because the ICU manages multiple patients simultaneously:

```
08:00: Bed 3, Mrs. Liu — vitals check: HR 118, BP 92/58 (lactate pending)
08:15: Bed 7, Mr. Okonkwo — AM labs back: Hgb 6.8, HR 112 → order 2 units PRBC
08:30: Bed 3, Mrs. Liu — lactate resulted: 3.2 (down from 6.1) → wean norepi
08:45: Bed 5, Mr. Reeves — ABG back: pH 7.22, PaCO2 55 → increase vent rate
09:00: Bed 7, Mr. Okonkwo — transfusion started, vitals q15min
09:15: Bed 2, Ms. Park — extubated after passing SBT, SpO2 96% on 3L NC
09:30: Bed 3, Mrs. Liu — repeat vitals: HR 110, MAP 72 → norepi now 0.05
...
```

Constraint: same patient should not have two consecutive events unless they are causally linked (e.g., lab result → immediate intervention).

## 6. Distinguishing Setups

### Variation dimensions:
1. **Diagnosis composition** — all sepsis, all cardiac, mixed medical, mixed surgical, etc.
2. **ICU type** — MICU, SICU, CCU, neuro ICU, mixed
3. **Shift** — day shift (07:00-19:00), night shift (19:00-07:00), 24h span
4. **Patient trajectory mix** — all improving, all worsening, mixed (most realistic)
5. **Acuity level** — quiet night (stable patients) vs chaotic shift (multiple deteriorations)
6. **Queried patient** — which patient the RI/PI question asks about
7. **Queried attribute** — HR vs lactate vs creatinine vs SpO2 etc.
8. **Hospital/attending names** — cosmetic variation

### Shift archetypes:

```python
SHIFT_ARCHETYPES = [
    "quiet_night",            # stable patients, routine checks, minimal changes
    "busy_admission_day",     # 2-3 new admissions, lots of initial workups
    "active_resuscitation",   # 1-2 critically ill patients getting aggressive intervention
    "weaning_and_recovery",   # patients improving, vent weaning, vasopressor weaning
    "multi_code",             # rare but dramatic — code blue + rapid response
    "steady_state",           # established patients, slow trends, lab-driven management
    "transfer_day",           # patients transferring out, new ones coming in
    "goals_of_care",          # one or more patients with family meetings, code status changes
    "post_op_recovery",       # surgical patients returning from OR
    "diagnostic_workup",      # new symptoms, consults, imaging, procedures ordered
]
```

## 7. Name Pools

### Patient Names (diverse, realistic)

```python
PATIENT_NAMES = {
    "female": [
        "Mrs. Liu", "Ms. Okafor", "Mrs. Patel", "Ms. Williams", "Mrs. Rodriguez",
        "Ms. Johansson", "Mrs. Nakamura", "Ms. Dubois", "Mrs. Campbell",
        "Ms. Reeves", "Mrs. Al-Rashid", "Ms. Kowalski", "Mrs. Chen",
        "Ms. Thakur", "Mrs. Whitehorse", "Ms. Okonkwo", "Mrs. Fitzgerald",
        "Ms. Tanaka", "Mrs. Bjornsson", "Ms. Guerrero", "Mrs. Abadi",
        "Ms. Delgado", "Mrs. Ibrahim", "Ms. Kim", "Mrs. Novak",
        "Ms. Fernandez", "Mrs. Chandra", "Ms. Browning", "Mrs. Sato",
        "Ms. Achebe", "Mrs. Volkov", "Ms. Hernandez", "Mrs. Kwon",
        "Ms. Moreau", "Mrs. Singh", "Ms. O'Brien", "Mrs. Yakubu",
    ],
    "male": [
        "Mr. Okonkwo", "Mr. Patel", "Mr. Williams", "Mr. Chen",
        "Mr. Rodriguez", "Mr. Johansson", "Mr. Nakamura", "Mr. Campbell",
        "Mr. Reeves", "Mr. Al-Rashid", "Mr. Kowalski", "Mr. Dubois",
        "Mr. Thakur", "Mr. Whitehorse", "Mr. Fitzgerald", "Mr. Liu",
        "Mr. Tanaka", "Mr. Bjornsson", "Mr. Guerrero", "Mr. Abadi",
        "Mr. Delgado", "Mr. Ibrahim", "Mr. Kim", "Mr. Novak",
        "Mr. Park", "Mr. Sato", "Mr. Achebe", "Mr. Volkov",
        "Mr. Hernandez", "Mr. Kwon", "Mr. Moreau", "Mr. Singh",
        "Mr. O'Brien", "Mr. Yakubu", "Mr. Fernandez", "Mr. Chandra",
    ],
}
```

### Medical Staff Names

```python
ATTENDING_NAMES = [
    "Dr. Mehta", "Dr. Johansson", "Dr. Washington", "Dr. Nakamura",
    "Dr. Torres", "Dr. Al-Farsi", "Dr. Petrov", "Dr. Okafor",
    "Dr. Lindstrom", "Dr. Gutierrez", "Dr. Yamamoto", "Dr. Brennan",
    "Dr. Chowdhury", "Dr. Morrison", "Dr. Kang", "Dr. Osei",
]

NURSE_NAMES = [
    "RN Martinez", "RN Johnson", "RN Pham", "RN Okafor",
    "RN Kowalski", "RN Washington", "RN Kim", "RN Brennan",
    "RN Thakur", "RN Campbell", "RN Hernandez", "RN Tanaka",
]

RESIDENT_NAMES = [
    "Dr. Park (PGY-2)", "Dr. Santos (PGY-3)", "Dr. Mueller (PGY-1)",
    "Dr. Adeyemi (PGY-2)", "Dr. Huang (PGY-3)", "Dr. Mikhailova (PGY-1)",
]
```

## 8. Question Templates

```python
RI_TEMPLATES = [
    "What was {patient_name}'s {attribute} at the first recorded check?",
    "At the earliest mention of {patient_name}'s {attribute}, what was the value?",
    "When {patient_name} first appears in these notes, what was {pronoun} {attribute}?",
    "What was the initial {attribute} documented for {patient_name}?",
]

PI_TEMPLATES = [
    "What was {patient_name}'s {attribute} at the most recent check?",
    "In the last recorded entry for {patient_name}, what was {pronoun} {attribute}?",
    "What was {patient_name}'s most recently documented {attribute}?",
    "At the final mention of {patient_name}, what was {pronoun} {attribute}?",
]
```

## 9. Output Format

Same schema as other domains (universal):

```json
{
    "id": "icu_001",
    "domain": "hospital_icu",
    "num_keys": 5,
    "num_updates": 7,
    "narrative": "Medical ICU, University Medical Center. January 14, 2024, 07:00...",
    "questions": {
        "RI": {
            "question": "What was Mrs. Liu's heart rate at the first recorded check?",
            "expected_answer": "125",
            "target_entity": "Mrs. Liu",
            "target_attribute": "heart_rate"
        },
        "PI": {
            "question": "What was Mrs. Liu's heart rate at the most recent check?",
            "expected_answer": "88",
            "target_entity": "Mrs. Liu",
            "target_attribute": "heart_rate"
        }
    },
    "entity_tracking": {
        "Mrs. Liu / heart_rate": ["125", "118", "110", "102", "95", "88"],
        "Mr. Okonkwo / heart_rate": ["112", "108", "115", "105", "98", "92"],
        "Mr. Reeves / heart_rate": ["92", "95", "88", "85", "82", "80"],
        "Ms. Park / heart_rate": ["105", "110", "102", "98", "95", "90"],
        "Mr. Chen / heart_rate": ["78", "82", "85", "80", "76", "74"]
    },
    "config": {
        "diagnosis_mode": "same",
        "diagnoses": ["sepsis"],
        "icu_type": "micu",
        "shift_start": "day_shift_am",
        "archetype": "active_resuscitation",
        "tracked_attribute": "heart_rate",
        "attribute_mode": "same",
        "filler_budget": "medium",
        "voice": "progress_note",
        "seed": 42
    }
}
```

## 10. Implementation Order

1. **Diagnosis database** — 15 diagnosis profiles with initial state ranges, trajectories (improving/worsening), typical labs/vitals
2. **Physiological correlation engine** — enforce cross-attribute relationships (fever→tachycardia, hypotension→tachycardia, etc.)
3. **Name pools** — patient names, staff names, hospital/unit names
4. **State class** — PatientState with all mutable attributes (vitals, labs, meds, vent settings)
5. **Event generators** — one function per event type, each enforcing clinical constraints
6. **Event scheduler** — decides which event happens next based on time of day, patient acuity, and archetype
7. **Narrative renderer** — converts event sequence into prose using templates
8. **Trial generator** — orchestrates everything into a single trial output
9. **Validation** — verify all tracked values appear in narrative, RI/PI answers correct

## 11. Design Decisions (Resolved)

### Q1: Which attribute is the primary interference target?
**DECISION: Randomly sample from full pool.** Heart rate is the "gold equivalent" (numeric, tight range across patients, frequent updates) but the pool has 30+ options. Random sampling = maximum variety.

### Q2: Same-diagnosis or mixed-diagnosis?
**DECISION: Sample 50/50.** Same-diagnosis = maximum interference (all sepsis patients have similar vital ranges). Mixed = medium interference (cardiac vs respiratory vs sepsis create somewhat different ranges, but still overlap for most vitals).

### Q3: Time scale?
**DECISION: Realistic ICU timing.** Vitals q1h for routine, q15min for unstable. Labs q4-24h. This is MUCH faster than wildlife (hours not weeks) but generates high-density narratives naturally. No compression needed.

### Q4: How many diagnoses?
**DECISION: Start with 15 diagnosis profiles.** Sepsis, STEMI, DKA, ARDS, GI bleed, COPD exacerbation, hemorrhagic stroke, post-cardiac surgery, acute liver failure, alcohol withdrawal, AKI, PE, pneumonia, drug overdose, pancreatitis. Each with full improving/worsening trajectories from the mechanics spec.

### Q5: Physiological correlations — how strict?
**DECISION: Enforce the major ones, allow noise for the minor ones.** Must enforce: fever↔tachycardia, hypotension↔tachycardia, vasopressor dose↔MAP, FiO2↔SpO2. Allow ±10-20% noise (reflecting real clinical variability). Don't enforce every obscure correlation (that's overengineering for interference testing).

### Q6: Patient mortality during narrative?
**DECISION: Allow it, same as wildlife.** If a patient's trajectory is worsening and they deteriorate to code blue, they may die. Dead patients stop generating events. This naturally reduces num_keys mid-narrative. Tracked in ground truth.

### Q7: Shift handoff as a special event?
**DECISION: Yes, it's a natural panoramic event.** Shift handoffs touch ALL patients briefly, providing a burst of tracked values in condensed form. This creates interesting interference patterns — many values in rapid succession.

## 12. Trial Config Schema (Final)

```python
@dataclass
class ICUTrialConfig:
    num_keys: int              # 2-12 patients
    num_updates: int           # 3-30 per patient (target)
    seed: int                  # reproducibility
    tracked_attribute: str     # "heart_rate" | "systolic_bp" | "map" | "spo2" |
                               # "temperature_c" | "respiratory_rate" |
                               # "potassium" | "sodium" | "creatinine" | "lactate" |
                               # "hemoglobin" | "wbc" | "platelets" | "glucose" |
                               # "troponin" | "inr" | "bilirubin" | "procalcitonin" |
                               # "gcs" | "rass" | "pain_score" | "sofa_score" |
                               # "norepinephrine_dose" | "fio2" | "peep" |
                               # "urine_output_ml_hr" | "fluid_balance_ml" |
                               # "ventilator_mode" | "code_status" | "mixed"
    attribute_mode: str        # "same" (all patients same attr) | "mixed" (per-patient)
    diagnosis_mode: str        # "same" (all same diagnosis) | "mixed" (different diagnoses)
    diagnoses: list            # list of diagnoses for each patient
    icu_type: str              # "micu" | "sicu" | "ccu" | "nicu" | "mixed"
    shift_start: str           # "day_shift_am" | "day_shift_pm" | "night_shift"
    filler_budget: str         # "minimal" | "light" | "medium" | "heavy"
    voice: str                 # "shift_handoff" | "progress_note"
    shift_archetype: str       # see archetypes above
    queried_patient_idx: int   # which patient to ask about (0 to num_keys-1)
    trajectory_mix: str        # "all_improving" | "all_worsening" | "mixed" | "fluctuating"
```

When `seed` is provided, all other fields can be auto-sampled deterministically:
```python
def auto_config(num_keys, num_updates, condition, seed):
    rng = Random(seed)
    diagnosis_mode = rng.choice(["same", "mixed"])

    if diagnosis_mode == "same":
        dx = rng.choice(["sepsis", "stemi", "dka", "ards", "gi_bleed",
                          "copd_exacerbation", "hemorrhagic_stroke",
                          "post_cardiac_surgery", "acute_liver_failure",
                          "alcohol_withdrawal", "aki", "pe", "pneumonia",
                          "drug_overdose", "pancreatitis"])
        diagnoses = [dx] * num_keys
    else:
        pool = ["sepsis", "stemi", "dka", "ards", "gi_bleed",
                "copd_exacerbation", "hemorrhagic_stroke",
                "post_cardiac_surgery", "acute_liver_failure",
                "alcohol_withdrawal", "aki", "pe", "pneumonia",
                "drug_overdose", "pancreatitis"]
        diagnoses = [rng.choice(pool) for _ in range(num_keys)]

    return ICUTrialConfig(
        num_keys=num_keys,
        num_updates=num_updates,
        condition=condition,
        seed=seed,
        tracked_attribute=rng.choice([
            "heart_rate", "systolic_bp", "map", "spo2",
            "temperature_c", "lactate", "creatinine", "potassium",
            "hemoglobin", "wbc", "glucose", "gcs", "rass",
            "norepinephrine_dose", "fio2",
        ]),
        attribute_mode=rng.choice(["same", "mixed"]),
        diagnosis_mode=diagnosis_mode,
        diagnoses=diagnoses,
        icu_type=rng.choice(["micu", "sicu", "ccu", "nicu", "mixed"]),
        shift_start=rng.choice(["day_shift_am", "day_shift_pm", "night_shift"]),
        filler_budget=rng.choice(["minimal", "light", "medium", "heavy"]),
        voice=rng.choice(["shift_handoff", "progress_note"]),
        shift_archetype=rng.choice([
            "quiet_night", "busy_admission_day", "active_resuscitation",
            "weaning_and_recovery", "multi_code", "steady_state",
            "transfer_day", "goals_of_care", "post_op_recovery",
            "diagnostic_workup",
        ]),
        queried_patient_idx=rng.randint(0, num_keys - 1),
        trajectory_mix=rng.choices(
            ["all_improving", "all_worsening", "mixed", "fluctuating"],
            weights=[30, 15, 45, 10]
        )[0],
    )
```
