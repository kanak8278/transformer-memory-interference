# Hospital ICU Patient Monitoring — Domain Specification

> Exhaustive reference for building a realistic patient state simulator.
> All values are standard clinical references; individual institutions may vary slightly.

---

## 1. Vital Signs — Normal Ranges by Age Group

### 1.1 Heart Rate (beats per minute)

| Age Group | Normal Range | Bradycardia | Tachycardia |
|-----------|-------------|-------------|-------------|
| Newborn (0-1 mo) | 100-160 | <100 | >160 |
| Infant (1-12 mo) | 100-150 | <100 | >150 |
| Toddler (1-3 yr) | 90-140 | <90 | >140 |
| Preschool (3-5 yr) | 80-120 | <80 | >120 |
| School-age (6-12 yr) | 70-110 | <70 | >110 |
| Adolescent (13-17 yr) | 60-100 | <60 | >100 |
| Adult (18-64 yr) | 60-100 | <60 | >100 |
| Elderly (65+ yr) | 60-100 | <60 | >100 (may be blunted by beta-blockers) |

**Notes:**
- Athletes may have resting HR of 40-60 bpm (normal variant)
- HR increases ~10 bpm per 1 degree C of fever above 37 C
- HR increases ~8-10 bpm per degree F above 98.6 F
- Pain, anxiety, hypovolemia, and medications all affect HR

### 1.2 Blood Pressure (mmHg) — Systolic / Diastolic

| Age Group | Normal Systolic | Normal Diastolic | Hypotension (SBP) | Hypertensive Crisis (SBP) |
|-----------|----------------|-----------------|-------------------|--------------------------|
| Newborn (0-1 mo) | 60-90 | 20-60 | <60 | >90 |
| Infant (1-12 mo) | 72-104 | 37-56 | <70 | >110 |
| Toddler (1-3 yr) | 86-106 | 42-63 | <70 + (age x 2) | >110 + (age x 2) |
| Preschool (3-5 yr) | 89-112 | 46-72 | <75 | >115 |
| School-age (6-12 yr) | 97-115 | 57-76 | <80 | >125 |
| Adolescent (13-17 yr) | 110-131 | 64-83 | <90 | >140 |
| Adult (18-64 yr) | 90-140 | 60-90 | <90 | >180 |
| Elderly (65+ yr) | 90-150 | 60-90 | <90 | >180 |

**BP Classification (Adult):**
- Normal: <120 / <80
- Elevated: 120-129 / <80
- Stage 1 HTN: 130-139 / 80-89
- Stage 2 HTN: >=140 / >=90
- Hypertensive Crisis: >180 / >120

### 1.3 Respiratory Rate (breaths per minute)

| Age Group | Normal Range | Tachypnea | Bradypnea |
|-----------|-------------|-----------|-----------|
| Newborn (0-1 mo) | 30-60 | >60 | <30 |
| Infant (1-12 mo) | 25-50 | >50 | <25 |
| Toddler (1-3 yr) | 20-30 | >30 | <20 |
| Preschool (3-5 yr) | 20-28 | >28 | <20 |
| School-age (6-12 yr) | 18-25 | >25 | <12 |
| Adolescent (13-17 yr) | 12-20 | >20 | <12 |
| Adult (18-64 yr) | 12-20 | >20 | <12 |
| Elderly (65+ yr) | 12-20 | >20 | <12 |

### 1.4 Oxygen Saturation (SpO2, %)

| Category | SpO2 Range |
|----------|-----------|
| Normal | 95-100% |
| Mild hypoxemia | 90-94% |
| Moderate hypoxemia | 85-89% |
| Severe hypoxemia | <85% |
| Critical | <80% (imminent organ damage) |

**Notes:**
- COPD patients may have baseline SpO2 of 88-92% (target 88-92% to avoid CO2 retention)
- Pulse oximetry is unreliable with poor perfusion, nail polish, carbon monoxide poisoning, methemoglobinemia
- Neonates (first 10 min of life): SpO2 rises from 60% to >90% over minutes

### 1.5 Temperature

| Measurement Site | Normal Range | Fever Threshold |
|-----------------|-------------|-----------------|
| Oral | 36.5-37.5 C (97.7-99.5 F) | >=38.0 C (100.4 F) |
| Rectal | 36.6-38.0 C (97.9-100.4 F) | >=38.3 C (100.9 F) |
| Axillary | 36.0-37.0 C (96.8-98.6 F) | >=37.5 C (99.5 F) |
| Tympanic | 36.4-37.8 C (97.5-100.0 F) | >=38.0 C (100.4 F) |
| Temporal artery | 36.4-37.6 C (97.5-99.7 F) | >=38.0 C (100.4 F) |

**Conversion offsets (approximate):**
- Rectal = Oral + 0.3-0.6 C (0.5-1.0 F)
- Axillary = Oral - 0.3-0.6 C (0.5-1.0 F)

**Classification:**
- Hypothermia: <36.0 C (96.8 F)
- Normal: 36.5-37.5 C
- Low-grade fever: 37.5-38.3 C
- Fever: 38.3-40.0 C
- Hyperpyrexia: >40.0 C (104.0 F) — medical emergency
- Lethal: >42.0 C (107.6 F)

### 1.6 Mean Arterial Pressure (MAP)

**Formula:** MAP = (SBP + 2 x DBP) / 3

| Category | MAP (mmHg) |
|----------|-----------|
| Normal | 70-105 |
| Adequate organ perfusion minimum | >=65 |
| Hypotension (ICU threshold) | <65 (initiate vasopressors) |
| Hypertensive | >105 |
| Cerebral autoregulation range | 50-150 |

**Typical ICU target:** MAP >= 65 mmHg (sepsis guidelines per Surviving Sepsis Campaign)

---

## 2. Lab Values — Complete Reference Ranges

### 2.1 Complete Blood Count (CBC)

| Parameter | Normal Range (Adult Male) | Normal Range (Adult Female) | Units |
|-----------|--------------------------|---------------------------|-------|
| Hemoglobin | 13.5-17.5 | 12.0-16.0 | g/dL |
| Hematocrit | 41-50 | 36-44 | % |
| RBC | 4.5-5.5 | 4.0-4.9 | million/uL |
| WBC | 4.5-11.0 | 4.5-11.0 | thousand/uL |
| Platelets | 150-400 | 150-400 | thousand/uL |
| MCV | 80-100 | 80-100 | fL |
| MCH | 27-33 | 27-33 | pg |
| MCHC | 32-36 | 32-36 | g/dL |
| RDW | 11.5-14.5 | 11.5-14.5 | % |

**WBC Differential:**

| Cell Type | Normal % | Absolute Count (cells/uL) |
|-----------|---------|--------------------------|
| Neutrophils | 40-70% | 1,800-7,700 |
| Lymphocytes | 20-40% | 1,000-4,800 |
| Monocytes | 2-8% | 100-800 |
| Eosinophils | 1-4% | 50-400 |
| Basophils | 0-1% | 0-100 |
| Bands (immature neutrophils) | 0-5% | 0-500 |

**Bandemia:** >10% bands = "left shift" = acute infection/inflammation

### 2.2 Basic Metabolic Panel (BMP)

| Parameter | Normal Range | Critical Low | Critical High | Units |
|-----------|-------------|-------------|--------------|-------|
| Sodium (Na+) | 135-145 | <120 | >160 | mEq/L |
| Potassium (K+) | 3.5-5.0 | <2.5 | >6.5 | mEq/L |
| Chloride (Cl-) | 98-106 | <80 | >120 | mEq/L |
| Bicarbonate (HCO3-) | 22-28 | <10 | >40 | mEq/L |
| BUN | 7-20 | — | >100 | mg/dL |
| Creatinine | 0.6-1.2 (M), 0.5-1.1 (F) | — | >10 | mg/dL |
| Glucose (fasting) | 70-100 | <40 | >500 | mg/dL |
| Calcium (total) | 8.5-10.5 | <6.0 | >13.0 | mg/dL |
| Ionized Calcium | 4.6-5.3 | <3.0 | >6.5 | mg/dL |
| Magnesium | 1.7-2.2 | <1.0 | >4.0 | mg/dL |
| Phosphorus | 2.5-4.5 | <1.0 | >8.0 | mg/dL |

**Anion Gap:** Na - (Cl + HCO3) = 8-12 mEq/L (normal)
- Elevated (>12): metabolic acidosis (DKA, lactic acidosis, renal failure, toxins)
- Mnemonic for elevated AG: MUDPILES (Methanol, Uremia, DKA, Propylene glycol, Isoniazid/Iron, Lactic acidosis, Ethylene glycol, Salicylates)

### 2.3 Liver Function Tests (LFTs)

| Parameter | Normal Range | Units | Significance |
|-----------|-------------|-------|-------------|
| ALT (SGPT) | 7-56 | U/L | Hepatocellular injury (more specific to liver) |
| AST (SGOT) | 10-40 | U/L | Hepatocellular injury (also cardiac, muscle) |
| Alkaline Phosphatase (ALP) | 44-147 | U/L | Cholestasis, bone disease |
| Total Bilirubin | 0.1-1.2 | mg/dL | Hemolysis, liver disease, obstruction |
| Direct (Conjugated) Bilirubin | 0.0-0.3 | mg/dL | Obstructive/hepatocellular disease |
| Indirect Bilirubin | 0.1-1.0 | mg/dL | Hemolysis, Gilbert syndrome |
| Albumin | 3.5-5.0 | g/dL | Synthetic function, nutrition |
| Total Protein | 6.0-8.3 | g/dL | Nutrition, inflammation |
| GGT | 9-48 | U/L | Cholestasis, alcohol use |

**Patterns:**
- Hepatocellular: ALT/AST >> ALP (viral hepatitis, drug toxicity, ischemia)
- Cholestatic: ALP >> ALT/AST (obstruction, drug-induced)
- ALT > AST: viral hepatitis
- AST > ALT (ratio > 2:1): alcoholic liver disease
- AST/ALT > 1000: ischemic hepatitis, acetaminophen toxicity, acute viral

### 2.4 Cardiac Markers

| Parameter | Normal Range | Units | Significance |
|-----------|-------------|-------|-------------|
| Troponin I | <0.04 | ng/mL | MI (rises 3-6h, peaks 12-24h, elevated 7-14 days) |
| Troponin T (high-sensitivity) | <14 (F), <22 (M) | ng/L | MI (more sensitive, rises 1-3h) |
| BNP | <100 | pg/mL | Heart failure (>400 = likely HF) |
| NT-proBNP | <125 (age <75), <450 (age >=75) | pg/mL | Heart failure (age-adjusted) |
| CK-MB | 0-5 | ng/mL | MI (rises 4-6h, peaks 12-24h, normalizes 48-72h) |
| CK Total | 22-198 (M), 22-170 (F) | U/L | Muscle damage, rhabdomyolysis |
| Myoglobin | 0-85 | ng/mL | Early MI marker (rises 1-3h, not specific) |
| LDH | 140-280 | U/L | Non-specific tissue damage |

### 2.5 Inflammatory Markers

| Parameter | Normal Range | Units | Significance |
|-----------|-------------|-------|-------------|
| CRP | <1.0 | mg/dL | Infection, inflammation (rises in 6-8h) |
| High-sensitivity CRP | <0.3 | mg/dL | Cardiovascular risk |
| Procalcitonin (PCT) | <0.05 | ng/mL | Bacterial infection (<0.25 = unlikely bacterial, >0.5 = likely bacterial, >2.0 = severe sepsis, >10 = septic shock) |
| ESR | 0-20 (M), 0-30 (F) | mm/hr | Chronic inflammation (slow responder) |
| Lactate | 0.5-2.0 | mmol/L | Tissue hypoperfusion (>2 = concerning, >4 = severe, associated with mortality) |
| Ferritin | 12-150 (F), 12-300 (M) | ng/mL | Iron stores, inflammation (acute phase reactant) |
| IL-6 | <7 | pg/mL | Cytokine storm, severe sepsis |

### 2.6 Coagulation Studies

| Parameter | Normal Range | Units | Significance |
|-----------|-------------|-------|-------------|
| PT | 11-13.5 | seconds | Extrinsic pathway (warfarin monitoring) |
| INR | 0.8-1.1 | ratio | Standardized PT (therapeutic on warfarin: 2.0-3.0) |
| aPTT | 25-35 | seconds | Intrinsic pathway (heparin monitoring) |
| Fibrinogen | 200-400 | mg/dL | DIC if <100, acute phase reactant if elevated |
| D-dimer | <0.50 | mg/L FEU | DVT/PE screening (sensitive, not specific), DIC |
| Thrombin Time | 14-19 | seconds | Heparin contamination, fibrinogen disorders |
| Anti-Xa | 0.3-0.7 (therapeutic heparin) | IU/mL | Heparin monitoring (alternative to aPTT) |

**DIC Pattern:** elevated PT/INR, elevated aPTT, low fibrinogen (<100), elevated D-dimer, low platelets, schistocytes on smear

### 2.7 Arterial Blood Gas (ABG)

| Parameter | Normal Range | Units |
|-----------|-------------|-------|
| pH | 7.35-7.45 | — |
| PaCO2 | 35-45 | mmHg |
| PaO2 | 80-100 | mmHg |
| HCO3 | 22-26 | mEq/L |
| Base Excess (BE) | -2 to +2 | mEq/L |
| SaO2 | 95-100 | % |

**P/F Ratio (PaO2/FiO2):**
- Normal: >400
- Mild ARDS: 200-300
- Moderate ARDS: 100-200
- Severe ARDS: <100

**ABG Interpretation Framework:**
1. Look at pH: <7.35 = acidosis, >7.45 = alkalosis
2. Look at PaCO2: if opposite direction of pH = respiratory cause
3. Look at HCO3: if same direction as pH = metabolic cause
4. Check for compensation (Winters formula for metabolic acidosis: expected PaCO2 = 1.5 x HCO3 + 8 +/- 2)
5. Calculate anion gap if metabolic acidosis

**Acid-Base Disorders Quick Reference:**

| Disorder | pH | PaCO2 | HCO3 |
|----------|-----|-------|------|
| Respiratory Acidosis | LOW | HIGH | normal (acute) or HIGH (chronic) |
| Respiratory Alkalosis | HIGH | LOW | normal (acute) or LOW (chronic) |
| Metabolic Acidosis | LOW | normal or LOW (compensation) | LOW |
| Metabolic Alkalosis | HIGH | normal or HIGH (compensation) | HIGH |

---

## 3. Common ICU Admission Diagnoses (25 Conditions)

### 3.1 Sepsis / Septic Shock

**Abnormal vitals:** Fever (>38.3 C) or hypothermia (<36 C), tachycardia (HR >90), tachypnea (RR >20), hypotension (SBP <90, MAP <65)
**Abnormal labs:** WBC >12k or <4k or >10% bands, lactate >2 mmol/L, procalcitonin elevated, positive blood cultures, elevated CRP, possible coagulopathy (elevated PT/INR, low platelets), elevated creatinine (AKI), elevated bilirubin, metabolic acidosis on ABG
**Key features:** qSOFA score (altered mentation, RR >=22, SBP <=100)

### 3.2 STEMI (ST-Elevation Myocardial Infarction)

**Abnormal vitals:** Tachycardia or bradycardia, hypotension or hypertension, diaphoresis
**Abnormal labs:** Troponin elevated (serial rise and fall), CK-MB elevated, BNP elevated if heart failure, possibly elevated lactate if cardiogenic shock
**Key features:** ECG with ST elevation, chest pain, door-to-balloon time <90 min

### 3.3 Acute Ischemic Stroke

**Abnormal vitals:** Hypertension (often >180/100), may be normal otherwise
**Abnormal labs:** Generally normal labs; check glucose (hypo/hyperglycemia can mimic stroke), INR (if on anticoagulation), platelets
**Key features:** NIH Stroke Scale, CT/MRI, tPA window (4.5h), thrombectomy window (up to 24h)

### 3.4 Hemorrhagic Stroke (ICH/SAH)

**Abnormal vitals:** Severe hypertension (SBP often >180), bradycardia with hypertension (Cushing reflex = brainstem herniation)
**Abnormal labs:** Generally normal; check coagulation (PT/INR, platelets), type and screen
**Key features:** GCS monitoring, ICP management, SBP target <140 for ICH

### 3.5 Acute Respiratory Failure / ARDS

**Abnormal vitals:** Tachypnea (RR >30), hypoxemia (SpO2 <90%), tachycardia
**Abnormal labs:** ABG: low PaO2, P/F ratio <300 (ARDS), may have respiratory acidosis (high PaCO2), elevated lactate if severe
**Key features:** Bilateral infiltrates on CXR, P/F ratio classification, lung-protective ventilation

### 3.6 COPD Exacerbation

**Abnormal vitals:** Tachypnea, low SpO2 (may be baseline 88-92%), tachycardia
**Abnormal labs:** ABG: elevated PaCO2 (often >50, chronic baseline may be 50-55), low pH if acute, elevated HCO3 if chronic, normal/elevated WBC
**Key features:** Wheezing, accessory muscle use, may need BiPAP or intubation

### 3.7 Diabetic Ketoacidosis (DKA)

**Abnormal vitals:** Tachycardia, tachypnea (Kussmaul breathing, RR 30-40+), hypotension (dehydration), possible hypothermia
**Abnormal labs:** Glucose >250 mg/dL (often 400-800+), pH <7.30, HCO3 <18, anion gap >12 (often >20), elevated beta-hydroxybutyrate, low potassium (total body depletion though serum may be normal/high initially), elevated BUN/creatinine (prerenal from dehydration), ketonuria
**Key features:** Fruity breath, altered mental status, fluid deficit of 3-6 L

### 3.8 Hyperosmolar Hyperglycemic State (HHS)

**Abnormal vitals:** Tachycardia, hypotension, altered mental status
**Abnormal labs:** Glucose >600 mg/dL (often >1000), serum osmolality >320 mOsm/kg, pH >7.30 (no significant ketoacidosis), minimal ketones, elevated BUN/creatinine, sodium may be artificially low (corrected sodium needed)
**Key features:** Profound dehydration (fluid deficit 8-10 L), high mortality (~20%)

### 3.9 Acute Kidney Injury (AKI)

**Abnormal vitals:** May be hypertensive, oliguria (<0.5 mL/kg/hr)
**Abnormal labs:** Elevated creatinine (rising >0.3 mg/dL in 48h or >1.5x baseline in 7 days = KDIGO Stage 1), elevated BUN, hyperkalemia, metabolic acidosis, hyperphosphatemia, hypocalcemia
**Key features:** KDIGO staging, urine output tracking, fluid balance

### 3.10 GI Bleed (Upper or Lower)

**Abnormal vitals:** Tachycardia (earliest sign), hypotension (with significant blood loss), orthostatic changes
**Abnormal labs:** Low hemoglobin/hematocrit (may be normal initially due to hemoconcentration, drops over 6-24h), elevated BUN (upper GI — digested blood), BUN/Cr ratio >20 suggests upper GI, may have coagulopathy
**Key features:** Melena (upper), hematochezia (lower or brisk upper), hematemesis, monitor serial H/H q6-8h

### 3.11 Pulmonary Embolism (PE)

**Abnormal vitals:** Tachycardia, tachypnea, hypoxemia, hypotension (massive PE)
**Abnormal labs:** Elevated D-dimer, elevated troponin (RV strain), elevated BNP, ABG shows hypoxemia with hypocapnia (respiratory alkalosis), elevated lactate if massive PE
**Key features:** Wells score, CT angiography, RV strain on echo, consider tPA for massive PE

### 3.12 Acute Pancreatitis

**Abnormal vitals:** Tachycardia, fever, hypotension (third-spacing)
**Abnormal labs:** Lipase >3x upper normal (>180 U/L), amylase elevated, elevated WBC, elevated CRP, elevated BUN, hypocalcemia (severe), elevated glucose, elevated LDH
**Key features:** Ranson criteria, BISAP score, aggressive IV fluid resuscitation

### 3.13 Traumatic Brain Injury (TBI)

**Abnormal vitals:** Cushing triad (hypertension, bradycardia, irregular respirations) = increased ICP, may be hypo/hypertensive
**Abnormal labs:** May be normal; check coagulation, electrolytes (DI can cause hypernatremia, SIADH can cause hyponatremia), glucose
**Key features:** GCS monitoring q1-2h, ICP monitoring (target <22 mmHg), CPP target 60-70 mmHg

### 3.14 Polytrauma

**Abnormal vitals:** Tachycardia, hypotension (hemorrhagic shock), tachypnea
**Abnormal labs:** Low hemoglobin/hematocrit, elevated lactate, metabolic acidosis, coagulopathy (trauma triad: hypothermia, acidosis, coagulopathy), elevated CK (rhabdomyolysis), elevated creatinine
**Key features:** ATLS protocol, massive transfusion protocol (1:1:1 RBC:FFP:platelets), damage control surgery

### 3.15 Post-Cardiac Surgery (CABG/Valve)

**Abnormal vitals:** Controlled by vasoactive drips, pacing wires, chest tube output monitoring
**Abnormal labs:** Elevated troponin (expected post-surgery), possibly low H/H, electrolyte shifts (low K, low Mg common), elevated glucose (stress response)
**Key features:** Chest tube output (>200 mL/hr = surgical bleeding), cardiac index monitoring, mediastinal re-exploration criteria

### 3.16 Post-Major Abdominal Surgery

**Abnormal vitals:** Tachycardia (pain, hypovolemia), possible fever (expected in first 48h)
**Abnormal labs:** Elevated WBC (stress response, normal up to 48-72h post-op), electrolyte derangements, elevated lactate (early sign of anastomotic leak or ischemia)
**Key features:** NG tube output, urine output, return of bowel function, wound assessment

### 3.17 Drug Overdose / Toxicology

**Abnormal vitals (varies by substance):**
- Opioids: bradypnea (RR <8), bradycardia, hypotension, hypothermia, miosis
- Stimulants: tachycardia, hypertension, hyperthermia, mydriasis
- Acetaminophen: may be initially normal, then hepatic failure at 48-72h
- Benzodiazepines: bradypnea, hypotension

**Abnormal labs:** Toxicology screen, acetaminophen level, salicylate level, ethanol level, metabolic acidosis (aspirin, methanol, ethylene glycol), elevated AST/ALT (acetaminophen at 24-72h), osmolar gap
**Key features:** Toxidrome identification, specific antidotes (naloxone, N-acetylcysteine, flumazenil, fomepizole)

### 3.18 Status Epilepticus

**Abnormal vitals:** Tachycardia, hypertension, hyperthermia, tachypnea (or apnea during seizure)
**Abnormal labs:** Metabolic acidosis (lactic), elevated CK (rhabdomyolysis risk), elevated WBC (stress), low antiepileptic drug levels, elevated prolactin (transient)
**Key features:** EEG monitoring, benzodiazepine first-line, escalation to phenytoin/levetiracetam/propofol, airway protection

### 3.19 Acute Liver Failure

**Abnormal vitals:** Tachycardia, hypotension (vasodilation), possible fever
**Abnormal labs:** AST/ALT massively elevated (>1000-10000), elevated INR (>1.5, may be >5+), elevated bilirubin, low albumin, low glucose (impaired gluconeogenesis), elevated ammonia, elevated lactate, low fibrinogen
**Key features:** Encephalopathy grading (West Haven I-IV), cerebral edema risk, transplant evaluation, King's College criteria

### 3.20 Decompensated Heart Failure

**Abnormal vitals:** Tachycardia, hypo- or hypertension, tachypnea, hypoxemia, elevated JVP
**Abnormal labs:** Elevated BNP (>400 pg/mL) or NT-proBNP, elevated creatinine (cardiorenal), hyponatremia (dilutional, poor prognosis), elevated troponin (demand ischemia), elevated lactate (if cardiogenic shock)
**Key features:** Volume status assessment, daily weights, I&O, diuretic response, ejection fraction

### 3.21 Atrial Fibrillation with Rapid Ventricular Response (RVR)

**Abnormal vitals:** HR 120-180+ (irregularly irregular), variable BP
**Abnormal labs:** Check TSH (thyrotoxicosis), electrolytes (low K, low Mg), troponin (demand ischemia), BNP
**Key features:** Rate vs. rhythm control, anticoagulation (CHA2DS2-VASc), hemodynamic stability assessment

### 3.22 Pneumonia (Community or Hospital-Acquired)

**Abnormal vitals:** Fever, tachycardia, tachypnea, hypoxemia
**Abnormal labs:** Elevated WBC (left shift), elevated CRP/procalcitonin, elevated lactate if severe, sputum/blood cultures, possible respiratory acidosis on ABG
**Key features:** CURB-65 score, need for ICU if bilateral infiltrates + hypotension/mechanical ventilation

### 3.23 Meningitis / Encephalitis

**Abnormal vitals:** Fever, tachycardia, possible Cushing triad if elevated ICP
**Abnormal labs:** CSF: elevated WBC (>5 cells/uL), elevated protein, low glucose (bacterial), elevated opening pressure; blood: elevated WBC, elevated CRP/procalcitonin, blood cultures
**Key features:** Empiric antibiotics before LP if delayed, dexamethasone before antibiotics (bacterial), nuchal rigidity, Kernig/Brudzinski signs

### 3.24 Adrenal Crisis

**Abnormal vitals:** Severe hypotension (refractory to vasopressors), tachycardia, possible hypothermia
**Abnormal labs:** Hyponatremia, hyperkalemia, hypoglycemia, low cortisol (<3 mcg/dL), eosinophilia, possible mild metabolic acidosis
**Key features:** IV hydrocortisone 100 mg bolus then q8h, aggressive fluid resuscitation, vasopressor refractory hypotension is a clue

### 3.25 Acute Alcohol Withdrawal / Delirium Tremens

**Abnormal vitals:** Tachycardia, hypertension, fever, tachypnea, diaphoresis
**Abnormal labs:** Low magnesium, low potassium, low phosphorus, elevated AST/ALT (AST > ALT 2:1), thrombocytopenia, elevated GGT, elevated MCV, hypoglycemia possible
**Key features:** CIWA scoring, benzodiazepine protocol, seizure risk (12-48h), DTs (48-96h), phenobarbital if refractory

---

## 4. Medication Categories with Specific Drugs and Dose Ranges

### 4.1 Vasopressors and Inotropes

| Drug | Dose Range | Receptors | Primary Effect | Notes |
|------|-----------|-----------|---------------|-------|
| **Norepinephrine** | 0.01-3.0 mcg/kg/min | alpha-1 >> beta-1 | Vasoconstriction + mild inotropy | First-line for septic shock |
| **Vasopressin** | 0.01-0.04 units/min | V1 receptors | Vasoconstriction | Second-line in sepsis; NOT weight-based; fixed dose typically 0.03-0.04 U/min |
| **Epinephrine** | 0.01-0.5 mcg/kg/min | beta-1, beta-2, alpha-1 | Inotropy + vasoconstriction | Low dose = inotropy; high dose = vasoconstriction |
| **Phenylephrine** | 0.1-3.0 mcg/kg/min | Pure alpha-1 | Vasoconstriction only | Risk of reflex bradycardia; avoid in low CO states |
| **Dopamine** | 2-20 mcg/kg/min | Dose-dependent | Dose-dependent | 1-3: dopaminergic (renal); 3-10: beta-1 (inotropy); >10: alpha-1 (vasoconstriction) |
| **Dobutamine** | 2.5-20 mcg/kg/min | Beta-1 >> beta-2 | Inotropy | May cause hypotension (beta-2); use with vasopressor |
| **Milrinone** | 0.125-0.75 mcg/kg/min | PDE-3 inhibitor | Inotropy + vasodilation | Inodilator; renally cleared; may cause hypotension |
| **Angiotensin II** | 1.25-40 ng/kg/min | AT1 receptor | Vasoconstriction | Rescue vasopressor for refractory shock |

**Norepinephrine Equivalent Dose (NED) for comparison:**
- Norepinephrine 0.1 mcg/kg/min = ~1 NED
- Epinephrine 0.1 mcg/kg/min = ~1 NED
- Dopamine 15 mcg/kg/min = ~1 NED
- Vasopressin 0.04 U/min = ~0.4 NED
- Phenylephrine 1.0 mcg/kg/min = ~0.1 NED

### 4.2 Sedation

| Drug | Bolus Dose | Infusion Range | Onset | Half-life | Notes |
|------|-----------|---------------|-------|-----------|-------|
| **Propofol** | 0.25-1 mg/kg | 5-80 mcg/kg/min | 15-30 sec | 3-12 hr (context-sensitive) | Monitor triglycerides q48-72h; risk of propofol infusion syndrome (PRIS) at high doses >5 mg/kg/hr for >48h; provides no analgesia |
| **Midazolam** | 0.01-0.05 mg/kg | 0.02-0.1 mg/kg/hr | 2-5 min | 3-11 hr (prolonged in renal/hepatic impairment) | Active metabolite accumulates; associated with worse delirium outcomes |
| **Dexmedetomidine** | No bolus recommended (risk of bradycardia/hypotension) | 0.2-1.5 mcg/kg/hr | 15 min | 2 hr | Alpha-2 agonist; allows arousable sedation; no respiratory depression; risk of bradycardia and hypotension |
| **Ketamine** | 0.5-2 mg/kg | 0.5-2 mg/kg/hr | 30-60 sec | 2-3 hr | Dissociative; bronchodilator; maintains BP; may increase ICP (controversial); emergence reactions |
| **Lorazepam** | 0.02-0.04 mg/kg | 0.01-0.1 mg/kg/hr | 5-15 min | 10-20 hr | Contains propylene glycol (monitor osmolar gap); avoid in continuous infusion if possible |

**RASS (Richmond Agitation-Sedation Scale):**

| Score | Term | Description |
|-------|------|-------------|
| +4 | Combative | Violent, danger to staff |
| +3 | Very agitated | Pulls tubes/lines, aggressive |
| +2 | Agitated | Frequent non-purposeful movement |
| +1 | Restless | Anxious, not aggressive |
| 0 | Alert and calm | — |
| -1 | Drowsy | Not fully alert, sustained awakening to voice (>10 sec) |
| -2 | Light sedation | Briefly awakens to voice (<10 sec) |
| -3 | Moderate sedation | Movement or eye opening to voice (no eye contact) |
| -4 | Deep sedation | No response to voice, movement to physical stimulation |
| -5 | Unarousable | No response to voice or physical stimulation |

**ICU target:** RASS -2 to 0 (light sedation) per PADIS guidelines
**Exception:** Deep sedation (RASS -4 to -5) for neuromuscular blockade, severe ARDS with proning, status epilepticus, elevated ICP

### 4.3 Analgesia

| Drug | Bolus Dose (IV) | Infusion Range | Equianalgesic (IV) | Notes |
|------|----------------|---------------|-------------------|-------|
| **Fentanyl** | 25-100 mcg q1-2h | 25-200 mcg/hr | 100 mcg | Short-acting; preferred in hemodynamic instability (no histamine release); lipophilic (accumulates with prolonged use) |
| **Morphine** | 2-4 mg q1-4h | 2-30 mg/hr | 10 mg | Histamine release (hypotension, bronchospasm); active metabolite (M6G) accumulates in renal failure; avoid in renal impairment |
| **Hydromorphone** | 0.2-1.0 mg q1-2h | 0.2-3 mg/hr | 1.5 mg | 5-7x more potent than morphine; less histamine release; preferred over morphine in renal impairment |
| **Remifentanil** | not typically bolused | 0.5-15 mcg/kg/hr | — | Ultra-short acting (esterase metabolism); does not accumulate; ideal for neuro assessments |
| **Acetaminophen (IV)** | 1000 mg q6h (max 4g/day) | — | — | Adjunct analgesic; avoid in liver failure; reduce to 2g/day in hepatic impairment or weight <50 kg |
| **Ketorolac** | 15-30 mg IV q6h | — | — | NSAID; max 5 days; avoid in renal impairment, GI bleed risk, coagulopathy, post-CABG |

**Pain Assessment Tools:**
- Verbal patients: Numeric Rating Scale (NRS) 0-10
- Non-verbal/intubated: BPS (Behavioral Pain Scale, 3-12) or CPOT (Critical-Care Pain Observation Tool, 0-8)
- Target: NRS <4, BPS <6, CPOT <3

### 4.4 Antibiotics (Common ICU Choices)

| Drug | Standard Dose | Renal Adjustment | Spectrum | Common Indications |
|------|--------------|-----------------|----------|-------------------|
| **Vancomycin** | 15-20 mg/kg IV q8-12h (AUC target 400-600) | Adjust by AUC/trough; may need q24-48h in renal failure | MRSA, gram-positive | Empiric MRSA coverage, endocarditis, meningitis |
| **Piperacillin-Tazobactam** | 4.5 g IV q6h (or 3.375 g q6h) | 2.25 g q6h (CrCl 20-40); 2.25 g q8h (CrCl <20) | Broad gram-neg + anaerobes | Empiric intra-abdominal, nosocomial pneumonia |
| **Meropenem** | 1-2 g IV q8h | 1 g q12h (CrCl 26-50); 500 mg q12h (CrCl 10-25) | Broadest beta-lactam | ESBL organisms, severe intra-abdominal, meningitis (2g q8h) |
| **Cefepime** | 2 g IV q8h | 1 g q12h (CrCl 30-60); 1 g q24h (CrCl 11-29) | Extended gram-neg | Nosocomial pneumonia, febrile neutropenia |
| **Ceftriaxone** | 1-2 g IV q24h | No renal adjustment | Community gram-neg | Community-acquired pneumonia, meningitis (2g q12h) |
| **Metronidazole** | 500 mg IV q8h | No renal adjustment (reduce in severe hepatic) | Anaerobes, C. difficile | Intra-abdominal (with gram-neg coverage), C. diff |
| **Azithromycin** | 500 mg IV q24h | No renal adjustment | Atypicals | Community-acquired pneumonia (with beta-lactam) |
| **Levofloxacin** | 750 mg IV q24h | 750 mg q48h (CrCl 20-49) | Broad (respiratory) | CAP, UTI; avoid in myasthenia gravis |
| **Linezolid** | 600 mg IV q12h | No renal adjustment | MRSA, VRE | Alternative to vancomycin; monitor for serotonin syndrome, thrombocytopenia after 10-14 days |
| **Micafungin** | 100-150 mg IV q24h | No renal adjustment | Candida spp. | Invasive candidiasis, empiric antifungal |
| **Fluconazole** | 400-800 mg IV q24h (loading 800 mg) | 50% dose reduction if CrCl <50 | Candida (not krusei/glabrata) | Candidemia (susceptible strains), fungal prophylaxis |
| **Amphotericin B (liposomal)** | 3-5 mg/kg IV q24h | Use cautiously in renal impairment | Broadest antifungal | Invasive aspergillosis, mucormycosis, refractory candidiasis |
| **TMP-SMX** | 5 mg/kg IV q6-8h (TMP component) | Avoid if CrCl <15 | PJP, MRSA, Stenotrophomonas | PCP prophylaxis/treatment, MRSA skin/soft tissue |
| **Colistin (polymyxin E)** | 5 mg/kg loading, then 2.5 mg/kg q12h (CBA) | Renal dose adjust critical | MDR gram-negatives | Last-resort for XDR Pseudomonas, Acinetobacter |

**Empiric Regimens (Common ICU Scenarios):**
- Sepsis (unknown source): vancomycin + piperacillin-tazobactam (or meropenem)
- Hospital-acquired pneumonia: vancomycin + cefepime (or piperacillin-tazobactam)
- Intra-abdominal: piperacillin-tazobactam or meropenem + /- vancomycin
- Meningitis: vancomycin + ceftriaxone + ampicillin (if >50 yr or immunocompromised)
- Febrile neutropenia: cefepime or meropenem

### 4.5 Anticoagulation

| Drug | Dose | Monitoring | Target | Notes |
|------|------|-----------|--------|-------|
| **Heparin (UFH) drip** | 80 U/kg bolus, then 18 U/kg/hr | aPTT q6h until therapeutic x2, then q12-24h | aPTT 60-80 sec (1.5-2.5x normal) or anti-Xa 0.3-0.7 | Nomogram-based titration; check platelets q2-3 days (HIT risk) |
| **Enoxaparin (prophylaxis)** | 40 mg SC q24h (or 30 mg q12h) | Anti-Xa if needed (trough 0.2-0.5) | — | Reduce dose for CrCl <30 (30 mg q24h); unreliable SC absorption with vasopressors |
| **Enoxaparin (therapeutic)** | 1 mg/kg SC q12h or 1.5 mg/kg q24h | Anti-Xa (peak 0.6-1.0 for q12h) | — | Avoid if CrCl <30 (use UFH instead) |
| **Apixaban** | 5 mg PO BID (2.5 mg BID if criteria met) | No routine monitoring | — | Not easily reversed (andexanet alfa available); avoid in severe renal/hepatic failure |
| **Bivalirudin** | 0.15-0.25 mg/kg/hr | aPTT q2h initially | aPTT 1.5-2.5x normal | Alternative in HIT; short half-life (25 min) |
| **Argatroban** | 0.5-2 mcg/kg/min | aPTT q2h initially | aPTT 1.5-3x normal | Hepatically cleared (reduce in liver failure); use in HIT |

### 4.6 Insulin

**IV Insulin Drip Protocol (typical):**
- **Indication:** BG >180 mg/dL in ICU, DKA, HHS
- **Target glucose:** 140-180 mg/dL (most protocols); some use 110-140 mg/dL
- **Starting rate:** BG / 100 = units/hr (e.g., BG 300 -> 3 units/hr)
- **Titration:** Check BG q1h; adjust rate by 0.5-2 U/hr based on BG change
- **Hypoglycemia protocol:** BG <70 = stop drip, give D50W 25 mL (12.5 g), recheck in 15 min

**Sliding Scale Insulin (subcutaneous, for non-critical patients):**

| Blood Glucose (mg/dL) | Low-Dose Scale | Medium-Dose Scale | High-Dose Scale |
|-----------------------|---------------|-------------------|-----------------|
| <70 | Hypoglycemia protocol | Hypoglycemia protocol | Hypoglycemia protocol |
| 70-140 | 0 units | 0 units | 0 units |
| 141-180 | 2 units | 4 units | 6 units |
| 181-220 | 4 units | 6 units | 8 units |
| 221-260 | 6 units | 8 units | 10 units |
| 261-300 | 8 units | 10 units | 12 units |
| 301-350 | 10 units | 12 units | 14 units |
| >350 | 12 units + call MD | 14 units + call MD | 16 units + call MD |

**DKA-Specific Protocol:**
- IV insulin drip 0.1-0.14 U/kg/hr (NO bolus per ADA guidelines)
- Transition to SC insulin when: BG <200, anion gap closed, pH >7.30, HCO3 >15, patient eating
- Overlap IV and SC insulin by 1-2 hours to prevent rebound DKA

---

## 5. Ventilator Settings and Modes

### 5.1 Ventilator Modes

| Mode | Abbreviation | Control Variable | Patient Triggers? | Typical Use |
|------|-------------|-----------------|-------------------|-------------|
| Assist-Control / Volume Control | AC/VC | Volume | Yes (patient can trigger additional breaths at set Vt) | Full support; most common initial mode |
| Assist-Control / Pressure Control | AC/PC | Pressure | Yes | Full support; decelerating flow pattern |
| Synchronized Intermittent Mandatory Ventilation | SIMV | Volume or Pressure | Yes (spontaneous breaths above set rate get PS only) | Weaning (less commonly used now) |
| Pressure Support Ventilation | PSV | Pressure | Patient-triggered only | Weaning, spontaneous breathing trial |
| Airway Pressure Release Ventilation | APRV | Pressure | Spontaneous breathing encouraged | Severe ARDS, rescue mode |
| Volume-Targeted Pressure Control | PRVC | Pressure (auto-adjusts to target volume) | Yes | Combines benefits of PC and VC |

**Non-Invasive Ventilation:**

| Mode | Settings | Indication |
|------|---------|-----------|
| BiPAP | IPAP 8-20 cmH2O, EPAP 4-10 cmH2O | COPD exacerbation, cardiogenic pulmonary edema, OSA |
| CPAP | 5-15 cmH2O | OSA, mild hypoxemia, post-extubation |
| HFNC | Flow 20-60 L/min, FiO2 21-100% | Mild-moderate hypoxemia, post-extubation, comfort |

### 5.2 Key Ventilator Settings

| Setting | Range | Typical Initial | Notes |
|---------|-------|----------------|-------|
| FiO2 | 21-100% | Start 100%, wean to target SpO2 92-96% | Goal: lowest FiO2 maintaining adequate oxygenation; FiO2 >60% for prolonged periods causes O2 toxicity |
| PEEP | 0-24 cmH2O | 5 cmH2O (standard); 10-18 in ARDS | Recruits alveoli; improves oxygenation; may decrease preload/CO; use ARDSnet PEEP/FiO2 table |
| Tidal Volume (Vt) | 4-8 mL/kg IBW | 6-8 mL/kg IBW (lung-protective) | ARDS: strict 4-6 mL/kg IBW; IBW based on height, not actual weight |
| Respiratory Rate | 10-35 breaths/min | 14-18 | Adjust for minute ventilation and PaCO2 target |
| Pressure Support (PS) | 5-20 cmH2O | 10-15 cmH2O | Above PEEP; for spontaneous breaths; wean toward 5 cmH2O |
| I:E Ratio | 1:1 to 1:4 | 1:2 | Inverse ratio (2:1) used in APRV; auto-PEEP risk if E too short |
| Plateau Pressure | <30 cmH2O target | — | Measure with inspiratory hold; >30 = risk of barotrauma |
| Driving Pressure | <15 cmH2O target | — | Plateau pressure - PEEP; strongest predictor of ARDS mortality |

**Ideal Body Weight (IBW) Calculation:**
- Male: 50 + 2.3 x (height in inches - 60)
- Female: 45.5 + 2.3 x (height in inches - 60)

### 5.3 ARDSnet Low Tidal Volume Protocol

| FiO2 | PEEP (lower table) | PEEP (higher table) |
|------|-------|---------|
| 0.30 | 5 | 5 |
| 0.40 | 5-8 | 8-10 |
| 0.50 | 8-10 | 10-12 |
| 0.60 | 10 | 12-14 |
| 0.70 | 10-14 | 14-16 |
| 0.80 | 14 | 16-18 |
| 0.90 | 14-18 | 18-22 |
| 1.00 | 18-24 | 22-24 |

### 5.4 Weaning and Extubation Criteria

**Readiness for Spontaneous Breathing Trial (SBT):**
- FiO2 <= 40%
- PEEP <= 8 cmH2O (some use <=5)
- Adequate oxygenation: PaO2 >= 60 mmHg on above settings (P/F >= 150-200)
- Hemodynamically stable (no/minimal vasopressors)
- Adequate mentation (follows commands)
- No active myocardial ischemia
- Respiratory drive present
- Cuff leak present (if concern for post-extubation stridor)

**SBT Methods:**
- T-piece trial: 30-120 min of spontaneous breathing
- Pressure support 5-8 cmH2O + PEEP 5
- Automatic tube compensation (ATC)

**Passing SBT (extubation criteria):**
- RR <35 for 30-120 min
- SpO2 >90%
- HR does not increase >20% from baseline
- No diaphoresis, accessory muscle use, paradoxical breathing
- Rapid Shallow Breathing Index (RSBI) = RR/Vt(L) <= 105
- Adequate cough strength
- Intact gag reflex
- Manageable secretions
- GCS >= 8 with intact airway reflexes

**Failing SBT indicators:**
- RR >35 for >5 min
- SpO2 <90%
- HR >140 or change >20%
- SBP >180 or <90
- Agitation, diaphoresis, distress

---

## 6. Monitoring Frequency

### 6.1 Vital Signs

| Acuity Level | Frequency | Circumstances |
|-------------|-----------|---------------|
| ICU standard | q1h (every hour) | Most ICU patients; includes HR, BP, RR, SpO2, temp |
| Unstable / acute resuscitation | q15 min or continuous | Active shock, new vasopressor titration, post-code, active hemorrhage |
| Improving / step-down eligible | q2h | Stable for >24h, weaning vasopressors |
| Continuous monitoring | Real-time | Telemetry (HR, rhythm), arterial line (beat-to-beat BP), SpO2 (pulse oximeter) — continuously displayed on bedside monitor |

**Note:** Most ICU patients have continuous cardiac monitoring with alarms. The "q1h" frequency refers to nurse-documented vital signs, not monitoring frequency.

### 6.2 Laboratory Studies

| Lab Panel | Standard Frequency | Circumstances for More Frequent |
|-----------|-------------------|-------------------------------|
| BMP (electrolytes) | Daily (q24h AM draw, typically 4-5 AM) | q6h in DKA, q4-6h with insulin drip, q6h with active electrolyte replacement |
| CBC | Daily | q6h with active bleeding, q12h post-transfusion |
| ABG | PRN (per respiratory changes) | q2-4h on ventilator with FiO2 changes, q1-2h in DKA, q4-6h routine on ventilator |
| Lactate | q2-6h in sepsis | q2h if >2 mmol/L (trending to clearance) |
| Coagulation (PT/INR, aPTT) | Daily if on anticoagulation | q6h with heparin drip (until therapeutic x2), q6h in DIC |
| LFTs | Daily to q48h | More frequent in hepatic failure or hepatotoxic drugs |
| Troponin | Serial q3-6h (ACS rule-out) | q6-8h x3 for ACS workup, then daily if elevated |
| Procalcitonin | q24-48h (if trending) | At admission, then q48h to guide antibiotic de-escalation |
| Blood glucose | q1h (on insulin drip) | q4-6h with sliding scale, q1h with IV insulin |
| Blood cultures | At fever onset | Before antibiotics; repeat in 48-72h if persistent bacteremia |
| Magnesium/Phosphorus | Daily | q6h if replacing aggressively, daily in DKA |
| Drug levels (vancomycin, etc.) | Per protocol | Vancomycin AUC q24-48h; aminoglycoside peak/trough |

### 6.3 Triggers for Increased Monitoring Frequency

- New vasopressor or dose change → q15 min vitals for 1-2 hours
- Ventilator setting change → ABG in 30-60 min
- Blood product transfusion → vitals q15 min during transfusion
- New onset arrhythmia → continuous monitoring + 12-lead ECG
- Deteriorating mental status → neuro checks q1-2h (was q4h)
- Active bleeding → q15 min vitals, serial H/H q4-6h
- Post-procedure (central line, chest tube, intubation) → q15 min x4, then q1h
- Seizure → continuous EEG, neuro checks q1h

---

## 7. Clinical Trajectories

### 7.1 Sepsis

**Improving trajectory (typical 3-7 days):**
- Day 0-1: Initial resuscitation — fluids, vasopressors started, antibiotics, lactate elevated (4-6), HR 110-130, MAP <65 initially
- Day 1-2: Lactate trending down (4 -> 2.5 -> 1.8), vasopressor weaning begins, MAP stable >65 off pressors, HR normalizing (100-110), WBC may paradoxically rise initially (12 -> 18 -> then down), fever breaking
- Day 2-3: Off vasopressors, lactate normal (<2), WBC trending down (18 -> 12), creatinine peaking then improving, procalcitonin declining
- Day 3-5: Afebrile >24h, WBC normalizing (8-10), tolerating oral/enteral nutrition, mental status improving, considering antibiotic de-escalation
- Day 5-7: Extubation if intubated, transfer to floor, antibiotic duration determination (typically 7-10 days)

**Worsening trajectory:**
- Day 0-1: Escalating vasopressor doses despite fluids, rising lactate (2 -> 4 -> 8), progressive organ dysfunction (rising creatinine, bilirubin, INR)
- Day 1-2: Multi-organ failure: ARDS developing (worsening P/F ratio), DIC (rising D-dimer, falling fibrinogen/platelets, rising INR), AKI progressing (oliguria, need for CRRT), liver injury (rising bilirubin/transaminases)
- Day 2-3: Refractory shock (multiple vasopressors at high doses), metabolic acidosis (pH <7.2), lactate >10, SOFA score rising
- Day 3+: Goals of care discussion, potential transition to comfort measures

### 7.2 STEMI

**Improving:**
- Hour 0-6: Reperfusion therapy (PCI), troponin rising (expected), hemodynamics stabilizing
- Day 1-2: Troponin peaks then begins to fall, hemodynamically stable, starting beta-blocker/ACEi/statin/aspirin/P2Y12
- Day 2-3: Troponin trending down, ambulating, echocardiogram to assess EF
- Day 3-5: Transfer to telemetry, cardiac rehab referral, discharge planning

**Worsening:**
- Hour 0-6: Cardiogenic shock post-MI (low CO, rising lactate, MAP <65)
- Day 1-2: Mechanical complications (papillary muscle rupture, VSD, free wall rupture), worsening heart failure, need for IABP or Impella
- Day 2-5: Persistent cardiogenic shock, multi-organ failure, arrhythmias (VT/VF)

### 7.3 DKA

**Improving (typical 12-24h resolution):**
- Hour 0-4: IV fluids (NS 1-2 L/hr initially), insulin drip, K+ replacement, glucose dropping 50-100 mg/dL/hr, anion gap beginning to close
- Hour 4-8: Glucose <250 (switch to D5W + insulin), anion gap closing (from >20 toward 12), pH improving (7.15 -> 7.25), HCO3 rising
- Hour 8-12: pH >7.30, HCO3 >15, anion gap closed (<12), glucose 150-250
- Hour 12-24: Transition to SC insulin (overlap 1-2h), tolerating PO, electrolytes normalizing

**Worsening:**
- Hour 0-6: Cerebral edema (rare in adults, more common in pediatric), refractory acidosis (pH <7.0 despite insulin), severe hypokalemia (<2.5 despite replacement, risk of arrhythmia), ARDS
- Hour 6+: Multi-organ failure, persistent acidosis suggests underlying trigger (MI, sepsis)

### 7.4 Acute Respiratory Failure / ARDS

**Improving:**
- Day 0-3: Intubation, lung-protective ventilation, FiO2 and PEEP titration, P/F ratio stable or improving
- Day 3-7: FiO2 weaning (from 80% -> 60% -> 40%), PEEP weaning (from 14 -> 10 -> 8), P/F ratio improving (100 -> 150 -> 200+)
- Day 7-14: SBT attempts, PSV weaning, extubation if passing SBT
- Day 14+: If still intubated, tracheostomy discussion (typically day 10-14)

**Worsening:**
- Day 0-3: Escalating FiO2 and PEEP, worsening P/F ratio, need for proning (16h/day), paralysis
- Day 3-7: Refractory hypoxemia (P/F <80 despite max settings), pneumothorax risk, progressive multi-organ failure
- Day 7+: Fibroproliferative phase, prolonged ventilator dependence, ventilator-associated pneumonia

### 7.5 GI Bleed

**Improving:**
- Hour 0-6: Resuscitation (fluids, blood products), hemodynamics stabilizing, HR normalizing, endoscopy within 24h
- Hour 6-24: Post-endoscopic intervention, H/H stabilizing (no further drop), no ongoing transfusion needs
- Day 1-2: Hemodynamically stable, H/H stable on serial checks (q8-12h), tolerating diet, PPI drip (72h for high-risk ulcers)
- Day 2-3: Step-down or floor transfer

**Worsening:**
- Hour 0-6: Massive transfusion protocol activated (>6 units PRBC in 24h), hemodynamic instability despite resuscitation, H/H continues to drop
- Hour 6-24: Endoscopy fails to control bleeding, interventional radiology or surgery needed
- Day 1-3: Rebleeding (occurs in 10-20% of high-risk lesions), need for repeat endoscopy, transfusion-related complications (TRALI, TACO)

---

## 8. Nursing Assessments

### 8.1 Standard ICU Nursing Documentation (per shift, q4h, or more frequently)

**Neurological:**
- Glasgow Coma Scale (GCS): Eye (1-4) + Verbal (1-5) + Motor (1-6) = 3-15
- RASS / sedation level
- Pupil size (mm) and reactivity (brisk, sluggish, fixed)
- Orientation (person, place, time, situation)
- Motor strength bilateral (0-5 scale per extremity)
- Cranial nerve assessment (if neuro patient)
- Presence of seizure activity
- CAM-ICU (Confusion Assessment Method for the ICU) — delirium screening q12h

**Cardiovascular:**
- Heart rhythm (monitor interpretation)
- Peripheral pulses (radial, dorsalis pedis, posterior tibial) — 0-3+ scale
- Capillary refill (<2 sec normal, >3 sec = poor perfusion)
- Skin color, temperature, moisture (warm/dry = good; cool/mottled = poor perfusion)
- Edema (0-4+ pitting scale, location)
- Central line / arterial line site assessment
- Vasopressor doses (documented q1h with vitals)

**Respiratory:**
- Breath sounds (clear, diminished, crackles, wheezes, rhonchi — bilateral comparison)
- Work of breathing (accessory muscle use, nasal flaring, retractions)
- Ventilator settings (documented q1-2h)
- ETT position (cm at lip, cuff pressure q8h: target 20-30 cmH2O)
- Secretion amount, color, consistency
- Chest tube output (amount, color, air leak)

**Gastrointestinal:**
- Bowel sounds (present/absent in all 4 quadrants)
- Abdominal assessment (soft, distended, rigid, tender)
- NG/OG tube output (amount, color)
- Stool (frequency, amount, color, consistency — Bristol scale)
- Enteral feeding tolerance (residual volume, abdominal distension)
- Last bowel movement

**Renal / Fluid Balance:**
- Intake and Output (I&O) — hourly in ICU
  - Intake: IV fluids, medications, blood products, enteral feeds, PO intake
  - Output: urine (hourly via Foley), drains, NG output, stool, chest tube, emesis
- Net fluid balance (calculated q12h and q24h)
- Urine output target: >=0.5 mL/kg/hr (>=30 mL/hr for average adult)
- Urine color and characteristics
- Daily weight (gold standard for fluid status)
- Foley catheter site assessment

**Skin / Wound:**
- Braden Scale for pressure injury risk (q24h; score 6-23; <=18 = at risk)
- Skin integrity assessment (all pressure points: sacrum, heels, occiput, ears)
- Wound assessment: size (L x W x D in cm), drainage, wound bed, periwound skin
- Surgical incision assessment (approximation, drainage, signs of infection)
- IV site assessment (PIV, central line, arterial line) q4h

**Pain:**
- Pain score (NRS 0-10 verbal; BPS 3-12 or CPOT 0-8 non-verbal)
- Pain location, quality, aggravating/alleviating factors
- Response to pain interventions (documented within 30-60 min of intervention)

**Psychosocial:**
- Anxiety level
- Family communication / visitation
- Advance directives / code status
- Restraint assessment (if applicable) — q1-2h: circulation, sensation, skin integrity

### 8.2 Glasgow Coma Scale (GCS) Detail

| Component | Response | Score |
|-----------|----------|-------|
| **Eye Opening** | Spontaneous | 4 |
| | To voice | 3 |
| | To pain | 2 |
| | None | 1 |
| **Verbal** | Oriented | 5 |
| | Confused | 4 |
| | Inappropriate words | 3 |
| | Incomprehensible sounds | 2 |
| | None | 1 |
| **Motor** | Obeys commands | 6 |
| | Localizes pain | 5 |
| | Withdrawal (flexion) | 4 |
| | Abnormal flexion (decorticate) | 3 |
| | Extension (decerebrate) | 2 |
| | None | 1 |

**Clinical significance:**
- GCS 15: fully alert
- GCS 13-14: mild brain injury
- GCS 9-12: moderate brain injury
- GCS 3-8: severe brain injury (intubation typically required at <=8)

---

## 9. ICU Scoring Systems

### 9.1 APACHE II (Acute Physiology and Chronic Health Evaluation)

**Components (scored within first 24h of ICU admission):**

**A. Acute Physiology Score (APS) — 12 variables, each scored 0-4:**

| Variable | 0 points | 1 point | 2 points | 3 points | 4 points |
|----------|----------|---------|----------|----------|----------|
| Temperature (C) | 36-38.4 | 34-35.9 or 38.5-38.9 | 32-33.9 | 30-31.9 or 39-40.9 | <30 or >41 |
| MAP (mmHg) | 70-109 | — | 50-69 or 110-129 | — | <50 or 130-159 or >=160 |
| Heart Rate | 70-109 | 55-69 or 110-139 | 40-54 or 140-179 | — | <40 or >=180 |
| Respiratory Rate | 12-24 | 10-11 or 25-34 | 6-9 | — | <6 or >=35 |
| Oxygenation | A-a gradient <200 (FiO2>=50%) or PaO2 >70 (FiO2<50%) | — | PaO2 61-70 or A-a 200-349 | A-a 350-499 | PaO2 <55 or A-a >=500 |
| Arterial pH | 7.33-7.49 | 7.25-7.32 or 7.50-7.59 | 7.15-7.24 or 7.60-7.69 | — | <7.15 or >=7.70 |
| Sodium | 130-149 | 120-129 or 150-154 | 111-119 or 155-159 | — | <=110 or >=160 |
| Potassium | 3.5-5.4 | 3.0-3.4 or 5.5-5.9 | 2.5-2.9 | — | <2.5 or >=6.0 |
| Creatinine | 0.6-1.4 | <0.6 or 1.5-1.9 | — | 2.0-3.4 | >=3.5 |
| Hematocrit | 30-45.9 | 20-29.9 or 46-49.9 | — | 50-59.9 | <20 or >=60 |
| WBC (thousands) | 3-14.9 | 1-2.9 or 15-19.9 | — | 20-39.9 | <1 or >=40 |
| GCS | Score = 15 - actual GCS | — | — | — | — |

**B. Age Points:**
| Age | Points |
|-----|--------|
| <45 | 0 |
| 45-54 | 2 |
| 55-64 | 3 |
| 65-74 | 5 |
| >=75 | 6 |

**C. Chronic Health Points (5 points if):**
- Severe organ insufficiency or immunocompromised prior to admission
- Specific criteria: liver (cirrhosis/portal HTN/encephalopathy), cardiovascular (NYHA IV), respiratory (chronic restrictive/obstructive/vascular disease), renal (chronic dialysis), immunocompromised

**APACHE II Score = APS + Age Points + Chronic Health Points**
- Range: 0-71
- Score 0-4: ~4% mortality
- Score 5-9: ~8% mortality
- Score 10-14: ~15% mortality
- Score 15-19: ~25% mortality
- Score 20-24: ~40% mortality
- Score 25-29: ~55% mortality
- Score 30-34: ~75% mortality
- Score >=35: ~85% mortality

### 9.2 SOFA Score (Sequential Organ Failure Assessment)

**Scored daily — each organ system 0-4 points:**

| Score | Respiration (PaO2/FiO2) | Coagulation (Platelets x10^3/uL) | Liver (Bilirubin mg/dL) | Cardiovascular | CNS (GCS) | Renal (Creatinine mg/dL or UO) |
|-------|------------------------|--------------------------------|----------------------|----------------|-----------|-------------------------------|
| 0 | >=400 | >=150 | <1.2 | MAP >=70, no vasopressors | 15 | <1.2 |
| 1 | <400 | <150 | 1.2-1.9 | MAP <70 | 13-14 | 1.2-1.9 |
| 2 | <300 | <100 | 2.0-5.9 | Dopamine <=5 or dobutamine (any) | 10-12 | 2.0-3.4 |
| 3 | <200 (with ventilator) | <50 | 6.0-11.9 | Dopamine >5 or epi <=0.1 or norepi <=0.1 | 6-9 | 3.5-4.9 or UO <500 mL/day |
| 4 | <100 (with ventilator) | <20 | >12 | Dopamine >15 or epi >0.1 or norepi >0.1 | <6 | >5.0 or UO <200 mL/day |

**SOFA Total = sum of 6 organ scores (range 0-24)**
- Initial SOFA: baseline organ function
- Delta SOFA: change from admission (increase of >=2 = sepsis definition per Sepsis-3)
- SOFA score predicts ICU mortality:
  - 0-6: <10%
  - 7-9: 15-20%
  - 10-12: 40-50%
  - 13-14: 50-60%
  - 15-24: >80%

### 9.3 Other Common ICU Scores

| Score | Use | Components |
|-------|-----|-----------|
| qSOFA | Sepsis screening (bedside) | RR >=22, altered mentation, SBP <=100 (>=2 positive = concern) |
| CURB-65 | Pneumonia severity | Confusion, Urea >7, RR >=30, BP <90/60, Age >=65 |
| Wells Score | PE probability | Clinical signs of DVT, alternative diagnosis unlikely, HR >100, immobilization/surgery, prior VTE, hemoptysis, cancer |
| MELD | Liver disease severity | Bilirubin, INR, Creatinine, Sodium (MELD-Na) |
| CHA2DS2-VASc | AFib stroke risk | CHF, HTN, Age >=75 (2), DM, Stroke (2), Vascular disease, Age 65-74, Sex (female) |
| Braden Scale | Pressure injury risk | Sensory perception, moisture, activity, mobility, nutrition, friction/shear |
| CIWA-Ar | Alcohol withdrawal | Nausea, tremor, anxiety, agitation, sweats, visual/auditory/tactile disturbances, headache, orientation |
| RASS | Sedation depth | -5 to +4 (see Section 4.2) |
| CAM-ICU | Delirium screening | Acute onset/fluctuation, inattention, disorganized thinking, altered consciousness |
| NIH Stroke Scale | Stroke severity | 11 items: LOC, gaze, visual fields, facial palsy, motor (arms/legs), ataxia, sensory, language, dysarthria, extinction |

---

## 10. Realistic Constraints and Physiological Correlations

### 10.1 Heart Rate and Blood Pressure

- **Compensatory tachycardia:** Hypotension (MAP <65) typically triggers reflex tachycardia (HR increase 20-40 bpm)
- **Hemorrhagic shock classification:**
  - Class I (<15% blood loss): HR normal, BP normal
  - Class II (15-30%): HR 100-120, BP normal (narrowed pulse pressure)
  - Class III (30-40%): HR 120-140, SBP <90
  - Class IV (>40%): HR >140, SBP <70, altered mental status
- **Beta-blockers blunt tachycardic response** — patient may be hypotensive with normal HR
- **Cushing reflex:** Hypertension + bradycardia = elevated ICP (brainstem compression)
- **Neurogenic shock:** Hypotension + bradycardia (spinal cord injury, loss of sympathetic tone)

### 10.2 Temperature and Heart Rate

- **Sinus tachycardia with fever:** HR increases ~10 bpm per 1 degree C above 37 C (or ~8 bpm per 1 degree F above 98.6 F)
- **"Relative bradycardia":** If HR does NOT increase appropriately with fever, consider: beta-blockers, intracellular organisms (typhoid, Legionella, psittacosis), drug fever
- **Hypothermia:** HR decreases; severe hypothermia (<30 C) causes bradycardia, risk of VFib
- **Temperature and WBC are often correlated:** Fever typically accompanies leukocytosis, but septic patients can be hypothermic with leukopenia (worse prognosis)

### 10.3 Vasopressor Dose and Blood Pressure

- **Linear relationship (within range):** Increasing norepinephrine dose generally increases MAP, but there is a ceiling effect
- **Vasopressor-dependent MAP:** If MAP = 55 on norepinephrine 0.3 mcg/kg/min, increasing to 0.5 might raise MAP to 60-65
- **Refractory shock:** If MAP <65 despite norepinephrine >0.5 mcg/kg/min, add vasopressin (0.03-0.04 U/min)
- **Tapering:** When improving, vasopressors are weaned 0.02-0.05 mcg/kg/min every 15-30 min while maintaining MAP >=65
- **Vasopressin is typically weaned last** (or first, depending on institutional protocol)
- **Bolus effect:** When starting vasopressors, BP response is seen within 1-5 min (norepinephrine), vasopressin effect in 5-15 min

### 10.4 Oxygenation, FiO2, and PEEP

- **PaO2 and SpO2 relationship (oxygen-hemoglobin dissociation curve):**
  - PaO2 60 mmHg ~ SpO2 90% (critical threshold — "cliff" of the curve)
  - PaO2 80 mmHg ~ SpO2 95%
  - PaO2 100 mmHg ~ SpO2 98-99%
  - PaO2 40 mmHg ~ SpO2 75% (mixed venous)

- **Right shift (decreased O2 affinity, easier unloading):** acidosis, hyperthermia, elevated 2,3-DPG, hypercarbia
- **Left shift (increased O2 affinity, harder unloading):** alkalosis, hypothermia, decreased 2,3-DPG, CO poisoning

- **FiO2 and PaO2:** Roughly, PaO2 should be ~5x FiO2% (e.g., FiO2 40% -> PaO2 ~200 in healthy lungs). In ARDS, this ratio is the P/F ratio.
- **PEEP and oxygenation:** Each 1 cmH2O PEEP increase may improve PaO2 by 5-10 mmHg (variable, depends on recruitability)
- **PEEP and hemodynamics:** Excessive PEEP decreases venous return, can reduce cardiac output and MAP (especially in hypovolemic patients)
- **FiO2 toxicity:** Prolonged FiO2 >60% causes absorptive atelectasis and oxygen toxicity; always wean FiO2 before PEEP

### 10.5 Renal Function and Electrolytes

- **Rising creatinine predicts electrolyte derangements:**
  - Hyperkalemia (most dangerous — cardiac risk at K >6.0)
  - Metabolic acidosis (loss of HCO3 regeneration)
  - Hyperphosphatemia (impaired excretion)
  - Hypocalcemia (from hyperphosphatemia — calcium-phosphate product)
  - Hyponatremia or hypernatremia (depending on volume status)
  - Elevated BUN (uremia — if BUN >100, consider dialysis)

- **Urine output as early AKI indicator:**
  - Normal: >0.5 mL/kg/hr
  - Oliguria: <0.5 mL/kg/hr for >=6h (KDIGO Stage 1)
  - Anuria: <100 mL/24h (KDIGO Stage 3)

- **Creatinine lag:** Creatinine may not rise for 24-48h after renal insult; urine output changes first
- **Medication dosing:** Many drugs require renal dose adjustment when CrCl <30 (see antibiotic table)

### 10.6 Liver Function and Coagulation

- **Synthetic function decline → coagulopathy:**
  - Liver produces clotting factors (II, V, VII, IX, X), fibrinogen, and anticoagulants (protein C, S, antithrombin)
  - Rising INR correlates with worsening liver function (INR >1.5 = significant hepatic dysfunction)
  - Factor VII has shortest half-life (6h) — INR rises first
  - Low albumin (<2.5) indicates chronic or severe liver dysfunction

- **Portal hypertension consequences:** ascites, varices (GI bleed risk), splenomegaly (thrombocytopenia from sequestration)
- **Hepatorenal syndrome:** liver failure leads to renal failure (rising creatinine without structural kidney damage)
- **Ammonia and encephalopathy:** elevated ammonia correlates loosely with hepatic encephalopathy grade
- **Drug metabolism:** hepatically cleared drugs (midazolam, propofol, acetaminophen, metronidazole) accumulate in liver failure

### 10.7 Fluid Balance and Hemodynamics

- **Positive fluid balance >10% body weight associated with worse outcomes in ICU**
- **Third-spacing:** In sepsis, pancreatitis, burns — fluid shifts to interstitial space; intravascular volume drops despite positive fluid balance
- **Daily weight change:** 1 kg = ~1 L fluid
- **CVP correlation with volume status is unreliable** (except at extremes)
- **Passive leg raise (PLR):** Fluid responsiveness test — raise legs 45 degrees, if CO increases >10%, patient will respond to fluids
- **Stroke volume variation (SVV):** >13% in mechanically ventilated patients suggests fluid responsiveness

### 10.8 Acid-Base and Respiratory Compensation

- **Metabolic acidosis triggers respiratory compensation:** pH drop stimulates hyperventilation (Kussmaul breathing in DKA — RR 30-40+)
- **Compensation limits:** Respiratory system can lower PaCO2 to ~10-15 mmHg minimum; if expected PaCO2 by Winters formula does not match actual, there is a concurrent respiratory disorder
- **Metabolic alkalosis suppresses respiratory drive:** Hypoventilation, but PaCO2 rarely rises above 55 as compensation
- **Chronic vs. acute:** Chronic conditions (COPD) have renal compensation — elevated HCO3 with elevated PaCO2 and near-normal pH

### 10.9 Sedation, Analgesia, and Hemodynamics

- **Propofol causes dose-dependent hypotension** (vasodilation + myocardial depression); expect MAP drop 10-20% at induction
- **Dexmedetomidine causes bradycardia and hypotension** (alpha-2 agonist effect); avoid boluses
- **Fentanyl is hemodynamically neutral** (preferred in unstable patients)
- **Morphine causes histamine release** → hypotension, bronchospasm
- **Ketamine increases HR and BP** (sympathomimetic); useful in hypotensive patients needing sedation
- **Over-sedation (RASS -4/-5) masks neurological exam** and is associated with prolonged intubation, delirium, and increased mortality

### 10.10 Hematological Correlations

- **Hemoglobin and oxygen delivery:** O2 delivery = CO x (1.34 x Hb x SaO2 + 0.003 x PaO2); low Hb = compensatory tachycardia to maintain O2 delivery
- **Transfusion threshold:** Hb <7 g/dL (restrictive, most ICU patients); <8 g/dL (cardiac patients, active ACS)
- **Platelet count and bleeding risk:**
  - >50k: safe for most procedures
  - >100k: safe for neurosurgery, ophthalmologic surgery
  - <20k: spontaneous bleeding risk, transfuse
  - <10k: high risk of spontaneous intracranial hemorrhage
- **DIC cascade:** Infection/trauma → coagulation activation → consumption of factors and platelets → bleeding AND thrombosis simultaneously

---

## Appendix: Quick Reference — Critical Values Requiring Immediate Action

| Parameter | Critical Value | Immediate Action |
|-----------|---------------|-----------------|
| HR | <40 or >150 | Atropine / cardioversion / code blue |
| SBP | <80 or >200 | Fluid bolus + vasopressors / antihypertensives |
| MAP | <60 | Start or escalate vasopressors |
| SpO2 | <85% | Increase FiO2, check ETT position, consider reintubation |
| RR | <8 or >35 | Naloxone (if opioid), prepare for intubation |
| Temperature | <34 C or >41 C | Active warming / cooling, blood cultures |
| Potassium | <2.5 or >6.5 mEq/L | IV replacement / calcium gluconate + insulin + glucose + kayexalate + emergent dialysis |
| Sodium | <120 or >160 mEq/L | Hypertonic saline (hyponatremia) / free water (hypernatremia); correct slowly (<8-10 mEq/L per 24h to prevent osmotic demyelination) |
| Glucose | <40 or >600 mg/dL | D50W bolus / insulin drip + fluids |
| pH | <7.15 or >7.60 | Ventilator adjustment, consider bicarb (controversial), treat underlying cause |
| Lactate | >4 mmol/L | Aggressive resuscitation, identify source |
| Hemoglobin | <7 g/dL | Transfuse PRBC (or <8 in ACS) |
| Platelets | <20k | Transfuse platelets |
| INR | >5 (unintended) | Vitamin K, FFP, or PCC (4-factor) |
| Troponin | Any elevation + symptoms | Cardiology consult, serial troponins, heparin, cath lab if STEMI |

---

*This specification is intended for use in building a realistic patient state simulator for narrative interference experiments. All values represent standard clinical practice and may vary by institution, patient population, and clinical context. Not intended for actual clinical decision-making.*
