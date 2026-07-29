# Hospital ICU Clinical Trajectories: Compiled Reference

> **Purpose**: This document is a compiled reference for the narrative interference generator.
> It contains detailed clinical trajectories for 22 ICU diagnoses, including specific numerical
> values for vital signs and labs at presentation, and hour-by-hour or day-by-day trajectories
> for both improving and worsening cases. All numbers are drawn from clinical guidelines,
> published studies, and evidence-based reviews.
>
> **Compiled from**: Three parallel web research tasks covering common ICU diagnoses (7),
> moderate-frequency ICU diagnoses (7), and less common but ICU-relevant diagnoses (8).

---

## Table of Contents

1. [Acute Kidney Injury (AKI)](#acute-kidney-injury-aki)
2. [Acute Liver Failure](#acute-liver-failure)
3. [Acute MI / STEMI](#acute-mi--stemi)
4. [Acute Pancreatitis](#acute-pancreatitis)
5. [Alcohol Withdrawal / Delirium Tremens](#alcohol-withdrawal--delirium-tremens)
6. [Anaphylaxis](#anaphylaxis)
7. [ARDS / Acute Respiratory Failure](#ards--acute-respiratory-failure)
8. [Bacterial Meningitis](#bacterial-meningitis)
9. [Cardiogenic Shock](#cardiogenic-shock)
10. [COPD Exacerbation](#copd-exacerbation)
11. [DKA (Diabetic Ketoacidosis)](#dka-diabetic-ketoacidosis)
12. [Drug Overdose (Opioid)](#drug-overdose-opioid)
13. [GI Bleed](#gi-bleed)
14. [Hemorrhagic Stroke / ICH](#hemorrhagic-stroke--ich)
15. [Hypertensive Emergency](#hypertensive-emergency)
16. [Major Burns](#major-burns)
17. [Massive Transfusion (Trauma)](#massive-transfusion-trauma)
18. [Post-Cardiac Surgery (CABG/Valve)](#post-cardiac-surgery-cabgvalve)
19. [Pulmonary Embolism (PE)](#pulmonary-embolism-pe)
20. [Sepsis / Septic Shock](#sepsis--septic-shock)
21. [Status Epilepticus](#status-epilepticus)
22. [Thyroid Storm](#thyroid-storm)

---

## Acute Kidney Injury (AKI)

### Creatinine Rise Rate
- **Complete renal failure (90% GFR drop):** creatinine rises ~1.8-2.0 mg/dL in the first 24 hours regardless of baseline CKD status
- **Time to 50% creatinine increase:** 4 hours (normal baseline) up to 27 hours (CKD stage 4)
- **Maximum expected daily rise:** ~1.5-2.0 mg/dL/day in anuric complete AKI
- **KDIGO Stage 1:** >=0.3 mg/dL rise within 48 hours OR 1.5-1.9x baseline
- **KDIGO Stage 2:** 2.0-2.9x baseline
- **KDIGO Stage 3:** >=3.0x baseline OR creatinine >=4.0 mg/dL OR initiation of RRT

### Potassium Trajectory
- Potassium rises rapidly in AKI because the increase is acute (unlike CKD where adaptation occurs)
- Mild hyperkalemia: 5.5-6.0 mEq/L -- monitor, restrict intake
- Moderate: 6.0-6.5 mEq/L -- active treatment (insulin/glucose, kayexalate)
- Severe/life-threatening: >6.5 mEq/L -- emergent dialysis consideration
- In oliguric AKI, expect K+ to rise ~0.5 mEq/L/day without intervention

### Urine Output Trajectory
- **Oliguric AKI:** <0.5 mL/kg/hr for >=6 hours (Stage 1), >=12 hours (Stage 2)
- **Severe/Anuric:** <0.3 mL/kg/hr for >=24 hours OR anuria >=12 hours (Stage 3)
- **Non-oliguric AKI:** urine output maintained but creatinine still rises
- Oliguric phase output: 50-500 mL/day total

### BUN Trajectory
- BUN rises proportionally with creatinine but is more variable (affected by catabolism, GI bleed, steroids)
- Typical BUN:Cr ratio ~10-15:1 in intrinsic AKI; >20:1 suggests prerenal

### Dialysis Initiation Thresholds
- **Emergent indications:** K+ >6.5 mEq/L refractory to medical therapy, pH <7.1, pulmonary edema refractory to diuretics, uremic pericarditis, uremic encephalopathy
- **No fixed creatinine or BUN threshold** -- KDIGO recommends clinical context over single values
- BUN >100 mg/dL is often a practical threshold prompting strong consideration

### Recovery Timeline
- **Transient AKI:** reversal within 48 hours -- good prognosis
- **Persistent AKI:** reversal within 2-7 days
- **Severe ATN:** weeks to months for full creatinine normalization
- In simulation models: 90% GFR drop at hour 8, recovery to baseline by day 7

### Improving vs Worsening Trajectory

| Parameter | Improving | Worsening |
|---|---|---|
| Creatinine | Peaks day 3-5, falls 0.3-0.5 mg/dL/day | Rises >0.5 mg/dL/day, no plateau |
| Urine output | >0.5 mL/kg/hr by day 2-3 | Remains <0.3 mL/kg/hr or anuric |
| Potassium | Stabilizes <5.5 with medical Rx | Rises despite treatment, >6.5 |
| pH | Corrects with bicarb if needed | Worsening acidosis <7.2 |

### Sources
- [Creatinine Kinetics and AKI Definition - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2653692/)
- [AKI - StatPearls](https://www.ncbi.nlm.nih.gov/books/NBK441896/)
- [AKI - EMCrit IBCC](https://emcrit.org/ibcc/aki/)
- [KDIGO AKI Guideline](https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf)
- [AKI Recovery - CJASN](https://journals.lww.com/cjasn/fulltext/2020/09000/defining_early_recovery_of_acute_kidney_injury.22.aspx)

---

## Acute Liver Failure

### Key Laboratory Trajectories
- **INR**: >=1.5 by definition; prognostic thresholds differ by etiology
  - Acetaminophen ALF: INR >6.5 (with creatinine >3.4 and grade III-IV encephalopathy) = poor prognosis
  - Non-acetaminophen ALF: INR >3.5 (with age, etiology, bilirubin factors) = poor prognosis
  - INR trajectory over days 1-3 is used in dynamic prediction models but sequential values did not add to admission values for prognosis
- **Bilirubin**: Peak typically reached day 3-4 after maximum liver damage. Non-acetaminophen threshold: >17.5 mg/dL = poor prognostic sign. Unconjugated bilirubin detectable as early as 12 hours post-acetaminophen overdose
- **Ammonia**: >150 micromol/L = increased herniation risk; >200 mcg/dL (117 micromol/L) = highly associated with cerebral edema and herniation
- **Monitoring frequency**: INR, lactate, factor V, liver biochemistries q8-12h; glucose q1h

### Encephalopathy Progression
- **Grade I**: Altered mood, impaired concentration
- **Grade II**: Drowsy, inappropriate behavior
- **Grade III**: Stuporous but rousable, incoherent
- **Grade IV**: Coma, unresponsive
- Deterioration can be precipitous once encephalopathy develops, with rapid progression to intracranial hypertension and death

### Transplant Consideration (King's College Criteria)
- **Acetaminophen**: Arterial pH <7.3 regardless of encephalopathy grade; OR INR >6.5 + creatinine >3.4 mg/dL + grade III/IV encephalopathy
- **Non-acetaminophen**: INR >6.5 alone; OR any 3 of: unfavorable etiology, jaundice >7 days before encephalopathy, age <10 or >40, INR >3.5, bilirubin >17.5 mg/dL

### Classification by Speed
- **Hyperacute**: <7 days (acetaminophen, viral hep A/E, ischemic); higher cerebral edema risk but better overall prognosis
- **Acute**: 1-4 weeks
- **Subacute**: >4 weeks (worst prognosis)

### Sources
- [Acute Liver Failure - StatPearls](https://www.ncbi.nlm.nih.gov/books/NBK482374/)
- [Acute Liver Failure - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7539708/)
- [King's College Criteria](https://en.wikipedia.org/wiki/King's_College_Criteria)
- [ALF - EMCrit](https://emcrit.org/ibcc/alf/)

---

## Acute MI / STEMI

### Troponin Kinetics
- **Initial rise**: 3-6 hours after symptom onset (high-sensitivity assays detect earlier)
- **Peak**:
  - hs-cTnT: ~11.8 hours post-PCI; hs-cTnI: ~10.2 hours post-PCI
  - Standard assays: 12-24 hours post symptom onset
  - Median peak hs-cTnT in STEMI: 3,110 ng/L (mean 5,124; range 20-57,875; reference <14 ng/L)
- **Second peak (hs-cTnT only)**: ~57.8% of STEMI patients show a biphasic curve with second peak at ~77 hours (day 3-5), concentration ~28.6% lower than first peak
- **Normalization**: Troponin I elevated 5-7 days; Troponin T elevated up to 14-21 days
- **Half-life**: hs-cTnI ~12 hours; hs-cTnT ~17 hours
- **Prognostic value**: Higher peak troponin inversely correlated with LVEF; higher peaks predict 30-day and 1-year mortality

### Post-PCI Hemodynamic Trajectory
- **Uncomplicated STEMI**: Median ICU stay = 1 day. Only 16% develop post-PCI complications requiring ICU. Risk of ICU-requiring complications very low in rapidly treated patients within 24-48 hours
- **Complicated (cardiogenic shock)**: ~50% 30-day mortality despite revascularization. Longer door-to-balloon time increases complication risk

### BNP/NT-proBNP
- **Rises markedly within 24 hours** post-STEMI
- **Correlates with infarct size** and LV systolic dysfunction
- Heart failure develops in ~22.6% of invasively treated patients vs ~47.3% of conservatively treated
- Higher BNP = worse systolic function

### Sources
- [Troponin Kinetics in STEMI](https://pubmed.ncbi.nlm.nih.gov/25943557/)
- [Peak Troponin T in STEMI](https://pmc.ncbi.nlm.nih.gov/articles/PMC9174820/)
- [Troponin - StatPearls](https://www.ncbi.nlm.nih.gov/books/NBK507805/)
- [NT-proBNP After MI](https://www.ahajournals.org/doi/10.1161/hy0102.100537)
- [ICU Utilization in STEMI After PCI](https://www.jacc.org/doi/10.1016/j.jcin.2019.01.230)

---

## Acute Pancreatitis

### Lipase Trajectory
- **Rise**: Within 4-8 hours of symptom onset
- **Peak**: ~24 hours
- **Normalization**: 8-14 days (varies by etiology and severity)
- **Diagnostic threshold**: >=3x upper limit of normal (sensitivity 80%, specificity 90%)

### CRP Trajectory
- **Peak**: 36-72 hours after disease onset (NOT useful at admission)
- **Severity threshold**: CRP >150 mg/L (>180 mg/dL per some sources) at 48-72 hours predicts necrosis with sensitivity and specificity both >80%
- **CRP in double figures (>=10 mg/dL)** strongly indicates severe pancreatitis

### Complications Timeline
- **Sterile necrosis**: Develops within first 4 days, progresses over 2 weeks
- **Infected necrosis**: Typically after 2 weeks (anti-inflammatory phase promotes bacterial translocation from gut)
- **Acute necrotic collection (ANC)**: First 4 weeks
- **Walled-off necrosis (WOPN)**: Develops after 4 weeks (1-9% of cases); typically 4-6 weeks post-onset, heralded by pain, fever, chills
- **CT imaging for complications**: Most useful at 48-72 hours, NOT at admission

### Severity Scoring
- **Persistent SIRS >48 hours**: Significantly increased mortality, indicates ICU admission
- **Ranson score**: Fully assessable only at 48 hours
  - <3: mild; hospital stay ~3.65 days
  - 3-5: moderate; ~10% mortality; hospital stay ~8.35 days
  - >5: severe; >50% mortality; hospital stay ~10.55 days; ICU stay 5-21 days
- **APACHE-II >=8**: Predicts severe pancreatitis

### Sources
- [Acute Pancreatitis - StatPearls](https://www.ncbi.nlm.nih.gov/books/NBK482468/)
- [Necrotizing Pancreatitis Management](https://pmc.ncbi.nlm.nih.gov/articles/PMC5565044/)
- [Ranson Criteria - StatPearls](https://www.ncbi.nlm.nih.gov/books/NBK482345/)
- [Acute Pancreatitis - AMBOSS](https://www.amboss.com/us/knowledge/acute-pancreatitis/)

---

## Alcohol Withdrawal / Delirium Tremens

### Timeline (Hours After Last Drink)

| Hours | Phase | Manifestations |
|---|---|---|
| 6-12h | Minor withdrawal | Tremor, anxiety, headache, diaphoresis, palpitations, GI upset |
| 12-24h | Alcoholic hallucinosis | Visual/auditory/tactile hallucinations (patient aware they are unreal) |
| 12-48h | Withdrawal seizures | Generalized tonic-clonic (3-5% of cases), peak at 24-36h |
| 48-72h | Delirium tremens onset | Peak incidence; confusion, agitation, autonomic storm |
| 72-120h | DT peak | Most intense; can last up to 5 days total |

### Vital Signs Trajectory

| Parameter | Minor (6-12h) | Moderate (12-48h) | Severe/DT (48-96h) |
|---|---|---|---|
| Heart rate | 90-110 bpm | 100-120 bpm | 120-150+ bpm |
| SBP | 140-160 mmHg | 150-180 mmHg | 160-200+ mmHg |
| Temperature | 37.0-37.5C | 37.5-38.5C | 38.5-40.5C |
| Diaphoresis | Mild | Moderate | Profuse |

### CIWA-Ar Score Trajectory
- **0-8:** Mild withdrawal -- observation, may not need medication
- **9-15:** Moderate -- symptom-triggered benzodiazepines
- **>15:** Severe -- high risk of DT and seizures, aggressive treatment
- **>20:** Very severe -- ICU-level care
- Improving: CIWA drops from peak of 15-25 to <10 within 24-48h of treatment
- Worsening: CIWA escalates from 12-15 to >20-25 despite treatment, progressing to DT

### Benzodiazepine Dosing
- **Symptom-triggered:** medicate when CIWA >8
- **Diazepam:** 10-20 mg IV q5-15min for acute severe withdrawal; typical 24h total 40-200 mg
- **Lorazepam:** 2-4 mg IV q15-30min (preferred in liver failure)
- **Seizure treatment:** diazepam 20-60 mg equivalent rapidly
- **Refractory DT:** phenobarbital 130-260 mg IV; may need propofol/dexmedetomidine infusion in ICU

### Electrolyte Abnormalities

| Electrolyte | Typical Deficit | Replacement | Correction Timeline |
|---|---|---|---|
| Magnesium | Severe total body depletion | MgSO4 4g IV q12h | 2-3 days to equilibrate |
| Potassium | Often 3.0-3.5 mEq/L | IV KCl; correct Mg first | 24-48h |
| Phosphorus | Low (often <2.0 mg/dL) | IV sodium phosphate | 24-48h |
| Glucose | Hypoglycemia common | Dextrose + thiamine first | Hours |
| Fluid deficit | Up to 10L | Aggressive IV NS/LR | 24-72h |

### Sources
- [Alcohol Withdrawal - StatPearls](https://www.ncbi.nlm.nih.gov/books/NBK441882/)
- [Delirium Tremens - StatPearls](https://www.ncbi.nlm.nih.gov/books/NBK482134/)
- [DT Assessment & Management - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6286444/)
- [Alcohol Withdrawal - EMCrit IBCC](https://emcrit.org/ibcc/etoh/)
- [Alcohol Withdrawal - AAFP](https://www.aafp.org/pubs/afp/issues/2004/0315/p1443.html)

---

## Anaphylaxis

### Hemodynamic Trajectory
- **BP:** SBP drops to <90 mmHg or >30% decrease from baseline within minutes of exposure
- **HR:** Compensatory tachycardia 110-150+ bpm
- **SpO2:** Desaturation from bronchospasm; can drop rapidly to 80s-low 90s with laryngeal edema/bronchospasm
- **Onset:** Symptoms typically begin within 5-30 minutes of exposure (faster with IV/parenteral allergens)

### Epinephrine Response Timeline
- **Dose:** 0.3-0.5 mg IM (1:1000) into anterolateral thigh
- **Peak plasma level:** ~8 minutes after IM injection (vs 34 minutes for subcutaneous)
- **Clinical response:** Within minutes -- vasoconstriction, bronchodilation, mast cell stabilization
- **Repeat dosing:** Every 5-10 minutes if inadequate response; most patients respond to 1-3 doses
- **IV epinephrine:** For refractory shock: 1-5 mcg/min infusion (push-dose: 10-20 mcg boluses)

### Hemodynamic Effect of Epinephrine
- HR increases ~12 bpm on average
- Cardiac output increases ~23%
- FEV1 improves ~14%, peak flow improves ~9%
- NOTE: MAP may NOT increase significantly despite cardiac output improvement

### Biphasic Reactions
- **Incidence:** 5-20% of anaphylaxis cases (higher with more severe initial reactions -- 30% if initial reaction was "very severe")
- **Timing:** Median onset at 11 hours (range 0.2-72 hours); most occur within 8-12 hours
- **Recommended observation:** 4-8 hours minimum; longer (12-24h) if severe initial presentation
- **Risk factors:** Severe initial reaction, unknown trigger, delayed epinephrine

### Tryptase
- Peaks 1-2 hours after onset; elevated for 6-24 hours
- Draw at 15 min to 3 hours, then baseline at 24+ hours

### Sources
- [StatPearls - Epinephrine](https://www.ncbi.nlm.nih.gov/books/NBK482160/)
- [PMC - Biphasic Anaphylaxis Incidence and Timing](https://pmc.ncbi.nlm.nih.gov/articles/PMC8323456/)
- [PMC - Epinephrine in Anaphylaxis](https://pmc.ncbi.nlm.nih.gov/articles/PMC10185359/)
- [AAFP - Anaphylaxis Recognition and Management](https://www.aafp.org/pubs/afp/issues/2020/0915/p355.html)

---

## ARDS / Acute Respiratory Failure

### Berlin Definition Severity

| Severity | P/F Ratio | Mortality |
|----------|-----------|-----------|
| Mild | 201-300 mmHg | 27% (meta-analysis); 10% (single study) |
| Moderate | 101-200 mmHg | 35%; 23% |
| Severe | <=100 mmHg | 45%; 55% |

### Improving Trajectory
- **P/F ratio**: Gradual improvement over days. Higher PEEP table recommended if P/F <200
- **Proning**: Initiated for P/F <150 on FiO2 >=0.6 and PEEP >=10 cmH2O. Duration: >=16 hours per session, then 4-8 hours supine reassessment. Average treatment: 4 days in PROSEVA trial (range: 4-8 days across studies). Stop criterion: PaO2/FiO2 >150 with PEEP <10 and FiO2 <0.6 for at least 4 hours in supine position
- **FiO2/PEEP weaning**: Daily titration as oxygenation improves. Use ARDSNet FiO2/PEEP tables (low or high PEEP strategy)
- **Ventilator days (survivors)**: Non-COVID ARDS median 4 days (IQR 2-8); COVID ARDS median 10 days (IQR 6-20). With dexamethasone: ventilator-free days 13 vs 7 at day 28 (p<0.001)
- **Extubation**: When FiO2 <=0.4, PEEP <=5-8, adequate cough, passing spontaneous breathing trial

### Worsening Trajectory
- **P/F ratio**: Declining or failing to improve despite escalating support
- **Progression**: Mild to moderate to severe can occur over hours to days
- **Mortality risk increases with**: refractory hypoxemia, need for rescue therapies (proning, inhaled NO, ECMO)

### Sources
- [ARDS - EMCrit](https://emcrit.org/ibcc/ards/)
- [ARDSNet Ventilator Protocol](http://www.ardsnet.org/files/ventilator_protocol_2008-07.pdf)
- [Berlin Definition of ARDS](https://www.mdcalc.com/calc/10294/berlin-criteria-acute-respiratory-distress-syndrome)
- [Management of ARDS - What Works](https://pmc.ncbi.nlm.nih.gov/articles/PMC7997862/)
- [Time to Extubation Among ARDS Subjects](https://pmc.ncbi.nlm.nih.gov/articles/PMC10506654/)

---

## Bacterial Meningitis

### CSF Parameters at Diagnosis

| Parameter | Bacterial | Normal |
|-----------|-----------|--------|
| WBC | >1000/uL (often >2000; neutrophil predominance >80%) | <5/uL |
| Protein | >220 mg/dL (often 100-500+) | 15-45 mg/dL |
| Glucose | <34 mg/dL | 40-70 mg/dL |
| CSF:serum glucose ratio | <0.4 (often <0.2) | 0.5-0.8 |
| Opening pressure | Elevated (>20 cm H2O, often >30) | 6-20 cm H2O |

- Any one of: glucose <34, protein >220, WBC >2000, or neutrophils >1180 = >99% certainty of bacterial meningitis

### CSF Changes with Treatment
- **48-72 hours on antibiotics:** CSF culture sterilizes in nearly all cases; glucose normalizes in ~71% of cases; protein decline is slow (only 11% normal at 48h); WBC count may still be elevated (PMN count and protein are NOT significantly decreased at 48h)
- **Key:** Microbiological sterilization is fast, but biochemical normalization lags

### Fever Trajectory
- Classic triad on presentation: fever (74%), neck stiffness (71%), GCS <14 (85%); full triad in 47%
- **Persistent fever:** Defined as lasting >24 hours after antibiotics
- **Expected:** Defervescence within 48-72 hours on appropriate antibiotics
- **Secondary fever:** Days 4-8 may indicate drug fever, nosocomial infection, or subdural collection

### GCS Trajectory
- **Admission median GCS:** 10 (IQR 8-12); 32% present in coma
- **Improving:** GCS should improve within 24-48 hours if antibiotics effective
- **Worsening:** Continued decline suggests cerebral edema, hydrocephalus, cerebral venous thrombosis, or wrong antibiotic coverage

### Vasopressor Need in Meningococcal Sepsis
- Meningococcemia can cause fulminant septic shock within hours of onset
- **Norepinephrine:** First-line, starting 0.05-0.15 mcg/kg/min, titrated to MAP >65 mmHg
- **DIC:** Common in meningococcal disease; monitor PT/INR, fibrinogen, platelets
- **Typical ICU course:** 3-7 days for uncomplicated; weeks for complicated (with cerebral edema, hydrocephalus, DIC)
- Dexamethasone 0.15 mg/kg q6h x 4 days: give with or before first antibiotic dose (reduces mortality for pneumococcal meningitis)

### Sources
- [AAFP - CSF Analysis](https://www.aafp.org/pubs/afp/issues/2021/0401/p422.html)
- [NEJM - Clinical Features and Prognosis in Adult Bacterial Meningitis](https://www.nejm.org/doi/abs/10.1056/NEJMoa040845)
- [StatPearls - Bacterial Meningitis](https://www.ncbi.nlm.nih.gov/books/NBK470351/)

---

## Cardiogenic Shock

### SCAI Staging with Hemodynamic Values

| Stage | Description | Lactate | SBP | Mortality |
|-------|------------|---------|-----|-----------|
| A | At risk | Normal | Normal | 0.6% |
| B | Beginning shock | Normal | SBP <90 or HR >100 | 2.7% |
| C | Classic shock | 2-5 mmol/L | <90 on pressors | 21.5% |
| D | Deteriorating | 5-10 mmol/L | Escalating support | 54.3% |
| E | Extremis | >10 mmol/L | <60, refractory | 90.6% |

### Hemodynamic Criteria
- **Cardiac index:** <2.2 L/min/m2
- **PCWP:** >15 mmHg (elevated filling pressures)
- **Cardiac power output:** <0.6 watts (strongest hemodynamic predictor of mortality)
- **SvO2:** <60% (indicates inadequate oxygen delivery)
- **CVP:** >15 mmHg
- **CVP/PCWP ratio:** >0.63 suggests RV failure

### Lactate Trajectory
- **Improving:** Lactate clearance >10% every 2 hours; target <2 mmol/L
- **Worsening:** Rising lactate despite support = failing resuscitation; prompts escalation

### Vasopressor/Inotrope Escalation

| Line | Agent | Dose Range | Onset |
|------|-------|-----------|-------|
| First | Norepinephrine | 0.05-0.4 mcg/kg/min | Seconds |
| Second | Dobutamine | 2-20 mcg/kg/min | 1-2 min |
| Alternative | Milrinone | 0.125-0.5 mcg/kg/min | 5-15 min |
| Rescue | Epinephrine | 0.01-0.2 mcg/kg/min | Seconds |

- Shock resolution rates: milrinone ~76%, dobutamine ~70% (no significant difference in RCT)
- Milrinone: fewer arrhythmias (32.8% vs 62.9% dobutamine)
- Escalation trigger: 2 moderate-dose or 1 high-dose inotrope failing = consider mechanical support

### Mechanical Circulatory Support Criteria
- PA catheter showing: CI <2.2, CPO <0.6W, SvO2 <60%, PAPI <1.0
- **IABP:** Limited support (~0.5 L/min augmentation); declining use
- **Impella CP:** 3-4 L/min support; preferred for isolated LV failure
- **VA-ECMO:** 4-6 L/min; for biventricular failure or cardiac arrest
- 7.5% of IABP patients require escalation to Impella or ECMO
- Early deployment with shock teams and standardized algorithms improves outcomes

### Sources
- [JACC - SCAI SHOCK Stage Classification](https://www.jacc.org/doi/10.1016/j.jacc.2022.01.018)
- [JACC - Criteria for Defining Stages of Cardiogenic Shock](https://www.jacc.org/doi/10.1016/j.jacc.2022.04.049)
- [PMC - Management of Cardiogenic Shock](https://pmc.ncbi.nlm.nih.gov/articles/PMC10980676/)
- [AHA - Contemporary Management of Cardiogenic Shock](https://www.ahajournals.org/doi/10.1161/cir.0000000000000525)
- [NEJM - Milrinone vs Dobutamine in Cardiogenic Shock](https://www.nejm.org/doi/full/10.1056/NEJMoa2026845)

---

## COPD Exacerbation

### PaCO2 and pH Trajectory
- **Baseline chronic COPD:** PaCO2 often 45-55 mmHg, pH 7.35-7.38 (compensated)
- **Acute exacerbation:** PaCO2 rises to 60-90+ mmHg, pH drops to 7.15-7.30
- **On BiPAP (improving):** pH should improve within 1-2 hours; PaCO2 drops but should NOT be reduced >20 mmHg or >50% of pre-treatment PaCO2 in first 24h (risk of CNS complications)
- **On BiPAP (worsening):** pH fails to improve or worsens after 2 hours = BiPAP failure

### SpO2 Targets
- **Target 88-92%** (85-95% acceptable range)
- Avoid hyperoxia (>96%) -- suppresses hypoxic drive in CO2 retainers

### BiPAP Settings Progression

| Setting | Starting | Titration | Typical Target |
|---|---|---|---|
| IPAP | 10 cmH2O | Increase 2-5 cmH2O q10min | 15-20 cmH2O |
| EPAP | 4 cmH2O | Increase if needed for oxygenation | 4-8 cmH2O |
| FiO2 | Titrate to SpO2 88-92% | Wean as tolerated | Lowest effective |

### Intubation Criteria / Timeline
- **Repeat ABG at 30 minutes and 2 hours** after starting NIV
- **2-hour decision point:** if no improvement in pH/PaCO2, consider intubation
- **4-hour hard decision:** proceed to intubation if NIV failing
- **Absolute intubation indications:** respiratory arrest, GCS <8, inability to protect airway, refractory hypoxemia (SpO2 <85% despite BiPAP), hemodynamic instability
- If >12h of BiPAP without improvement, strongly consider intubation

### ICU Length of Stay
- **BiPAP responders:** median ICU stay 3-5 days
- **Intubated patients:** median ICU stay 7-14 days
- **Overall hospital stay:** 5-9 days for severe exacerbation

### Sources
- [AECOPD - EMCrit IBCC](https://emcrit.org/ibcc/aecopd/)
- [NIV for COPD - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6483555/)
- [BiPAP Settings - Maimonides EM](https://www.maimonidesem.org/blog/bipap-settings)

---

## DKA (Diabetic Ketoacidosis)

### Presentation Values (by severity)

| Parameter | Mild | Moderate | Severe |
|-----------|------|----------|--------|
| Glucose | >250 mg/dL | >250 mg/dL | >250 mg/dL (often 400-800) |
| pH | 7.25-7.30 | 7.00-7.24 | <7.00 |
| Bicarbonate | 15-18 mEq/L | 10-<15 mEq/L | <10 mEq/L |
| Anion gap | >10 | >12 | >12 (often 20-30+) |
| Potassium | Often 5.0-6.0 mEq/L (falsely elevated despite total body depletion) |

### Improving Trajectory (Typical Resolution)
- **Glucose correction**: Target decrease of 50-75 mg/dL per hour on insulin drip (0.1 U/kg/hr). Do NOT let glucose fall below 200 mg/dL in first 4-5 hours. When glucose hits 200, add dextrose to fluids and reduce insulin rate to 2-3 U/hr
- **Anion gap closure**: Median normalization time = 8 hours (IQR 6-12 hours). Steep correction in first 6-8 hours, then slopes flatten by hours 14-16. This is the PRIMARY endpoint, not glucose
- **pH correction**: Lags behind anion gap; continues rising slowly. pH slopes flatten around hour 14. Resolution criteria: pH >7.30
- **Bicarbonate**: Rises to >=15 mEq/L; slopes flatten around hour 16
- **Potassium trajectory**: Drops rapidly during treatment (insulin drives K+ intracellular). ~2/3 of patients develop hypokalemia during treatment. Start K+ replacement when serum K+ falls below 5.0 mEq/L; target 4.0-5.0 mEq/L. If K+ <3.5 at presentation, replace at 10 mmol/hr and HOLD insulin until K+ >3.5
- **Resolution criteria** (all must be met): glucose <200 mg/dL AND 2 of 3: anion gap <12, bicarb >=15, pH >7.30
- **Typical total resolution time**: 8-16 hours for most cases; severe DKA may take 24 hours

### Vital Signs During Treatment
- **HR**: Tachycardia (often 110-130) improves with fluid resuscitation over first 2-4 hours
- **BP**: Hypotension corrects with 1-2L NS bolus in first 1-2 hours
- **RR**: Kussmaul breathing (deep, rapid, RR 24-30+) resolves as pH normalizes

### Sources
- [Management of adult DKA](https://pmc.ncbi.nlm.nih.gov/articles/PMC4085289/)
- [DKA Treatment - Medscape](https://emedicine.medscape.com/article/118361-treatment)
- [Anion gap normalization time in DKA](https://pmc.ncbi.nlm.nih.gov/articles/PMC10275565/)
- [DKA - EMCrit](https://emcrit.org/ibcc/dka/)
- [Adult DKA - StatPearls](https://www.ncbi.nlm.nih.gov/books/NBK560723/)

---

## Drug Overdose (Opioid)

### Naloxone Dosing and Response
- **Initial dose:** 0.04-0.4 mg IV (start low in known opioid-dependent patients to avoid withdrawal)
- **Standard rescue dose:** 0.4-2 mg IV/IM/IN
- **Repeat:** q2-3 minutes if no response
- **Maximum:** up to 10 mg total; if no response after 10 mg, reconsider diagnosis
- **Onset of action:** IV: 1-2 minutes; IM: 2-5 minutes; IN: 3-8 minutes
- **Duration:** ~30-90 minutes (naloxone); opioid may outlast naloxone -- observe for renarcotization

### Respiratory Rate Trajectory

| Timepoint | Without Naloxone | With Naloxone |
|---|---|---|
| Presentation | 4-8 breaths/min or apneic | -- |
| 2 min post-naloxone IV | -- | Improves to 12-16 breaths/min |
| 5 min post-naloxone | -- | 16-22 breaths/min |
| Target | -- | RR >12, SpO2 >90%, EtCO2 <45 |

### GCS Trajectory
- **Presentation:** GCS 3-6 (unresponsive)
- **Post-naloxone (1-2 min IV):** recovery of consciousness, GCS improves to 13-15
- **Renarcotization risk:** GCS may drop again 30-90 min after naloxone if long-acting opioid (methadone, fentanyl patches)
- **BIS <78** predicts need for intubation (88.5% specificity, 86.7% sensitivity)

### SpO2 Recovery
- **Presentation:** may be 60-85% or unobtainable
- **Key insight:** PaCO2 rises BEFORE SpO2 drops (EtCO2 >45 is an earlier warning sign)
- **Post-naloxone:** SpO2 recovers to >95% within 2-5 minutes if adequate ventilation restored
- **Persistent low SpO2 despite naloxone:** suspect aspiration pneumonia, pulmonary edema, or non-opioid co-ingestion

### Intubation vs Naloxone Alone
- **Naloxone sufficient:** responds to <=2 mg, RR >12, GCS >=13, protects airway
- **Intubation needed:** no response to >10-12 mg total naloxone, persistent apnea, aspiration, co-ingestion (benzos, barbiturates), pulmonary edema
- All patients need minimum 4-6 hour observation; methadone/extended-release opioids require 12-24 hours

### Rhabdomyolysis / CK Trajectory
- **Cause:** prolonged immobility (found down) -- compressed muscle groups
- **CK onset:** starts rising 2-12 hours after muscle injury
- **CK peak:** day 1-3 (24-72 hours post-injury)
- **Diagnostic threshold:** CK >5,000 IU/L (5x upper normal)
- **Severe (risk of AKI):** CK >15,000-25,000 IU/L
- **Incidence in opioid overdose:** ~31% of rhabdomyolysis cases in poisoning are opioid-related
- **CK clearance half-life:** ~1.5 days; should decrease ~50% every 36-48 hours if cause removed
- **AKI from rhabdomyolysis:** monitor creatinine, maintain UOP >200 mL/hr with IV fluids

### Sources
- [Opioid Toxicity - StatPearls](https://www.ncbi.nlm.nih.gov/books/NBK470415/)
- [Naloxone Dosing - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5753997/)
- [Rhabdomyolysis - StatPearls](https://www.ncbi.nlm.nih.gov/books/NBK448168/)
- [Rhabdomyolysis in Opioid Overdose - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5790980/)

---

## GI Bleed

### Hemorrhagic Shock Classification (ATLS, 70kg patient, ~5L blood volume)

| Class | Blood Loss | Blood Loss (mL) | HR | SBP | RR | Mental Status |
|-------|-----------|-----------------|-----|-----|-----|--------------|
| I | <15% | <750 | Normal-mild increase | Normal | Normal | Normal |
| II | 15-30% | 750-1500 | 100-120 | Normal to slightly decreased | 20-24 | Anxious |
| III | 30-40% | 1500-2000 | 120-140 | Decreased | 24-30 | Confused |
| IV | >40% | >2000 | >140 (thready) | Severely decreased | >35 | Lethargic |

### Key Trajectory Numbers
- **Hgb lag**: Hemoglobin and hematocrit remain UNCHANGED immediately after acute blood loss; decline only becomes apparent 1-4 hours later. Normal Hgb + shock = very brisk active bleed
- **Active bleed indicator**: Hgb dropping 3 g/dL over 2-4 hours
- **Transfusion response**: Each unit of PRBC raises Hgb by ~1 g/dL (or ~3% Hct) in a non-actively-bleeding patient
- **Transfusion threshold**: Restrictive (<7 g/dL) superior to liberal (<9 g/dL) -- reduced mortality and rebleeding (NEJM trial, n=1965)
- **Endoscopy timing**: Within 24 hours for upper GI bleed (ACG/ESGE); within 12 hours for suspected variceal hemorrhage (AASLD)

### Rebleeding
- Most rebleeding occurs within 72 hours (standard monitoring period)
- Low-risk (Rockall <=2): 4.3% rebleed rate
- High-risk (Rockall >=6): 15% rebleed rate
- Hgb monitoring: q8h for first 2 days, then daily

### Sources
- [Hemorrhagic Shock - StatPearls](https://www.ncbi.nlm.nih.gov/books/NBK470382/)
- [Transfusion Strategies for Upper GI Bleeding (NEJM)](https://www.nejm.org/doi/full/10.1056/NEJMoa1211801)
- [GI Bleeding - EMCrit](https://emcrit.org/ibcc/gib/)

---

## Hemorrhagic Stroke / ICH

### Blood Pressure Management
- **Target SBP:** 140-180 mmHg (2022 AHA guidelines), individualized
- **Initiate treatment within 2 hours** of ICH onset; reach target within 1 hour
- **Avoid SBP <130 mmHg** -- potentially harmful
- **Avoid SBP drops >70 mmHg within the first hour**
- **Preferred agents:** nicardipine 5-15 mg/hr IV (start 5 mg/hr, increase 2.5 mg/hr q15min); clevidipine alternative
- Maintain MAP >85 mmHg, CPP 60-80 mmHg

### GCS Trajectory
- **>20% of patients** deteriorate by >=2 GCS points between field and ED
- **Early deterioration:** within first 24-48 hours (hematoma expansion, edema)
- **Late deterioration (48h-1 week):** usually from cerebral edema, hydrocephalus
- GCS <=8: consider intubation, ICP monitoring

### ICP
- **Target ICP:** <20 mmHg
- **CPP target:** 50-70 mmHg (autoregulation dependent)
- **Treatment:** mannitol 20% (0.25-1 g/kg IV bolus) if ICP >20
- Hypertonic saline (23.4%, 30 mL) for acute herniation

### Hematoma Expansion
- **Highest risk:** first 3-6 hours after onset
- **Incidence:** up to 33% of patients within first 24 hours
- "Spot sign" on CTA predicts expansion
- Early hematoma expansion (>33% volume increase) in 3.4-22% depending on BP control achieved

### Cerebral Edema
- **Peaks at 72 hours (day 3)** post-ictus
- Perihematomal edema progression lasts 5 days to 2 weeks
- Largest increase in edema in first 72 hours

### Typical ICU Course

| Timepoint | Clinical Focus |
|---|---|
| Hour 0-6 | BP control, prevent hematoma expansion, surgical decision |
| Hour 6-24 | Monitor for expansion, repeat CT at 6-24h, ICP management |
| Day 1-3 | Cerebral edema peak approaching, neurological monitoring |
| Day 3-5 | Edema peak, highest risk for delayed deterioration |
| Day 5-14 | Edema resolution begins, rehabilitation assessment |
| 30-day mortality | Up to 50% (most within first 24h) |
| 6-month independence | Only 20% of patients |

### Sources
- [ICH BP Management - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11199276/)
- [2022 AHA ICH Guideline](https://www.ahajournals.org/doi/10.1161/STR.0000000000000407)
- [ICH - EMCrit IBCC](https://emcrit.org/ibcc/ich/)
- [ICH Hematoma Expansion - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7380105/)
- [ICH - StatPearls](https://www.ncbi.nlm.nih.gov/books/NBK553103/)
- [Late Deterioration in ICH - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4100944/)

---

## Hypertensive Emergency

### BP Reduction Targets (Stepwise)
- **Hour 1:** Reduce MAP by no more than 25% (or SBP reduction of ~25%)
- **Hours 2-6:** Gradually reduce to ~160/100-110 mmHg
- **Hours 24-48:** Cautious normalization toward patient's baseline BP
- CRITICAL: Overly aggressive reduction causes watershed ischemia (stroke, MI, renal failure)

### IV Antihypertensives -- Onset and Duration

| Drug | Onset | Duration | Starting Dose | Max Dose |
|------|-------|----------|---------------|----------|
| Clevidipine | 2-3 min | 5-15 min | 1-2 mg/hr | 32 mg/hr |
| Nicardipine | 5-15 min | 40-60 min | 5 mg/hr | 15 mg/hr |
| Labetalol | 2-5 min | 3-6 hr | 20 mg IV bolus | 300 mg total |
| Nitroprusside | 1-2 min | 1-10 min | 0.3 mcg/kg/min | 10 mcg/kg/min |
| Esmolol | 1-2 min | 10-30 min | 500 mcg/kg bolus | 300 mcg/kg/min |

- Nicardipine achieves target BP in 91.7% within 30 min vs 82.5% for labetalol
- Clevidipine preferred for tight control (shortest half-life, lowest overshoot risk)
- Nitroprusside falling out of favor due to cyanide toxicity risk

### End-Organ Damage Markers
- **Cardiac:** Troponin elevation (hs-TnT) common; guides intensity of BP reduction
- **Renal:** Creatinine rise, sometimes microangiopathic hemolytic anemia
- **Hematologic:** LDH >190 U/L, schistocytes, thrombocytopenia in malignant hypertension (thrombotic microangiopathy)
- **Neurologic:** Encephalopathy, papilledema, focal deficits

### Typical ICU Course
- IV infusion for 24-72 hours in ICU with arterial line monitoring
- Transition to oral agents once BP stable at target for 6-8 hours
- Monitor troponin q6-8h, creatinine q12-24h, urine output hourly

### Sources
- [EMCrit - Hypertensive Emergency](https://emcrit.org/ibcc/htn/)
- [StatPearls - Hypertensive Emergency](https://www.ncbi.nlm.nih.gov/books/NBK470371/)
- [PMC - Treatment of Hypertensive Emergencies](https://pmc.ncbi.nlm.nih.gov/articles/PMC5440310/)
- [PMC - Cardiac Biomarkers in Hypertensive Emergency](https://pmc.ncbi.nlm.nih.gov/articles/PMC10178101/)

---

## Major Burns

### Fluid Resuscitation (Parkland Formula)
- **Formula:** 4 mL x weight (kg) x %TBSA burned
- **Example:** 80 kg patient, 40% TBSA = 12,800 mL in 24 hours
- **Delivery:** First half (6,400 mL) over first 8 hours from time of burn; second half over next 16 hours
- **Urine output target:** 0.5-1.0 mL/kg/hr adults; 1.0-1.5 mL/kg/hr children
- **Over-resuscitation (fluid creep):** Occurs in up to 90% of patients with >10% TBSA; causes compartment syndrome, ARDS, pneumonia, multi-organ failure

### Hemodynamic Trajectory
- **0-24 hours (Ebb/Emergent Phase):** Capillary leak begins within 20 minutes. Hypovolemic shock with depressed cardiac output, tachycardia, hypotension. Maximum capillary permeability at 12 hours. Edema peaks at 24-48 hours.
- **24-36 hours:** Cardiac output remains depressed
- **48-72 hours (Flow/Hypermetabolic Phase):** Capillary permeability normalizes. Cardiac output becomes supranormal (2-3x normal), tachycardia, decreased SVR. This hypermetabolic state persists for months.

### Hematocrit Trajectory
- **0-24 hours:** Hemoconcentration (elevated Hct) due to plasma volume loss from capillary leak; Hct can rise to 55-60%+
- **24-72 hours:** As fluid resuscitation progresses and capillary leak resolves, hemodilution occurs; Hct drops, potentially to anemia range
- **Post-72 hours:** Ongoing anemia from RBC destruction, bone marrow suppression, repeated surgical blood loss
- Hematocrit correlates poorly with actual fluid volume administered (weak marker for guiding resuscitation)

### Albumin Trajectory
- **Falls rapidly** in first 24-48 hours from capillary leak (third-spacing)
- **Hypoalbuminemia** is an independent mortality predictor (OR 3.56)
- **Albumin administration:** Consider 5% albumin after 8-12 hours in >30% TBSA burns; more effective after 24 hours when capillary permeability normalizes
- Serial monitoring essential

### Temperature
- **0-16 hours:** Hypothermia risk (average temp does not exceed 36 C until at least 16 hours post-burn); from exposure, cooling, fluid resuscitation
- **48-72+ hours:** Thermoregulatory set point resets to ~38.5 C; persistent "fever" is normal in the hypermetabolic phase and does NOT necessarily indicate infection
- **Each 1 C drop in temperature** = 20% increase in transfusion requirements
- Ambient temperature must be maintained at 28-33 C in burn units

### Sources
- [StatPearls - Parkland Formula](https://www.ncbi.nlm.nih.gov/books/NBK537190/)
- [StatPearls - Burn Fluid Resuscitation](https://www.ncbi.nlm.nih.gov/books/NBK534227/)
- [PMC - Fluid Management in Major Burns](https://pmc.ncbi.nlm.nih.gov/articles/PMC3038406/)
- [PMC - Temperature Management of Adult Burn Patients](https://pmc.ncbi.nlm.nih.gov/articles/PMC10156506/)

---

## Massive Transfusion (Trauma)

### Transfusion Ratio and Protocol
- **1:1:1 ratio:** PRBC : FFP : Platelets (standard of care per PROPPR trial)
- Labs q30-60 min: Hb, platelets, PT/INR, fibrinogen, ionized calcium, lactate, ABG

### Hemoglobin Trajectory
- **Acute hemorrhage:** Hb may be NORMAL initially (hemoconcentration); takes hours to equilibrate and show true anemia
- **During resuscitation:** Target Hb >7-8 g/dL; stop transfusion when Hb >10 g/dL
- Each unit PRBC raises Hb by ~1 g/dL

### Coagulation Trajectories
- **Fibrinogen:** First factor to reach critically low levels; <1.5 g/L triggers cryoprecipitate
- **Platelets:** Target >50 x10^9/L (>100 x10^9/L for traumatic brain injury); stop when >150 x10^9/L
- **INR/PT:** Stop plasma when PT <18 sec and aPTT <35 sec
- **Cessation criteria:** Hb >10, platelets >150, PT <18, fibrinogen >1.8 g/L

### Lactate Clearance
- **Hemorrhagic shock:** Lactate 4-10+ mmol/L
- **Target:** pH >7.3, lactate <4 mmol/L
- **After hemorrhage control:** Lactate should clear at ~50% per hour if resuscitation adequate
- Persistent lactate elevation despite surgical hemostasis = ongoing shock or occult bleeding

### Temperature (Hypothermia)
- **Lethal triad** ("diamond of death" with hypocalcemia added): hypothermia + acidosis + coagulopathy (+ hypocalcemia)
- **Each 1 C drop** = 20% increase in transfusion requirements; coagulopathy worsens below 35 C; severe below 32 C
- **Target:** Maintain >36 C; active warming mandatory (warm fluids, forced air, thoracic warming)

### Calcium / Citrate Toxicity
- **Each unit PRBC** contains ~3 g citrate; normally cleared by liver in 5 minutes
- **Hypocalcemia incidence:** Up to 97% of massive transfusion patients
- **Ionized calcium:** Normal 1.15-1.33 mmol/L; <1.1 mmol/L = hypocalcemia; <0.9 mmol/L = severe
- **At >13 units PRBC:** High prevalence of severe hypocalcemia (iCa <1.0 mmol/L)
- **Replacement:** 1 g calcium gluconate after first unit of blood, then 1 g per every 4 units (Joint Trauma System 2019 guidelines)
- Hypothermia slows citrate metabolism by ~40%, prolonging hypocalcemia

### Sources
- [EMCrit - Massive Transfusion Protocol](https://emcrit.org/ibcc/mtp/)
- [StatPearls - Massive Transfusion](https://www.ncbi.nlm.nih.gov/books/NBK499929/)
- [PMC - Citrate Toxicity in Massive Transfusion](https://pmc.ncbi.nlm.nih.gov/articles/PMC10234463/)
- [European Society of Medicine - Citrate Toxicity and Hypocalcemia](https://esmed.org/citrate-toxicity-and-hypocalcemia-in-massive-transfusion/)
- [PMC - Transfusion-Related Hypocalcemia After Trauma](https://pmc.ncbi.nlm.nih.gov/articles/PMC7391918/)

---

## Post-Cardiac Surgery (CABG/Valve)

### Troponin Trajectory
- **Expected elevation:** 99.4% of patients have cTnT >=0.01 ng/mL post-op (virtually universal)
- **Typical peak:** 6-12 hours post-op
- **Normal expected post-CABG hs-cTnI:** up to 8,000 ng/L (~300x upper reference limit) -- clinically insignificant
- **Concerning:** hs-cTnI >13,000 ng/L (>500x URL) -- associated with need for repeat revascularization within 48h
- **Clinically meaningful cutoff:** hs-cTnI elevation at 12-16h post-op (early values are unreliable)
- **Normalization:** gradual decline over 5-14 days in uncomplicated cases

### Chest Tube Output
- **Normal:** decreasing trend, <100 mL/hr in first few hours, tapering
- **Concerning:** >200 mL/hr for 1 hour
- **Re-exploration threshold:** >200 mL/hr x1 hour, OR >2 mL/kg/hr for 2 consecutive hours within 6 hours of surgery
- **Removal criteria:** <200 mL in last 4 hours, no air leak, patient extubated and mobilized; earliest removal ~10 hours post-op

### Vasopressor Wean Timeline
- **Uncomplicated (fast-track):** wean off vasopressors/inotropes within 6-12 hours
- **Higher-risk / low EF:** slower wean over 24-72 hours; rapid weans risk decompensation
- Norepinephrine and vasopressin are commonly weaned first; milrinone/dobutamine last if cardiac function is impaired

### Hemoglobin Trajectory
- **Nadir:** day 3 post-op at ~9.0 g/dL (~33% decrease from preoperative level)
- **Average drift:** drop of 1.1-1.8 g/dL from initial post-op value
- **Recovery:** 79% of patients reach nadir then recover ~0.7 g/dL by discharge
- **Transfusion trigger:** historically Hb <10; newer evidence supports restrictive strategy (Hb <7-8 in stable patients)

### Extubation Timeline
- **Fast-track:** target <6 hours post-op; aggressive protocols achieve 3 hours
- **Standard:** 6-12 hours post-op
- **Delayed (>12 hours):** associated with worse outcomes; causes include hemodynamic instability, bleeding, neurological impairment
- Weaning starts after labs return and chest tube output stabilizes (~2 hours post-op)

### Pacing Wires
- **Removal:** typically day 3-5 post-op (can be as early as 24h if no pacing needed)
- **Keep longer** if: aortic valve replacement (higher AV block risk), new conduction abnormalities
- **Complication risk:** cardiac tamponade after removal in 9-18 per 10,000 surgeries

### Typical Post-Op Timeline

| Timepoint | Milestones |
|---|---|
| Hour 0-2 | Rewarming, hemodynamic stabilization, labs return |
| Hour 2-6 | Wean sedation, assess extubation readiness |
| Hour 4-8 | Extubation (fast-track target) |
| Hour 6-12 | Vasopressor wean begins, chest tube monitoring |
| Day 1 | Mobilization, diet, Foley removal, transition to oral meds |
| Day 2-3 | Chest tube removal, Hb nadir approaches |
| Day 3-5 | Pacing wire removal, transfer from ICU |
| Day 5-7 | Hospital discharge (uncomplicated) |

### Sources
- [Troponin After Cardiac Surgery - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3484554/)
- [Post-Cardiac Surgery - EMCrit IBCC](https://emcrit.org/ibcc/cts/)
- [hs-cTnI After CABG - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9246661/)
- [Hemoglobin Drift After Cardiac Surgery - Annals of Thoracic Surgery](https://www.annalsthoracicsurgery.org/article/S0003-4975(12)00655-8/pdf)
- [Fast-Track Extubation - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9801240/)
- [Epicardial Pacing Wire Removal - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10683326/)
- [Post-op Bleeding - AATS](https://www.aats.org/tsra-primer-post-operative-bleeding)

---

## Pulmonary Embolism (PE)

### Classification & Hemodynamics
- **Massive PE:** SBP <90 mmHg for >=15 minutes, or requiring vasopressors, or pulselessness, or bradycardia <40 bpm with shock
- **Submassive PE:** SBP >=90 but with RV dysfunction or myocardial necrosis markers
- **Low-risk PE:** No hemodynamic compromise, no RV dysfunction

### Troponin Trajectory (RV Strain)
- Elevated in 30-60% of moderate-large PE (depending on assay sensitivity)
- **Cutoffs:** troponin I >0.1 ng/mL or troponin T >0.03 ng/mL = increased risk
- Troponin I >0.1 ng/mL = 5-fold increased risk of adverse outcome
- **Peak:** typically within 6-12 hours of PE onset
- **Normalization:** 24-48 hours in submassive; may persist longer if RV dysfunction ongoing

### D-Dimer
- **Diagnostic cutoff:** <500 ng/mL rules out PE (moderate pretest probability); <1000 ng/mL rules out PE (low pretest probability)
- **Age-adjusted cutoff:** age x 10 for patients >50 years
- **Typical levels in PE:** often 2,000-20,000+ ng/mL depending on clot burden
- **Half-life:** ~8 hours
- **Normalization:** remains elevated ~7 days after the thrombotic event, then declines

### BNP
- **BNP >90 pg/mL** or **NT-proBNP >500 pg/mL** = RV dysfunction marker
- Elevated in ~40-50% of submassive PE
- Peaks within 12-24 hours, normalizes over days with treatment

### SpO2 Trajectory
- SpO2 is "extremely insensitive" -- normal in the majority of PE patients
- Massive PE: SpO2 may drop to 80-88% acutely
- Submassive: SpO2 often 90-95%
- PaCO2 and A-a gradient are better discriminators

### Thrombolysis Criteria
- **Indicated:** massive PE with hemodynamic instability
- **Considered:** submassive PE with RV dysfunction + biomarker elevation and clinical deterioration
- **Standard dose:** alteplase 100 mg IV over 2 hours
- **Reduced dose:** 50 mg (10 mg bolus + 40 mg over 2 hours) -- increasing evidence for intermediate-risk

### Improving vs Worsening

| Parameter | Improving | Worsening |
|---|---|---|
| HR | Tachycardia resolves <100 within 24-48h | Persistent HR >110-120, new shock |
| SBP | Maintains >90 | Drops <90, requires vasopressors |
| Troponin | Peaks then normalizes 24-48h | Rising or sustained elevation |
| SpO2 | Improves to >94% on room air | Persistent desaturation, rising O2 needs |
| RV function | Echo improves by 48-72h | Worsening RV dilation |

### Sources
- [PE D-Dimer Adjusted - NEJM](https://www.nejm.org/doi/full/10.1056/NEJMoa1909159)
- [Troponin in PE - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3506790/)
- [Submassive PE - AHA Circulation](https://www.ahajournals.org/doi/10.1161/circulationaha.110.961136)
- [Intermediate & High-Risk PE - EMCrit](https://emcrit.org/ibcc/pe/)
- [D-Dimer Test - StatPearls](https://www.ncbi.nlm.nih.gov/books/NBK431064/)
- [D-Dimer - JACC](https://www.jacc.org/doi/10.1016/j.jacc.2017.09.024)

---

## Sepsis / Septic Shock

### Presentation Values
- **HR**: >90 bpm (SIRS criterion); typically 110-130 in septic shock
- **MAP**: <65 mmHg (septic shock definition requires vasopressors to maintain MAP >=65)
- **Temperature**: >38.0C or <36.0C (SIRS); typically 38.5-40C in early sepsis
- **RR**: >20 (SIRS); typically 22-30 in moderate sepsis
- **SpO2**: variable, drops with ARDS development
- **Lactate**: >2 mmol/L defines septic shock; >4 mmol/L = severe hypoperfusion
- **WBC**: >12,000 or <4,000 cells/mm3

### Improving Trajectory
- **Lactate clearance**: Target >=10% clearance per 2 hours; >=20% relative clearance improves survival. Normalization within first 6 hours is the strongest independent predictor of survival (OR = 5.2, 95% CI 1.7-15.8). Measure q2h until >=10% clearance, then q4-6h until normal
- **Vasopressor wean**: ~70% of survivors weaned off vasopressors by day 3-4. Typical pattern: consecutive daily dose reductions. Norepinephrine starting range 0.25-0.5 mcg/kg/min, weaned over 1-3 days in survivors
- **HR normalization**: Norepinephrine has little direct effect on HR. HR normalizes as infection clears, typically over 24-72 hours
- **MAP**: Target >=65 mmHg maintained, then vasopressors weaned as volume status improves
- **Temperature**: Fever typically resolves within 48-72 hours with appropriate antibiotics
- **SOFA score**: Stable or declining over 72 hours in survivors

### Worsening Trajectory
- **Lactate**: Persistent >4 mmol/L or rising despite resuscitation; mortality >40% with lactate >2 mmol/L that fails to clear
- **Vasopressors**: Static or escalating doses over days 1-3 (key discriminator between survivors and non-survivors). High-dose norepinephrine up to 1.5-3.3 mcg/kg/min indicates refractory shock
- **SOFA**: Increase of >=2 points over 72 hours doubles mortality (42% vs 21%)
- **Organ failure progression**: Kidneys, liver, lungs, heart, CNS, and hematologic system are the most frequently affected, with multi-organ failure being the hallmark progression
- **HR fluctuation**: Difference of >=35 bpm between max and min HR in first 24h associated with increased mortality (J-shaped relationship)

### Sources
- [Surviving Sepsis Campaign Guidelines 2021](https://www.sccm.org/clinical-resources/guidelines/guidelines/surviving-sepsis-guidelines-2021)
- [Lactate Clearance for Assessing Response to Resuscitation in Severe Sepsis](https://pmc.ncbi.nlm.nih.gov/articles/PMC3982588/)
- [Whole Blood Lactate Kinetics in Severe Sepsis](https://pmc.ncbi.nlm.nih.gov/articles/PMC3673659/)
- [Lactate is THE target for early resuscitation in sepsis](https://pmc.ncbi.nlm.nih.gov/articles/PMC5496745/)
- [SOFA Score and Lactate in Sepsis Mortality](https://www.nature.com/articles/s41598-023-33227-7)
- [Organ Dysfunction in Sepsis: An Ominous Trajectory](https://pmc.ncbi.nlm.nih.gov/articles/PMC6913810/)
- [Vasopressor weaning in sepsis](https://pubmed.ncbi.nlm.nih.gov/25169693/)
- [Time-sensitive analysis of vasopressor dose in septic shock](https://associationofanaesthetists-publications.onlinelibrary.wiley.com/doi/10.1111/anae.15453)

---

## Status Epilepticus

### Treatment Escalation Timeline (Clock-Anchored)

| Time | Stage | Treatment | Response Rate |
|------|-------|-----------|---------------|
| 0-5 min | Early/developing | Midazolam 10 mg IM or Lorazepam 4 mg IV | ~72% terminate |
| 5-10 min | Repeat if needed | Second dose benzodiazepine | -- |
| 10-20 min | Established SE | Fosphenytoin 20 mg PE/kg IV, Levetiracetam 60 mg/kg, or Valproate 40 mg/kg | ~50-60% of BZD failures |
| 20-40 min | Continued SE | Repeat or switch second-line agent | -- |
| >40 min | Refractory SE | Propofol, midazolam drip, or pentobarbital; intubate | -- |
| >24 hr on anesthetics | Super-refractory SE | Ketamine, burst suppression, consider ketogenic diet | Mortality 30-50% |

- RSE occurs in 23-43% of SE patients
- RSE mortality is 3x that of non-refractory SE
- Key: Every minute of delay in first benzodiazepine correlates with longer time to seizure cessation and higher escalation rate

### EEG Patterns
- **Active seizure:** Continuous rhythmic epileptiform discharges
- **Burst suppression target:** Titrate IV anesthetics to 10-second interburst intervals (though evidence for specific suppression ratio is weak)
- **Resolution:** Aim for seizure cessation on EEG, NOT necessarily burst suppression
- Continuous EEG monitoring required for all intubated/pharmacologically paralyzed patients
- 47.9% of patients re-seize after weaning IV anesthetic therapy

### Lactate Trajectory
- **During seizure:** Lactate rises 8.7-fold from baseline; levels of 8-15+ mmol/L common in generalized convulsive SE
- **Diagnostic cutoff:** Lactate >2.45 mmol/L within 1 hour = sensitivity 0.94 / specificity 0.93 for tonic-clonic seizure
- **Prehospital:** Lactate >4.75 mmol/L strongly suggests seizure
- **Resolution:** Self-limited; half-life ~1 hour; normalizes within 1-2 hours of seizure cessation even without treatment

### CK / Rhabdomyolysis
- **CK rise:** Begins within hours, peaks on day 2-4 post-event
- **Rhabdomyolysis risk:** Increases with seizure duration; prolonged/serial seizures and SE carry highest risk
- **CK stays elevated** for several days post-event (unlike lactate which clears rapidly)

### GCS Trajectory
- **Ictal/immediately postictal:** GCS 3-6
- **Post-seizure control:** Gradual improvement over 12-48 hours if no anoxic injury
- **If on IV anesthetics:** GCS not assessable; follow EEG
- Persistent depressed GCS after seizure cessation suggests anoxic injury or ongoing non-convulsive SE

### Sources
- [EMCrit - Status Epilepticus](https://emcrit.org/ibcc/sz/)
- [PMC - Acute Metabolic Effects of Seizures](https://pmc.ncbi.nlm.nih.gov/articles/PMC6885665/)
- [PMC - Rhabdomyolysis Following Status Epilepticus](https://journals.lww.com/md-journal/fulltext/2018/06290/rhabdomyolysis_following_status_epilepticus_with.72.aspx)

---

## Thyroid Storm

### Vital Signs at Presentation
- **HR:** >140 bpm, frequently 150-180+; atrial fibrillation common
- **Temperature:** 39-41+ C (102-106 F); fever of 104-106 F with diaphoresis is classic
- **BP:** Wide pulse pressure initially, then hypotension as decompensation progresses
- **Burch-Wartofsky Score:** >45 = thyroid storm; 25-45 = impending storm; <25 = unlikely

### Lab Values
- **TSH:** Suppressed/undetectable (<0.01 mIU/L)
- **Free T4:** Elevated (but NOT necessarily higher than uncomplicated hyperthyroidism -- the storm is a clinical diagnosis, not a lab diagnosis)
- **Free T3:** Elevated; T3 levels more closely correlate with clinical severity
- Key point: Lab values do NOT distinguish storm from plain thyrotoxicosis. The Burch-Wartofsky score is clinical.

### Treatment Sequence and Timeline
1. **Beta-blocker (minute 0):** Esmolol loading 250-500 mcg/kg, then 50-100 mcg/kg/min drip. Propranolol 40-80 mg PO q4-6h if hemodynamically stable. HR control begins within minutes.
2. **Thionamide (minute 0-15):** PTU 500-1000 mg loading, then 250 mg q4h. Blocks new hormone synthesis within 1-2 hours. PTU preferred over methimazole because it also blocks peripheral T4-to-T3 conversion. PTU drops T3 by 45% in 24 hours vs methimazole at only 10-15%.
3. **Glucocorticoid (concurrent):** Hydrocortisone 300 mg load then 100 mg q8h, OR dexamethasone 2 mg q6h. Blocks peripheral T4-to-T3 conversion, treats relative adrenal insufficiency.
4. **Iodine (>60 min after thionamide):** SSKI 5 drops q6h or Lugol's solution. MUST wait at least 1 hour after thionamide to prevent iodine from being used as substrate for new hormone synthesis.

### Response Timeline
- **Hours 1-6:** HR should begin falling with beta-blockade
- **Hours 12-24:** Temperature should start trending down; clinical improvement expected within 24 hours
- **24-48 hours:** Substantial clinical improvement in most patients
- **Up to 1 week:** Full resolution of the precipitating cause
- **Mortality:** 10-30% even with treatment; higher with delayed recognition

### Improving vs Worsening
- **Improving:** HR <120 within 12-24h, defervescence within 24-48h, clearing mentation
- **Worsening:** Refractory tachycardia, new atrial fibrillation, progressive hypotension, obtundation, multi-organ failure. Mortality with cardiovascular collapse approaches 90%.

### Sources
- [EMCrit - Thyroid Storm](https://emcrit.org/ibcc/tstorm/)
- [StatPearls - Thyroid Storm](https://www.ncbi.nlm.nih.gov/books/NBK448095/)
- [MDCalc - Burch-Wartofsky Score](https://www.mdcalc.com/calc/3816/burch-wartofsky-point-scale-bwps-thyrotoxicosis)
