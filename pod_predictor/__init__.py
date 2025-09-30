# This file marks the 'pod_predictor' directory as a Python package.
# It can be used to initialize package-level settings or
# perform additional setup in the future if needed.

COEFFICIENTS = {
    'Estimated Cut-to-Suture Time (minutes)': 0.53,
    'Age (months)': 0.10,
    'GFR (Cockcroft-Gault, ml/min)': -0.03,
    'ASA Class (score)': 0.39,
    'MoCA Orientation (subscore)': -0.27,
    'MoCA Memory (subscore)': -0.34,
    'Number of Medications (n)': 0.11,
    'Multimorbidity (score)': 0.14,
    'Clinical Frailty Scale (score)': 0.21,
    'MoCA Verbal Fluency (subscore)': -0.26,
    'Dementia (Yes/No)': 2.53,
    'Recent Fall (Yes/No)': 0.31,
    'Post-OP Isolation (Yes/No)': 0.15,
    'Pre-OP Benzodiazepines (Yes/No)': 0.42,
    'Cardio-Pulmonary Bypass (Yes/No)': 0.53
}

DEFAULT_VALUES = {
    'Estimated Cut-to-Suture Time (minutes)': 146,
    'Age (months)': 934,
    'GFR (Cockcroft-Gault, ml/min)': 69,
    'ASA Class (score)': 3,
    'MoCA Orientation (subscore)': 6,
    'MoCA Memory (subscore)': 2,
    'Number of Medications (n)': 6,
    'Multimorbidity (score)': 1,
    'Clinical Frailty Scale (score)': 3,
    'MoCA Verbal Fluency (subscore)': 0,
    'Dementia (Yes/No)': 0,
    'Recent Fall (Yes/No)': 0,
    'Post-OP Isolation (Yes/No)': 0,
    'Pre-OP Benzodiazepines (Yes/No)': 0,
    'Cardio-Pulmonary Bypass (Yes/No)': 0
}

NORMALIZATION_MEAN_SD = {
    'Estimated Cut-to-Suture Time (minutes)': (146.05, 85.29),
    'Age (months)': (934.47, 58.86),
    'GFR (Cockcroft-Gault, ml/min)': (69.21, 22.39),
    'ASA Class (score)': (2.81, 0.6),
    'MoCA Orientation (subscore)': (5.85, 0.56),
    'MoCA Memory (subscore)': (2.24, 1.66),
    'Number of Medications (n)': (6.09, 3.42),
    'Multimorbidity (score)': (1.27, 1.35),
    'Clinical Frailty Scale (score)': (3.62, 1.37)
}
