from json import load
from pathlib import Path
import pandas as pd
from pod_predictor import DEFAULT_VALUES


# Standardized set of values representing missing or null data across inconsistent database entries
NULL = {None, 'NaN', '', ' '}


def load_variable_mapping():
    """
    Loads VARIABLE_MAPPING.json from the same directory as preprocessing.py.

    Returns:
        dict: The contents of VARIABLE_MAPPING.json.
    """
    json_path = Path(__file__).parent / 'VARIABLE_MAPPING.json'
    with json_path.open('r') as f:
        return load(f)


def safe_get(sample, key, sentinel=None, cast=int):
    """
    Helper function to safely get and cast a value from a dict.

    Returns the sentinel if the key is missing, value is null-like, or cast fails.
    Also handles 'True'/'False' strings safely.
    """
    val = sample.get(key, sentinel)
    
    if val in NULL:
        return sentinel

    val_str = str(val).strip().lower()
    if val_str == "true":
        return True
    if val_str == "false":
        return False

    try:
        return cast(val)
    except (ValueError, TypeError):
        return sentinel

    
def count_true_values(sample, prefix):
    """
    Counts the number of True values for keys with a given prefix and fixed suffixes 1–5.

    Args:
        sample (dict): Dictionary containing patient data.
        prefix (str): e.g., 'preop_moca_4'

    Returns:
        int: Number of True values.
    """
    return sum(bool(safe_get(sample, f"{prefix}_{i}")) for i in range(1, 6))


def resolve_conflicts(sample):
    """
    Resolves conflicts between preop_moca_4_* and preop_moca_5_* fields in-place.
    If both values are equal (True), sets preop_moca_5_* to False.

    Args:
        sample (dict): Dictionary containing patient data.

    Returns:
        None
    """
    for i in range(1, 6):
        if safe_get(sample, f'preop_moca_4_{i}') == safe_get(sample, f'preop_moca_5_{i}'):
            sample[f'preop_moca_5_{i}'] = False

    return None


def get_moca_memory(sample):
    """
    Computes the MoCA memory score from preop_moca_4_* and preop_moca_5_* fields.

    Returns None if all relevant fields are null-like.

    Args:
        sample (dict): Dictionary containing patient data.

    Returns:
        int or None: MoCA memory score.
    """
    if all(sample.get(f"preop_moca_4_{i}") in NULL for i in range(1, 6)) and \
       all(sample.get(f"preop_moca_5_{i}") in NULL for i in range(1, 6)):
        return None

    resolve_conflicts(sample)

    score = (
        2 * count_true_values(sample, "preop_moca_4") +
        count_true_values(sample, "preop_moca_5")
    )

    return score


def get_number_of_medications(sample):
    """
    Counts the number of preoperative medications recorded in the sample.

    Scans keys 'medication_preop_0' through 'medication_preop_20' and
    returns the number of non-None fields. Returns None if all are None.
    """
    count = sum(
        sample[f'medication_preop_{i}'] not in NULL.union({False, 0})
        for i in range(1, 21)
    )
    return count if count > 0 else None # Returns None if no medications are recoreded, which is likey incomplete data


def get_cci(sample):
    """
    Calculates the Charlson Comorbidity Index (CCI) for a single patient sample.

    Parameters:
        sample (dict): A dictionary containing the required 'comorbidity_1_*' fields.

    Returns:
        int: The computed CCI score based on comorbidity indicators.
    """

    mi = (
        safe_get(sample, 'comorbidity_1_mi4w') == 1 or
        safe_get(sample, 'comorbidity_1_mi1m') == 1
    )
    chf = safe_get(sample, 'comorbidity_1_myocard', sentinel=-1) > 1
    pvd = safe_get(sample, 'comorbidity_1_vasc_avk') == 1
    cvd = safe_get(sample, 'comorbidity_1_apoplextia', sentinel=-1) >= 0
    dementia = get_dementia(sample) == 1
    cpd = (
        safe_get(sample, 'comorbidity_1_asthma') == 1 or
        safe_get(sample, 'comorbidity_1_copd') == 1 or
        safe_get(sample, 'comorbidity_1_res') == 1
    )
    ld = safe_get(sample, 'comorbidity_1_liver', sentinel=-1) > 1
    ld_s = safe_get(sample, 'comorbidity_1_liver') == 5
    dm = safe_get(sample, 'comorbidity_1_diabetes', sentinel=-1) > 1
    dm_c = safe_get(sample, 'comorbidity_1_diab_cons') == 1
    rd = safe_get(sample, 'comorbidity_1_kidney', sentinel=-1) > 1

    cci_score = (
        int(mi) +
        int(chf) +
        int(pvd) +
        int(cvd) +
        int(dementia) +
        int(cpd) +
        int(ld) +
        2 * int(ld_s) +
        int(dm) +
        int(dm_c) +
        2 * int(rd)
    )

    return cci_score


def get_dementia(sample):
    """
    Determines presence of dementia based on anesthetist and anamnesis assessments.

    Parameters:
        sample (dict): A dictionary containing the keys
            'comorbidity_1_dementia' and 'comorbidity_2_dementia' with their corresponding values.

    Returns:
        int or float:
            - 1 if dementia is present (anesthetist == 1 or anamnesis in [1, 2])
            - 0 if dementia is absent (anesthetist == 0 or anamnesis == 0)
            - None if the information is inconclusive or missing
    """
    anesthetist = safe_get(sample, 'comorbidity_1_dementia')
    anamnesis = safe_get(sample, 'comorbidity_2_dementia')

    if anesthetist == 1 or anamnesis in {1, 2}:
        return 1
    elif anesthetist == 0 or anamnesis == 0:
        return 0
    else:
        return None


def get_isolation(sample):
    """
    Returns the most recent available isolation status from the sample dictionary.

    Checks keys in priority order: 'isolation_postop_3' (latest), then 'isolation_postop_1',
    then 'isolation_preop' (earliest). Returns the first non-None value found.

    Args:
        sample (dict): Dictionary potentially containing isolation keys.

    Returns:
        value or None: Value of the most recent isolation key found, or None if none exist.
    """
    for key in ('isolation_postop_3', 'isolation_postop_1', 'isolation_preop'):
        if key in sample and safe_get(sample, key) not in NULL:
            return sample[key]
    return None


def get_benzodiazepine(sample):
    """
    Checks if any preoperative medication in the sample is a benzodiazepine.

    Args:
        sample (dict): Dictionary with keys 'medication_preop_0' to 'medication_preop_20'.

    Returns:
        int or None: 1 if benzodiazepine found, 0 if not, None if data likely missing.
    """
    benzodiazepine = [
        'Diazepam', 'Diazepanum', 'Chlordiazepoxid', 'Alprazolam', 'Bromazepam',
        'Fludiazepam', 'Lorazepam', 'Medazepam', 'Oxazepam', 'Flurazepam',
        'Flunitrazepam', 'Lormetazepam', 'Midazolam', 'Nitrazepam', 'Triazolam',
        'Clonazepam', 'Tetrazepam'
    ]

    # If no medication is documented, likely incomplete data
    if sample.get('medication_preop_1') in NULL.union({False, 0}):
        return None

    for i in range(1, 21):
        med = sample[f'medication_preop_{i}']
        if med in benzodiazepine:
            return 1

    return 0


def get_features(sample):
    """
    Maps or computes feature values for a given sample using a variable mapping dictionary.

    Parameters:
        sample (dict): A dictionary containing source variables for a single patient.
        variable_mapping (dict): A dictionary defining how each target variable is computed or mapped.
            - str: Direct mapping from another key in sample, or a function name starting with 'get'.
            - dict: Nested mapping with source variable(s) and value mappings.

    Returns:
        dict: The input dictionary updated with new or transformed features based on the mapping.
    """

    variable_mapping = load_variable_mapping()
    feature_functions = {
        'get_moca_memory': get_moca_memory,
        'get_number_of_medications': get_number_of_medications,
        'get_cci': get_cci,
        'get_dementia': get_dementia,
        'get_isolation': get_isolation,
        'get_benzodiazepine': get_benzodiazepine,
        'get_age': lambda s: v * 12 if (v := safe_get(s, 'alter', cast=float)) else None,
        'get_moca_orientation': lambda s: v.split(',').count("True") if (v := safe_get(s, 'moca_preop_3', cast=str)) else None,
        'get_moca_verbal_fluency': lambda s: v / 2 if (v := safe_get(s, 'moca_preop_2')) is not None else None,
    }

    for key, value in variable_mapping.items():
        if isinstance(value, str):
            if value in feature_functions:
                sample[key] = feature_functions[value](sample)
            else:
                sample[key] = safe_get(sample, value, cast=float)
        elif isinstance(value, dict):
            # Mapping required
            for variable, mapping in value.items():
                raw_val = safe_get(sample, variable, sentinel=99)
                sample[key] = mapping.get(str(raw_val))
        else:
            continue

    return sample


def prepare_sample(sample, default_values=DEFAULT_VALUES):
    """
    Extracts a single patient sample into a one-row DataFrame matching the model's expected input format.

    For each feature in `default_values`, this function attempts to retrieve its value from `sample`.
    If a feature is missing, it is set to None (missing). Default values are not applied here — they may
    be used later in the pipeline (e.g., during imputation).

    Parameters:
        sample (dict): Raw input dictionary containing patient data.
        default_values (dict): Dictionary whose keys define the required features for the model.
                               The values are ignored in this function but may be used later for filling.

    Returns:
        pd.DataFrame: A one-row DataFrame with all required features (columns), possibly containing NaNs.
    """

    sample = get_features(sample)

    row = {
        feature: sample.get(feature, None)
        for feature in default_values.keys()
    }

    return pd.DataFrame([row], dtype=float)
