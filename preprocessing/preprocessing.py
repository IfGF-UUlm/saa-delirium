from json import load
from pathlib import Path
import pandas as pd
from pod_predictor import DEFAULT_VALUES


# Standardized set of values representing missing or null data across inconsistent database entries
NULL = {None, 'NaN', '', ' '}


def load_variable_mapping():
    """
    Loads VARIABLE_MAPPING.json from the same directory as utils.py.

    Returns:
        dict: The contents of VARIABLE_MAPPING.json.
    """
    json_path = Path(__file__).parent / 'VARIABLE_MAPPING.json'
    with open(json_path, 'r') as f:
        return load(f)


def safe_get(sample, key, sentinel=-1, cast=int):
    """
    Helper function to safely get and cast a value from a dict.

    Returns the sentinel if the key is missing, value is None, or cast fails.
    """
    val = sample.get(key, sentinel)
    if val in NULL:
        return sentinel
    try:
        return cast(val)
    except (ValueError, TypeError):
        return sentinel

    
def count_true_values(arr_str):
    """
    Counts the number of 'True' string values in a comma-separated string.
    
    If the input is None, returns None.
    """
    if arr_str in NULL:
        return None
    else:
        return sum(x == "True" for x in arr_str.split(','))


def resolve_conflicts(arr1_str, arr2_str):
    """
    Resolves conflicts between two comma-separated 'True'/'False' strings.
    If both have 'True' at the same index, arr2's value is set to 'False' at that index.

    Args:
        arr1_str (str): First boolean array as a comma-separated string.
        arr2_str (str): Second boolean array as a comma-separated string.

    Returns:
        str: Modified arr2 string with conflicts resolved.
    """
    arr1 = arr1_str.split(',')
    arr2 = arr2_str.split(',')

    resolved_arr2 = [
        "False" if a1 == "True" and a2 == "True" else a2
        for a1, a2 in zip(arr1, arr2)
    ]

    return ','.join(resolved_arr2)


def get_moca_memory(sample):
    """
    Computes MoCA memory score from 'moca_preop_4' and 'moca_preop_5'.
    Returns None if both inputs are None.

    Args:
        sample (dict): Contains 'moca_preop_4' and 'moca_preop_5' as comma-separated 'True'/'False' strings or None.

    Returns:
        int or None: MoCA memory score or None if no data.
    """
    if sample['moca_preop_4'] in NULL and sample['moca_preop_5'] in NULL:
        return None
    else:
        resolved = resolve_conflicts(sample['moca_preop_4'], sample['moca_preop_5'])
        moca_memory =  2 * count_true_values(sample['moca_preop_4']) + count_true_values(resolved)
        return moca_memory


def get_number_of_medications(sample):
    """
    Counts the number of preoperative medications recorded in the sample.

    Scans keys 'medication_preop_0' through 'medication_preop_20' and
    returns the number of non-None fields. Returns None if all are None.
    """
    count = sum(
        sample[f'medication_preop_{i}'] not in NULL
        for i in range(1, 21)
    )
    return count if count > 0 else None # Returns 0 if no medications are recoreded, which is likey incomplete data


def get_cci(sample):
    """
    Calculates the Charlson Comorbidity Index (CCI) for a single patient sample.

    Parameters:
        sample (dict): A dictionary containing the required comorbidity fields,
            including 'dementia' and 'comorbidity_1_*' keys.

    Returns:
        int: The computed CCI score based on comorbidity indicators.
    """

    mi = (
        safe_get(sample, 'comorbidity_1_mi4w') == 1 or
        safe_get(sample, 'comorbidity_1_mi1m') == 1
    )
    chf = safe_get(sample, 'comorbidity_1_myocard') > 1
    pvd = safe_get(sample, 'comorbidity_1_vasc_avk') == 1
    cvd = safe_get(sample, 'comorbidity_1_apoplextia') >= 0
    dementia = get_dementia(sample) == 1
    cpd = (
        safe_get(sample, 'comorbidity_1_asthma') == 1 or
        safe_get(sample, 'comorbidity_1_copd') == 1 or
        safe_get(sample, 'comorbidity_1_res') == 1
    )
    ld = safe_get(sample, 'comorbidity_1_liver') == 2
    ld_s = safe_get(sample, 'comorbidity_1_liver') == 4
    dm = safe_get(sample, 'comorbidity_1_diabetes') > 1
    dm_c = safe_get(sample, 'comorbidity_1_diab_cons') == 1
    rd = safe_get(sample, 'comorbidity_1_kidney') > 1

    cci_score = (
        int(mi) +
        int(chf) +
        int(pvd) +
        int(cvd) +
        int(dementia) +
        int(cpd) +
        int(ld) +
        3 * int(ld_s) +
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

    if anesthetist == 1 or anamnesis in [1, 2]:
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
        if key in sample and sample[key] not in NULL:
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
    if sample.get('medication_preop_1') in NULL:
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
        'get_age': lambda sample: sample['adm_age'] * 12 if sample['adm_age'] not in NULL else None,
        'get_moca_orientation': lambda sample: count_true_values(sample['moca_preop_3']),
        'get_moca_verbal_fluency': lambda sample: safe_get(sample, 'moca_preop_2')/2 if sample['moca_preop_2'] not in NULL else None,
    }

    for key, value in variable_mapping.items():
        if isinstance(value, str):
            if value in feature_functions:
                sample[key] = feature_functions[value](sample)
            else:
                sample[key] = sample.get(value)
        elif isinstance(value, dict):
            # Mapping required
            for variable, mapping in value.items():
                raw_val = sample.get(variable, "99")
                sample[key] = mapping.get(str(int(raw_val)))
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
