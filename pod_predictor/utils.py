import pandas as pd


def load_file(path_to_file):
    """
    Load a CSV file from the specified path and return a DataFrame.

    Args:
    path_to_file (str): The file path of the CSV file to be loaded.

    Returns:
    pandas.DataFrame: A DataFrame containing the contents of the CSV file.

    Raises:
    FileNotFoundError: If the file is not found at the provided path.
    """
    try:
        data = pd.read_csv(path_to_file)
    except FileNotFoundError:
        raise

    return data


def preprocess_data(data):
    """
    Preprocess data by converting 'yes'/'no' to 1/0.

    Args:
    data (pandas.DataFrame): The DataFrame containing the data to be preprocessed.

    Returns:
    pandas.DataFrame: A DataFrame containing the preprocessed data.
    """
    # TO DO: Implement option that calculates the multimorbidity score from diagnoses.
    data = data.replace(
        {'yes': 1, 'Yes': 1, 'YES': 1, 'no': 0, 'No': 0, 'NO': 0})
    return data


def load_data(path_to_file):
    """
    Load, preprocess, and normalize data from a CSV file.

    This function loads a CSV file, preprocesses the data by converting specified string values to integers, and returns the processed DataFrame.

    Args:
    path_to_file (str): The file path of the CSV file to be loaded.

    Returns:
    pandas.DataFrame: A DataFrame containing the preprocessed data.

    Raises:
    FileNotFoundError: If the file is not found at the provided path.
    """
    data = load_file(path_to_file)
    data = preprocess_data(data)
    return data


def preprocess(func):
    """
    Decorator to preprocess the input data before applying a method if the data is not the calibration set.

    This decorator checks if the input data is the calibration set (i.e., self.X). If not, it preprocesses the data using
    `self.preprocess_input` and `self.impute` methods before applying the decorated function.

    Args:
    func: The method to be decorated.

    Returns:
    A decorated function with preprocessed input data if the data is the calibration set.
    """

    def wrapper(self, X, *args, **kwargs):
        if not hasattr(self, 'X') or X is not self.X:
            X = self.preprocess_input(X)
            if func.__name__ == "feature_importance":
                X = self.normalize(X)
            else:
                X = self.impute(X) # Feature importance should not contain imputed data
        return func(self, X, *args, **kwargs)
    return wrapper
