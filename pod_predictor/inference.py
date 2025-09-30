import numpy as np
import pandas as pd
from pod_predictor import COEFFICIENTS, DEFAULT_VALUES, NORMALIZATION_MEAN_SD
from pod_predictor.utils import load_data, preprocess
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from venn_abers import VennAbersCalibrator
import warnings


class PODPredictor:
    def __init__(self, path_to_file='./data/calibration.csv', calibration=None, imputation=None):
        """
        Initialize a new PODPredictor instance.

        This class is used to predict postoperative delirium (POD) using our pre-trained model. The constructor loads
        the calibration dataset from the specified path, preprocesses the data, and initializes the imputer based on the
        provided imputation method.

        Args:
        path_to_file (str, optional): The file path of the calibration dataset. Defaults to './data/calibration.csv'.
        calibration (str, optional): The calibration method to use. Can be None, 'platt', or 'va'. Defaults to None.
        imputation (str, optional): The imputation method to use. Can be None or 'knn'. Defaults to None.
        """

        self.coefficients = np.array(list(COEFFICIENTS.values()))
        self.default_values = DEFAULT_VALUES
        self.normalization = NORMALIZATION_MEAN_SD

        if calibration or imputation:

            # Loading Data
            try:
                self.data = load_data(path_to_file)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Calibration dataset '{path_to_file}' not found.")

            for key in self.data.columns:
                if key != 'Delirium' and key not in self.default_values.keys():
                    raise KeyError(
                        f"Key {key} not found in data. Use keys in data/calibration_template.csv.")

            self.X = self.data.drop(['Delirium'], axis=1)
            self.y = self.data['Delirium'].values

            # Normalization and Imputation
            if imputation is None:
                self.imputer = None
            elif imputation == 'knn':
                self.imputer = KNNImputer(n_neighbors=5)
                self.imputer.fit(self.normalize(self.X))
            # TO DO: Add more imputation methods (e.g. MissForest)
            else:
                warnings.warn(
                    f"Invalid imputation value '{imputation}'. Proceeding with default None", UserWarning)
                self.imputer = None

            self.X = self.impute(self.X)

            # Calibration
            if calibration is None:
                self.calibrator = None
            elif calibration == 'platt':
                self.calibrator = LogisticRegression()
                self.calibrator.fit(self.decision_function(
                    self.X).reshape(-1, 1), self.y)
            elif calibration == 'va':
                self.calibrator = VennAbersCalibrator()
                # VennAbersCalibrator class from venn_abers (MIT license)
                # Source: https://github.com/ip200/venn-abers
                # Copyright (c) 2023 Ivan Petej
            else:
                warnings.warn(
                    f"Invalid calibration value '{calibration}'. Proceeding with default None", UserWarning)
                self.calibration = None

        else:
            self.imputer = None
            self.calibrator = None

    def preprocess_input(self, X_test):
        """
        Preprocess the given input data for prediction.

        The method checks the input type and ensures it has the correct dimensions, format, and keys.

        Args:
        X_test (numpy.ndarray, dict, or pandas.DataFrame): The input data to preprocess.

        Raises:
        TypeError: If the input is not a numpy array, pandas DataFrame, or a dictionary.
        ValueError: If the input is a numpy array with incorrect dimensions.
        KeyError: If a key is not found in the JSON template for the data.

        Returns:
        pandas.DataFrame: The preprocessed input data with the correct format, dimensions, and keys.
        """

        if isinstance(X_test, np.ndarray):
            X_test = X_test.reshape(
                1, -1) if len(X_test.shape) == 1 else X_test
            if X_test.shape[1] != 15:
                raise ValueError(
                    f"Expected X_test to have shape (n, 15), but got array with shape {X_test.shape} instead.")
            X_test = pd.DataFrame(
                data=X_test, columns=list(self.default_values.keys()))

        if isinstance(X_test, dict):
            X_test = pd.DataFrame(X_test)

        if not isinstance(X_test, pd.DataFrame):
            raise TypeError(
                f"Expected a np.array, pd.DataFrame or dictionary as input, but got {type(X_test).__name__} instead.")

        for key in X_test.columns:
            if key not in self.default_values.keys():
                raise KeyError(
                    f"Key '{key}' not found in Features. Use keys in data/JSON_template.json.")

        try:
            # Check and reorder DataFrame
            X_test = X_test[list(self.default_values.keys())]
        except KeyError as e:
            key = e.args[0].strip("'['] not in index")
            raise KeyError(
                f"Key '{key}' not found in Features. Use keys in data/JSON_template.json.")

        return X_test

    def normalize(self, X):
        """
        Normalize data columns using mean and standard deviation.

        Args:
        X (pandas.DataFrame): DataFrame to be normalized.

        Returns:
        pandas.DataFrame: Normalized data.
        """

        X = X.copy()
        for (key, value) in self.normalization.items():
            X[key] = X[key].apply(lambda x: (x - value[0]) / value[1])

        return X

    def impute(self, X):
        """
        Impute missing values in the given input data.

        Args:
        X (pandas.DataFrame): The input data to impute missing values for.

        Returns:
        pandas.DataFrame: The input data with missing values imputed.
        """

        if self.imputer is None:
            for (key, value) in self.default_values.items():
                X[key] = X[key].fillna(value)
            X = self.normalize(X)

        else:
            X = self.normalize(X)
            X = pd.DataFrame(data=self.imputer.transform(
                X), columns=X.columns)

        return X

    # The decorator preprocesses the input data, unless it is the (already preprocessed) calibration data 'self.X'.
    # This distinction allows testing the uncalibrated model on the calibration data.
    @preprocess
    def decision_function(self, X):
        """
        Calculate the decision function for the given input data.

        Args:
        X (pandas.DataFrame): The input data to calculate the decision function for.

        Returns:
        numpy.ndarray: The decision function values.
        """

        z = np.dot(X, self.coefficients.T) - 0.61
        return z

    # The decorator preprocesses the input data, unless it is the (already preprocessed) calibration data 'self.X'.
    # For the following function, the impute process is excluded.
    @preprocess
    def feature_importance(self, X):
        """
        Calculate the feature importance for the given input data.

        The method calculates the feature importance by element-wise multiplication of the input data with the coefficients.

        Args:
        X (pandas.DataFrame): The input data to calculate the feature importance for.

        Returns:
        pandas.DataFrame: The feature importance values.
        """

        feature_importance = np.multiply(
            X, self.coefficients)
        return feature_importance

    def naive_proba(self, X):
        """
        Calculate the naive probability estimates for the given decision function value.

        Args:
        X (pandas.DataFrame): The input data to calculate the naive probability estimates for.

        Returns:
        numpy.ndarray: A 2D numpy array containing the naive probability estimates for the given decision function value.
        """

        z = self.decision_function(X)
        result = 1/(1 + np.exp(-0.97 * z + 1.07))
        return np.column_stack((1 - result, result))

    def predict(self, X_test):
        """
        Predict the binary outcome of postoperative delirium for the given input data.

        This method applies the decision function to the input data and uses the sign function to return a binary prediction.
        Note: In general, probabilistic outputs are preferred, as they convey the uncertainty of the prediction.

        Args:
        X_test (pandas.DataFrame): The input data to predict outcomes for.

        Returns:
        numpy.ndarray: A binary array with 0 indicating no delirium and 1 indicating delirium.
        """

        z = self.decision_function(X_test)
        return np.where(z < 0, 0, 1)  # labels 0 as positive cases

    def predict_proba(self, X_test):
        """
        Predict the probability of postoperative delirium for the given input data.

        The method applies the selected calibration technique to the naive probability estimates.

        Args:
        X_test (pandas.DataFrame): The input data to predict probabilities for.

        Returns:
        numpy.ndarray: A 2D numpy array containing the predicted probabilities for the given input data.
        """

        # No Calibration (naive probabilities)
        if self.calibrator is None:
            probas = self.naive_proba(X_test)

        # Platt Scaling
        elif isinstance(self.calibrator, LogisticRegression):
            probas = self.calibrator.predict_proba(
                self.decision_function(X_test).reshape(-1, 1))

        # Venn-ABERS
        elif isinstance(self.calibrator, VennAbersCalibrator):
            probas = self.calibrator.predict_proba(
                p_cal=self.naive_proba(self.X),
                y_cal=self.y,
                p_test=self.naive_proba(X_test),
                p0_p1_output=False
            )

        return probas

    def get_report(self, X_test):
        """
        Generate a report predicting the probability of postoperative delirium, including confidence intervals (if applicable) and feature importance for the given input data.

        The report DataFrame is initialized with the predicted probabilities and confidence intervals (if the Venn-ABERS calibration method is used) and then expanded with feature importance data.

        Args:
            X_test (pandas.DataFrame): The input data used for predicting probabilities and calculating feature importance.

        Returns:
            pandas.DataFrame: A DataFrame containing:
                - 'Delirium Probability': The predicted probability of postoperative delirium.
                - 'Confidence Interval (lower and upper bounds)': The range within which the true probability is expected to lie, if available; otherwise, None.
                - Feature importance scores for each input feature.
        """

        if isinstance(self.calibrator, VennAbersCalibrator):
            va_proba = self.calibrator.predict_proba(
                p_cal=self.naive_proba(self.X),
                y_cal=self.y,
                p_test=self.naive_proba(X_test),
                p0_p1_output=True
            )
            proba = va_proba[0][:, 1]
            ci_0, ci_1 = va_proba[1][:, 0], va_proba[1][:, 1]
        else:
            proba = self.predict_proba(X_test)[:, 1]
            ci_0, ci_1 = None, None

        report = pd.DataFrame({
            'Delirium Probability': proba,
            'Confidence Interval (lower bound)': ci_0,
            'Confidence Interval (upper bound)': ci_1
        })

        feature_importance = self.feature_importance(X_test)
        report = pd.concat([report, feature_importance], axis=1)

        return report
