from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from feature_engine.imputation import MeanMedianImputer
from feature_engine.encoding import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

class FeatureSelector:

    """
    A filter-based feature selection method using ANOVA F-value.

    This class implements a feature selection method that automatically selects the top k features
    based on the ANOVA F-value. It is considered a filter-based approach, where features are ranked
    and selected based on their statistical significance.

    Parameters:
    - k (int): The number of top features to select.

    Usage:
    >>> feature_selector = FeatureSelector(k=5)
    >>> X_train_selected = feature_selector.fit_transform(X_train, y_train)

    Attributes:
    - k (int): The number of top features to select.

    Methods:
    - fit_transform(X, y): Fit the feature selector on the input features X and target variable y,
      and return the selected features.
    """
    def __init__(self, k=None, method = 'filter', top_k=None):
        """
        Parameters:
        - k: Number of top features to select (None if not specified)
        """
        self.k = k
        self.top_k = top_k
        self.method = method
        self.selector = None
        self.imputer = MeanMedianImputer(imputation_method='mean', variables=None)
        self.encoder = OrdinalEncoder(encoding_method='arbitrary', variables=None)

    def fit_transform(self, X, y):
        # Handle missing values in target variable y
        y_imputed = self.imputer.fit_transform(y.to_numpy().reshape(-1, 1)).squeeze()

        # Handle missing values in input features X
        X_imputed = self.imputer.fit_transform(X)

        # Encode categorical variables
        if X_imputed.select_dtypes(include=['object']).shape[1] > 0:
            X_encoded = self.encoder.fit_transform(X_imputed)
        else:
            X_encoded = X_imputed.copy()

        # Check for NaN or Infinite values in y_imputed
        nan_indices_y = np.isnan(y_imputed)
        inf_indices_y = np.isinf(y_imputed)
        if nan_indices_y.any() or inf_indices_y.any():
            imputer_y = SimpleImputer(strategy='mean')
            y_imputed[nan_indices_y] = imputer_y.fit_transform(y_imputed[nan_indices_y].reshape(-1, 1)).squeeze()
            y_imputed[inf_indices_y] = imputer_y.fit_transform(y_imputed[inf_indices_y].reshape(-1, 1)).squeeze()

        if self.k is None:
            print("Number of features not specified. Feature selection will not be performed.")
            return X_encoded

        self.selector = SelectKBest(score_func=f_regression, k=self.k)
        X_selected = self.selector.fit_transform(X_encoded, y_imputed)

        return X_selected

    def transform(self, X):
        """
        Transform the input features.

        Parameters:
        - X: Input features

        Returns:
        - Transformed features
        """
        if self.selector is not None:
            X_selected = self.selector.transform(X)
            return X_selected
        else:
            print("Fit the feature selector first using fit_transform method.")
            return X
    def check_missing_values(self, dataset):
        """
        Check for missing values in the dataset.

        Parameters:
        - dataset: An instance of the GenericDataset class.

        Returns:
        - True if missing values are found, False otherwise.
        """
        return dataset.check_missing_values()
    def plot_feature_importance(self, feature_names, feature_scores, top_k=None):
        sorted_idx = np.argsort(feature_scores)[::-1]
        sorted_feature_names = np.array(feature_names)[sorted_idx]

        plt.figure(figsize=(12, 6))
        if top_k is not None:
            plt.bar(range(top_k), feature_scores[sorted_idx][:top_k], align="center")
            plt.xticks(range(top_k), sorted_feature_names[:top_k], rotation=45)
        else:
            plt.bar(range(len(feature_scores)), feature_scores[sorted_idx], align="center")
            plt.xticks(range(len(feature_scores)), sorted_feature_names, rotation=45)
        plt.title("Top Features by Importance")
        plt.xlabel("Feature")
        plt.ylabel("Score")
        plt.show()

    
    def plot_feature_correlation(self, X):
        if isinstance(X, np.ndarray):
            # Convert NumPy array to pandas DataFrame
            X = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(X.shape[1])])

        corr_matrix = X.corr()
        plt.figure(figsize=(9, 5))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Feature Correlation Heatmap')
        plt.show()

    def plot_feature_distribution(self, X):
        if isinstance(X, np.ndarray):
            # Convert NumPy array to pandas DataFrame
            X = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(X.shape[1])])

        plt.figure(figsize=(16, 12))
        for column in X.columns:
            sns.histplot(X[column], kde=True, label=column)

        plt.title('Feature Distributions')
        plt.xlabel('Feature Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    def check_correlation(self, X, threshold=0.9):
        """
        Check for highly correlated features.

        Parameters:
        - X: Input features
        - threshold: Threshold for correlation. Features with correlation above this threshold will be identified.

        Returns:
        - A DataFrame containing correlated feature pairs.
        """
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        correlated_pairs = np.where(upper_triangle > threshold)
        correlated_features = set()
        for i in range(len(correlated_pairs[0])):
            feature1 = X.columns[correlated_pairs[0][i]]
            feature2 = X.columns[correlated_pairs[1][i]]
            correlated_features.add(feature1)
            correlated_features.add(feature2)
        return pd.DataFrame(list(correlated_features), columns=['Correlated Features'])

    def handle_correlation(self, X, threshold=0.9, strategy='drop'):
        """
        Handle correlated features in the dataset.

        Parameters:
        - X: Input features
        - threshold: Threshold for correlation. Features with correlation above this threshold will be identified.
        - strategy (str): The strategy for handling correlated features. 'drop' to drop correlated features,
          'average' to average them (default is 'drop').

        Returns:
        - Transformed features after handling correlated features.
        """
        correlated_features = self.check_correlation(X, threshold)
        if strategy == 'drop':
            X = X.drop(columns=correlated_features['Correlated Features'])
        elif strategy == 'average':
            for feature in correlated_features['Correlated Features']:
                features_to_average = correlated_features['Correlated Features'].to_list()
                features_to_average.remove(feature)
                X[feature] = X[features_to_average].mean(axis=1)
                X = X.drop(columns=features_to_average)
        return X
    
class RegressionModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        return mse

class ClassificationModel:
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        return accuracy
