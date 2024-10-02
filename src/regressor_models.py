from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from typing import Union
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array


class BaseRegressor:
    """
       A base class for regression models that encapsulates fitting, predicting, and evaluation.
       Attributes:
           model: The regression model instance to be used for fitting and predicting.
    """

    def __init__(self, model: RegressorMixin) -> None:
        """
        Initializes the BaseRegressor with a specific regression model.
        Args:
            model (RegressorMixin): An instance of a regression model from scikit-learn.
        """
        self.model = model

    def fit(self, X: Union[pd.DataFrame, np.array], y: Union[pd.Series, np.array]) -> None:
        """
        Fits the regression model to the training data.
        Args:
            X (pd.DataFrame): Features to train the model on.
            y (pd.Series): Target variable to train the model on.
        """
        # X, y = check_X_y(X, y)
        self.model.fit(X, y)
        return self

    def predict(self, X: Union[pd.DataFrame, np.array]) -> Union[pd.Series, np.array]:
        """
        Predicts the target variable for the given features using the fitted model.
        Args:
            X (pd.DataFrame): Features to make predictions on.
        Returns:
            pd.Series: Predicted target variable.
        """

        # input validation
        # X = check_array(X)

        return self.model.predict(X)

    @staticmethod
    def evaluate(y: Union[pd.Series, np.array], y_pred: Union[pd.Series, np.array], metric: str) -> float:
        """
        Evaluates the performance of the model using the specified metric.
        Args:
            y_pred (pd.Series): Predicted value of y.
            y (pd.Series): True target variable for comparison.
            metric (str): The evaluation metric to use. Options are 'rmse', 'mae', or 'r2_score'.
        Returns:
            float: The calculated evaluation metric.
        Raises:
            ValueError: If an unsupported metric is specified.
        """
        if metric == 'rmse':
            # calculates root mean squared error
            return mean_squared_error(y, y_pred) ** 0.5
        elif metric == 'mae':
            # calculates mean absolute error
            return mean_absolute_error(y, y_pred)
        elif metric == 'r2_score':
            # calculates r2_score
            return r2_score(y, y_pred)
        else:
            raise ValueError(f"Metric {metric} is not supported. Use 'mae', 'rmse', or r2 score'. ")

    # Add get_params to conform with scikit-learn API
    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    # Add set_params to conform with scikit-learn API
    def set_params(self, **params):
        self.model.set_params(**params)
        return self


class LinearRegressor(BaseRegressor, RegressorMixin):
    """
    A wrapper class for the Linear Regression model.
    Inherits from BaseRegressor to encapsulate fitting, predicting, and evaluating functionalities.
    Args:
        **kwargs: Additional keyword arguments to pass to the LinearRegression model.
    """

    def __init__(self, **kwargs):
        """
       Initializes the LinearRegressor with the specified parameters.
       Args:
           **kwargs: Additional keyword arguments for the LinearRegression model.
       """
        super().__init__(model=LinearRegression(**kwargs))


class RidgeRegressor(BaseRegressor, RegressorMixin):
    """
    A wrapper class for the Ridge Regression model. This is nothing but Linear least squares with l2 regularization.

    Inherits from BaseRegressor to encapsulate fitting, predicting, and evaluating functionalities.
    Args:
        **kwargs: Additional keyword arguments to pass to the Ridge model.
    """

    def __init__(self, **kwargs):
        """
       Initializes the LinearRegressor with the specified parameters.
       Args:
           **kwargs: Additional keyword arguments for the LinearRegression model.
       """
        super().__init__(model=Ridge(**kwargs))


class RandomForest(BaseRegressor, RegressorMixin):
    """
            A wrapper class for the Random Forest Regression model.
            Inherits from BaseRegressor to encapsulate fitting, predicting, and evaluating functionalities.
            Args:
                **kwargs: Additional keyword arguments to pass to the RandomForestRegressor model.
    """

    def __init__(self, **kwargs):
        """
        Initializes the RandomForest with the specified parameters.
        Args:
            **kwargs: Additional keyword arguments for the RandomForestRegressor model.
        """
        super().__init__(model=RandomForestRegressor(**kwargs))


class XGBoost(BaseRegressor, RegressorMixin):
    """
    A wrapper class for the XGBoost Regression model.
    Inherits from BaseRegressor to encapsulate fitting, predicting, and evaluating functionalities.
    Args:
        **kwargs: Additional keyword arguments to pass to the XGBRegressor model.
    """

    def __init__(self, **kwargs):
        """
        Initializes the XGBoost with the specified parameters.
        Args:
            **kwargs: Additional keyword arguments for the XGBRegressor model.
        """
        super().__init__(model=xgboost.XGBRegressor(**kwargs))


class PreProcessData(BaseEstimator, TransformerMixin):
    """
        A custom data preprocessing class that handles numerical and categorical features.

        This class combines various preprocessing steps into a single pipeline,
        allowing for easy fitting and transformation of data.

        Attributes:
            numerical_cols (list): List of numerical column names.
            categorical_cols (list): List of categorical column names.
            pipeline (Pipeline): The complete preprocessing pipeline.
        """

    def __init__(self,
                 numerical_cols: list[str],
                 categorical_cols: list[str],
                 apply_pca: bool = False,
                 apply_svd: bool = False,
                 n_components_num: int = None,
                 n_components_cat: int = None
                 ) -> None:
        """
       Initializes the PreProcessData instance.
       Args:
           numerical_cols (list[str]): List of numerical column names.
           categorical_cols (list[str]): List of categorical column names.
           apply_pca: Boolean flag to apply PCA to numerical columns.
           apply_svd: Boolean flag to apply TruncatedSVD to categorical columns.
           n_components_num: Number of components for PCA on numerical data (optional).
           n_components_cat: Number of components for TruncatedSVD on categorical data (optional)
       """
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.apply_pca = apply_pca
        self.apply_svd = apply_svd
        self.n_components_num = n_components_num
        self.n_components_cat = n_components_cat
        self.pipeline = self.create_pipeline()

    def create_pipeline(self) -> ColumnTransformer:
        """
        Creates a preprocessing pipeline for numerical and categorical features.
        Returns:
            ColumnTransformer: The combined preprocessing pipeline.
        """
        numerical_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
            ('pca',
             PCA(n_components=self.n_components_num) if self.apply_pca and self.n_components_num else 'passthrough')
            # Optional PCA
        ])

        categorical_pipeline = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ('svd', TruncatedSVD(
                n_components=self.n_components_cat) if self.apply_svd and self.n_components_cat else 'passthrough')
            # Optional SVD
        ])

        # combine both pipelines using Column Transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, self.numerical_cols),
                ('cat', categorical_pipeline, self.categorical_cols)
            ],
            remainder='passthrough'  # To include any remaining columns
        )

        return preprocessor

    def fit(self, X: Union[pd.DataFrame, np.array], y: [pd.Series, np.array] = None) -> 'PreProcessData':
        """
        Fits the preprocessing pipeline to the data.
        Args:
            X (pd.DataFrame): The input data to fit the pipeline.
            y (pd.Series, optional): Target variable (not used in this method).
        Returns:
            PreProcessData: Returns self.
        """
        self.pipeline.fit(X)
        return self

    def transform(self, X: Union[pd.DataFrame, np.array]):
        """
        Transforms the input data using the fitted preprocessing pipeline.
        Args:
            X (pd.DataFrame): The input data to transform.
        Returns:
            pd.DataFrame: The transformed data.
        Raises:
            NotFittedError: If the pipeline has not been fitted yet.
        """
        return self.pipeline.transform(X)

    def fit_transform(self, X: Union[pd.DataFrame, np.array], y: Union[pd.Series, np.array] = None) -> Union[
        pd.Series, np.array]:
        """
        Fits the preprocessing pipeline to the data and then transforms it.

        Args:
            X (pd.DataFrame): The input data to fit and transform.
            y (pd.Series, optional): Target variable (not used in this method).

        Returns:
            pd.DataFrame: The transformed data.
        """
        return self.pipeline.fit_transform(X)

    def get_feature_names_out(self) -> list[str]:
        """
        Retrieves the feature names after transformation, including one-hot encoded column names.

        Returns:
            List of feature names corresponding to the transformed dataset.
        """
        # Get numerical column names (they remain unchanged)
        numerical_feature_names = self.numerical_cols

        # Get categorical feature names after one-hot encoding
        categorical_feature_names = self.pipeline.named_transformers_['cat'] \
            .named_steps['onehot'].get_feature_names_out(self.categorical_cols)

        # Combine numerical and categorical feature names
        feature_names = list(numerical_feature_names) + list(categorical_feature_names)

        return feature_names
