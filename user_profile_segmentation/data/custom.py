import os
import io
import boto3
import pickle
import logging
import tempfile
import mlflow.tracking
import numpy as np
import pandas as pd
import urllib.parse
import mlflow
import mlflow.sklearn
import mlflow.tracking
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from botocore.exceptions import BotoCoreError, ClientError
from minio import Minio
from minio.error import MinioException, S3Error
from datetime import datetime
from typing import List, Dict, Any, Tuple, Union, Optional
from matplotlib import pyplot as plt
from skopt import BayesSearchCV
from scipy.spatial.distance import cdist
from itertools import combinations
from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin, clone
from sklearn.preprocessing import PowerTransformer, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import BaseCrossValidator, GridSearchCV
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import pairwise_distances
from sklearn.metrics import (
    calinski_harabasz_score,
    silhouette_score,
    davies_bouldin_score,
)
from user_profile_segmentation.config.config import (
    MINIO_URL,
    MINIO_ROOT_USER,
    MINIO_ROOT_PASSWORD,
    MLFLOW_TRACKING_URI,
)


class Transformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer for densifying sparse features and stabilizing skewed features.

    Parameters
    ----------
    transformation : str, default=None
        The transformation to apply:
        - "add_constant": Add a small constant to all values (for sparse features).
        - "box-cox": Apply Box-Cox transformation to stabilize skewness (for skewed features).
    constant : float, default=0.001
        The constant to add for "add_constant" transformation.

    Attributes
    ----------
    transformation : str
        The transformation to apply.
    constant : float
        The constant to add for "add_constant" transformation.
    box_cox_transformer : PowerTransformer
        The Box-Cox transformer for stabilizing skewed features.

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data.
    transform(X)
        Transform the input data.
    """

    def __init__(self, transformation=None, constant=0.001):
        self.transformation = transformation
        self.constant = constant
        self.box_cox_transformer = PowerTransformer(method="box-cox", standardize=True)

    def fit(self, X, y=None):
        if self.transformation == "box-cox":
            X = np.maximum(
                X, 1e-9
            )  # Replace zeros with a very small positive value to affirm positivity
            self.box_cox_transformer.fit(X)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        if self.transformation == "add_constant":
            return X + self.constant  # Densify sparse features
        elif self.transformation == "box-cox":
            X = np.maximum(X, 1e-9)
            return self.box_cox_transformer.transform(X)
        else:
            return X


class NoCV(BaseCrossValidator):
    """
    Custom CV splitter that returns the entire dataset as training and validation.
    """

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        yield indices, indices


def preprocess_training_set(
    X: pd.DataFrame,
    sparse_features: list,
    skewed_features: list,
) -> pd.DataFrame:
    """
    Preprocesses the input DataFrame for training by:
    - Adding a small constant to sparse features.
    - Applying Box-Cox transformation to stabilize skewed features.
    - Scaling all features in the DataFrame.

    Parameters
    ----------
    X : pd.DataFrame
        The input DataFrame.
    sparse_features : list
        List of sparse feature names to densify.
    skewed_features : list
        List of skewed feature names to stabilize.

    Returns
    -------
    Tuple[pd.DataFrame, Pipeline]
        - A transformed DataFrame with all features scaled.
        - A fitted pipeline for transforming the test set.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "densify_sparse",
                Transformer(transformation="add_constant"),
                sparse_features,
            ),
            (
                "stabilize_skewed",
                Transformer(transformation="box-cox"),
                skewed_features,
            ),
        ],
        remainder="passthrough",  # Keep all other features untouched
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("scaling", RobustScaler()),  # Scales all features, including passthrough
        ]
    )

    processed_array = pipeline.fit_transform(X)

    transformed_column_names = [
        f"{feature}_densified" for feature in sparse_features
    ] + [f"{feature}_stabilized" for feature in skewed_features]

    untouched_features = [
        col for col in X.columns if col not in sparse_features + skewed_features
    ]
    transformed_column_names += untouched_features

    processed_df = pd.DataFrame(processed_array, columns=transformed_column_names)
    return processed_df, pipeline


def preprocess_test_set(
    X: pd.DataFrame,
    preprocessor: Pipeline,
    sparse_features: list,
    skewed_features: list,
) -> pd.DataFrame:
    """
    Preprocesses the test set using the fitted training pipeline.

    Parameters
    ----------
    X : pd.DataFrame
        The input test DataFrame.
    preprocessor : Pipeline
        A fitted preprocessing pipeline.
    sparse_features : list
        List of sparse feature names to densify.
    skewed_features : list
        List of skewed feature names to stabilize.

    Returns
    -------
    pd.DataFrame
        Transformed test set.
    """

    processed_array = preprocessor.transform(X)

    transformed_column_names = [
        f"{feature}_densified" for feature in sparse_features
    ] + [f"{feature}_stabilized" for feature in skewed_features]

    untouched_features = [
        col for col in X.columns if col not in sparse_features + skewed_features
    ]

    transformed_column_names += untouched_features
    processed_df = pd.DataFrame(processed_array, columns=transformed_column_names)
    return processed_df


def capitalize_and_replace(
    s: str,
    delimiter: str,
) -> str:
    words = s.split(delimiter)
    capitalized_words = [word.capitalize() for word in words]
    return " ".join(capitalized_words)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file into a Pandas DataFrame.

    Parameters
    ----------
    file_path : str
        The path to the CSV file that contains the data to be loaded.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame containing the data from the specified CSV file.
    """

    raw_data = pd.read_csv(file_path)
    return raw_data


def read_csv_from_minio(
    bucket_name: str,
    object_name: str,
) -> pd.DataFrame:
    """
    Reads a CSV file from MinIO and loads it into a pandas DataFrame.

    Parameters
    ----------
    bucket_name : str
        The name of the MinIO bucket where the CSV file is stored.
    object_name : str
        The name of the CSV file (object) in the MinIO bucket.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the data from the CSV file.

    Raises
    -------
    Exception
        For any errors that may occur while reading the CSV file.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        s3_client = boto3.client(
            "s3",
            endpoint_url=MINIO_URL,
            aws_access_key_id=MINIO_ROOT_USER,
            aws_secret_access_key=MINIO_ROOT_PASSWORD,
        )

        response = s3_client.get_object(Bucket=bucket_name, Key=object_name)
        df = pd.read_csv(io.BytesIO(response["Body"].read()))

        logger.info(
            f"Successfully loaded CSV data from '{object_name}' in bucket '{bucket_name}'."
        )
        return df
    except Exception as e:
        logger.error(f"Error reading CSV file '{object_name}': {e}")
        return None


def upload_data_to_minio(
    dataframe: pd.DataFrame,
    bucket_name: str,
    object_name: str,
) -> None:
    """
    Uploads a pandas DataFrame to MinIO as a pickled object.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The pandas DataFrame to be uploaded to MinIO.
    bucket_name : str
        The name of the MinIO bucket where the pickled dataframe will be stored.
    object_name : str
        The name of the pickled dataframe (object) in the MinIO bucket.

    Returns
    -------
    None
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        s3_client = boto3.client(
            "s3",
            endpoint_url=MINIO_URL,
            aws_access_key_id=MINIO_ROOT_USER,
            aws_secret_access_key=MINIO_ROOT_PASSWORD,
        )

        with io.BytesIO() as buffer:
            pickle.dump(dataframe, buffer)
            buffer.seek(0)

            s3_client.upload_fileobj(
                Fileobj=buffer,
                Bucket=bucket_name,
                Key=object_name,
                ExtraArgs={"ContentType": "application/octet-stream"},
            )

            logger.info(
                f"Successfully uploaded DataFrame to MinIO bucket '{bucket_name}' as '{object_name}'"
            )
    except Exception as e:
        logger.error(f"An error occurred while uploading data to MinIO: {e}")


def load_data_from_minio(
    bucket_name: str,
    object_name: str,
) -> pd.DataFrame:
    """
    Load a pickled pandas DataFrame from MinIO.

    Parameters
    ----------
    bucket_name : str
        The name of the MinIO bucket where the pickled file is stored.
    object_name : str
        The name of the pickled object (file) in the MinIO bucket.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame deserialized from the pickled file.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        s3_client = boto3.client(
            "s3",
            endpoint_url=MINIO_URL,
            aws_access_key_id=MINIO_ROOT_USER,
            aws_secret_access_key=MINIO_ROOT_PASSWORD,
        )

        response = s3_client.get_object(Bucket=bucket_name, Key=object_name)

        with io.BytesIO(response["Body"].read()) as buffer:
            dataframe = pickle.load(buffer)

        logger.info(
            f"Successfully loaded pickled DataFrame '{object_name}' from bucket '{bucket_name}'."
        )
        return dataframe
    except Exception as e:
        logger.error(f"An error occurred while loading data from MinIO: {e}")
        raise e


def mutual_info(
    df: pd.DataFrame,
    n_bins=10,
) -> pd.DataFrame:
    """
    Calculate the mutual information matrix for a given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame for which to calculate the mutual information matrix.
    n_bins : int, optional
        The number of bins to use for discretizing the data, by default 10.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the mutual information scores between each pair of columns in the input DataFrame.
    """

    mi_matrix = pd.DataFrame(index=df.columns, columns=df.columns)
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
    df_discrete = pd.DataFrame(discretizer.fit_transform(df), columns=df.columns)

    for col1 in df.columns:
        for col2 in df.columns:
            if col1 != col2:
                mi_score = mutual_info_regression(
                    df_discrete[[col1]], df_discrete[col2]
                )
                mi_matrix.loc[col1, col2] = mi_score[0]
            else:
                mi_matrix.loc[col1, col2] = 0

    return mi_matrix


def trustworthiness(
    X: np.ndarray,
    Y: np.ndarray,
    n_neighbors: int = 5,
) -> float:
    """
    Calculate the trustworthiness score between two datasets.

    Parameters
    ----------
    X : np.ndarray
        The original dataset.
    Y : np.ndarray
        The reduced dataset.
    n_neighbors : int, optional

    Returns
    -------
    float
        The trustworthiness score between the two datasets.
    """

    n_samples = X.shape[0]

    dist_X = pairwise_distances(X)
    dist_Y = pairwise_distances(Y)

    rank_X = np.argsort(np.argsort(dist_X, axis=1), axis=1)
    rank_Y = np.argsort(np.argsort(dist_Y, axis=1), axis=1)

    trusted = 0

    for i in range(n_samples):
        original_neighbors = np.where(rank_X[i] <= n_neighbors)[0]
        reduced_neighbors = np.where(rank_Y[i] <= n_neighbors)[0]

        mismatched = set(reduced_neighbors) - set(original_neighbors)
        trusted += sum(rank_X[i][j] - n_neighbors for j in mismatched)

    normalizer = n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1)
    trustworthiness_score = 1 - (2 * trusted) / normalizer
    return trustworthiness_score


def recursive_cluster_feature_elimination(
    estimator: ClusterMixin,
    X_train: pd.DataFrame,
    features: List[str],
    group_size: int = 1,
    patience: int = 4,
) -> Tuple[Optional[ClusterMixin], List[str], float]:
    """
    Perform Recursive Cluster Feature Elimination (RCFE) with group evaluation
    to find the optimal set of features based on the silhouette score.

    Parameters
    ----------
    estimator : ClusterMixin
        The clustering model used for feature evaluation.
    X_train : pd.DataFrame
        The training dataset with all features.
    features : List[str]
        The initial list of feature names to consider.
    group_size : int, optional
        The number of features to remove as a group during each iteration.
    patience : int, optional
        Number of iterations to wait for improvement before stopping early.

    Returns
    -------
    Tuple[Optional[ClusterMixin], List[str], float]
        - The clustering model trained on the optimal set of features (or None).
        - The list of optimal features.
        - The highest silhouette score achieved (or -1 if failed).
    """
    n_features = len(features)
    if n_features == 0:
        raise ValueError("Feature list is empty. At least one feature is required.")

    current_features = features.copy()
    best_silhouette_score = -1
    optimal_features = current_features.copy()
    optimal_model = None
    no_improvement_count = 0

    print(f"Starting RCFE with group size {group_size}, patience {patience}.")

    while len(current_features) > group_size:
        scores = []
        models = []
        feature_groups = list(combinations(current_features, group_size))

        print(f"Evaluating {len(feature_groups)} groups of size {group_size}...")

        # Evaluate removing each group of features
        for group in feature_groups:
            remaining_features = [f for f in current_features if f not in group]

            if len(remaining_features) == 0:
                continue  # Skip if no features remain

            X_train_subset = X_train[remaining_features]
            try:
                model = clone(estimator)
                labels = model.fit_predict(X_train_subset)

                # Exclude single-cluster or noise-only cases
                unique_labels = set(labels)
                if len(unique_labels) > 1:
                    # For algorithms like DBSCAN, exclude noise points
                    mask = labels != -1 if -1 in labels else np.full(len(labels), True)
                    if np.sum(mask) > 1:
                        score = silhouette_score(X_train_subset[mask], labels[mask])
                    else:
                        score = -1  # Not enough valid points
                else:
                    score = -1  # Only one cluster or noise
            except Exception as e:
                print(f"Error evaluating features {remaining_features}: {e}")
                score = -1

            scores.append(score)
            models.append((remaining_features, model))

        if not scores or max(scores) == -1:
            print("No valid scores found in this iteration. Stopping...")
            break  # No valid scores, stop early

        # Find the best group removal
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        best_remaining_features, best_model = models[best_idx]

        print(f"Best group removal leads to silhouette score: {best_score:.4f}")

        # Check for improvement
        if best_score > best_silhouette_score:
            best_silhouette_score = best_score
            optimal_features = best_remaining_features
            optimal_model = best_model
            no_improvement_count = 0  # Reset counter
            print(f"Improved score: {best_silhouette_score:.4f}. Continuing...")
        else:
            no_improvement_count += 1
            print(f"No improvement for {no_improvement_count} iteration(s).")

        # Early stopping condition
        if no_improvement_count >= patience:
            print(f"Early stopping after {patience} iterations with no improvement.")
            break

        current_features = optimal_features

    if optimal_model is None:
        print("RCFE failed to find a valid model or features.")
    else:
        print(f"RCFE completed. Best score: {best_silhouette_score:.4f}")

    return (
        optimal_model,
        optimal_features,
        best_silhouette_score,
    )


def calinski_harabasz_scorer(
    estimator: ClusterMixin,
    X: pd.DataFrame,
) -> float:
    """
    Computes the Calinski-Harabasz score for the given estimator's clustering results, ensuring there are at least 2 valid clusters.

    Parameters:
    ----------
    estimator : ClusterMixin
        The clustering model to evaluate.
    X : pd.DataFrame
        The dataset on which to perform clustering and scoring.

    Returns:
    -------
    float
        The Calinski-Harabasz score if valid, or -1 otherwise.
    """

    labels = estimator.fit_predict(X)
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Exclude noise label (-1) for DBSCAN

    if len(unique_labels) < 2:
        # Return an invalid score when fewer than 2 clusters are present
        print(f"Invalid clustering: Only {len(unique_labels)} valid cluster(s) found.")
        return -1.0

    return calinski_harabasz_score(X, labels)


def bayesian_hyperparameter_tuning(
    estimator: ClusterMixin,
    search_spaces: Dict[str, Any],
    X_train: pd.DataFrame,
    optimal_features: List[str],
) -> ClusterMixin:
    """
    Performs hyperparameter tuning for a given clustering model using BayesSearchCV optimizing for the Calinski-Harabasz score.

    Parameters
    ----------
    estimator : ClusterMixin
        The clustering model to tune (e.g., KMeans, DBSCAN).
    search_spaces : Dict[str, Any]
        A dictionary defining the search space for each hyperparameter.
    X_train : pd.DataFrame
        The training dataset containing all features.
    optimal_features : List[str]
        The list of optimal feature names for use in model training.

    Returns
    -------
    ClusterMixin
        The best clustering model found by BayesSearchCV after hyperparameter tuning.
    """

    if estimator is None:
        raise ValueError("The estimator parameter for BayesSearchCV cannot be None.")
    if not isinstance(estimator, ClusterMixin):
        raise TypeError(f"Expected a clustering model, got {type(estimator)} instead.")

    bayes_search = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_spaces,
        scoring=calinski_harabasz_scorer,
        cv=NoCV(),
        refit=True,
        random_state=42,
        n_iter=50,
        n_jobs=6,
        verbose=1,
    )

    X_train_optimal = X_train[optimal_features]
    bayes_search.fit(X_train_optimal)
    tuned_model = bayes_search.best_estimator_
    return tuned_model


def train_kmeans_clusterer(
    X_train: pd.DataFrame,
    search_spaces: Dict[str, Any],
    features: List[str],
) -> Dict[str, Any]:
    """
    Trains a KMeans clustering model, performs recursive feature elimination to find the optimal set of features, and tunes the model using Bayesian search for hyperparameter optimization.

    Parameters:
    ----------
    X_train : pd.DataFrame
        The training dataset containing all features.
    search_spaces : Dict[str, Any]
        A dictionary defining the search space for each hyperparameter.
    features : List[str]
        The list of feature names to consider for recursive feature elimination.

    Returns:
    -------
    Dict[str, Any]
        A dictionary containing the tuned KMeans model, optimal features, and the best silhouette score.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_name = (
        f"KMeans Clusterer Experiment Run - "
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    registered_model_name = "User Profile Segmentation KMeans Clusterer"

    with mlflow.start_run(run_name=model_name) as run:
        kmeans_model = KMeans(random_state=42)

        (
            kmeans_fitted_model,
            kmeans_optimal_features,
            kmeans_silhouette_score,
        ) = recursive_cluster_feature_elimination(
            kmeans_model,
            X_train,
            features,
        )

        kmeans_tuned_model = bayesian_hyperparameter_tuning(
            kmeans_fitted_model,
            search_spaces,
            X_train,
            kmeans_optimal_features,
        )

        artifact_path = f"{registered_model_name.lower().replace("(", "").replace(")", "").replace(" ", "_")}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            mlflow.sklearn.log_model(
                sk_model=kmeans_tuned_model,
                artifact_path=artifact_path,
            )
        except mlflow.exceptions.MlflowException as e:
            print(f"MLflow error logging model: {e}")

        try:
            mlflow.set_tag("artifact_path", artifact_path)
        except Exception as e:
            print(f"Error logging model tag: {e}")

        try:
            mlflow.log_metrics({"Silhouette Score": kmeans_silhouette_score})
        except Exception as e:
            print(f"Error logging metric: {e}")

        model_uri = f"runs:/{run.info.run_id}/{urllib.parse.quote(model_name)}"
        model_description = (
            f"KMeans Clusterer for User Profile Segmentation with "
            f"optimal feature(s): {kmeans_optimal_features} after RCFE."
        )

        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )

            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=registered_model_name,
                version=registered_model.version,
                description=model_description,
            )

            client.set_model_version_tag(
                name=registered_model_name,
                version=registered_model.version,
                key="Trained at",
                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            client.set_registered_model_alias(
                name=registered_model_name,
                version=registered_model.version,
                alias="Staging",
            )
        except MlflowException as e:
            print(f"Model registration for {model_name} failed: {e}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
            pickle.dump(kmeans_optimal_features, temp_file)
            temp_file_path = temp_file.name

        mlflow.log_artifact(temp_file_path, artifact_path="optimal_features")
        os.remove(temp_file_path)
        logger.info("Optimal features logged as artifact to MLflow.")
    except Exception as e:
        logger.error(f"Error logging optimal features to MLflow: {e}")

    return {
        "Model": kmeans_tuned_model,
        "Optimal Features": kmeans_optimal_features,
        "Silhouette Score": kmeans_silhouette_score,
    }


def train_spectral_clusterer(
    X_train: pd.DataFrame,
    search_spaces: Dict[str, Any],
    features: List[str],
) -> Dict[str, Any]:
    """
    Trains a Spectral clustering model, performs recursive feature elimination to find the optimal set of features, and tunes the model using Bayesian search for hyperparameter optimization.

    Parameters:
    ----------
    X_train : pd.DataFrame
        The training dataset containing all features.
    search_spaces : Dict[str, Any]
        A dictionary defining the search space for each hyperparameter.
    features : List[str]
        The list of feature names to consider for recursive feature elimination.

    Returns:
    -------
    Dict[str, Any]
        A dictionary containing the tuned Spectral model, optimal features, and the best silhouette score.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_name = (
        f"Spectral Clusterer Experiment Run - "
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    registered_model_name = "User Profile Segmentation Spectral Clusterer"

    with mlflow.start_run(run_name=model_name) as run:
        spectral_model = SpectralClustering(
            random_state=42,
        )

        (
            spectral_fitted_model,
            spectral_optimal_features,
            spectral_silhouette_score,
        ) = recursive_cluster_feature_elimination(
            spectral_model,
            X_train,
            features,
        )

        spectral_tuned_model = bayesian_hyperparameter_tuning(
            spectral_fitted_model,
            search_spaces,
            X_train,
            spectral_optimal_features,
        )

        artifact_path = f"{registered_model_name.lower().replace('(', '').replace(')', '').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            mlflow.sklearn.log_model(
                sk_model=spectral_tuned_model,
                artifact_path=artifact_path,
            )
        except mlflow.exceptions.MlflowException as e:
            print(f"MLflow error logging model: {e}")

        try:
            mlflow.set_tag("artifact_path", artifact_path)
        except Exception as e:
            print(f"Error logging model tag: {e}")

        try:
            mlflow.log_metrics({"Silhouette Score": spectral_silhouette_score})
        except Exception as e:
            print(f"Error logging metric: {e}")

        model_uri = f"runs:/{run.info.run_id}/{urllib.parse.quote(model_name)}"
        model_description = (
            f"Spectral Clusterer for User Profile Segmentation with "
            f"optimal feature(s): {spectral_silhouette_score} after RCFE."
        )

        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )

            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=registered_model_name,
                version=registered_model.version,
                description=model_description,
            )

            client.set_model_version_tag(
                name=registered_model_name,
                version=registered_model.version,
                key="Trained at",
                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            client.set_registered_model_alias(
                name=registered_model_name,
                version=registered_model.version,
                alias="Staging",
            )
        except MlflowException as e:
            print(f"Model registration for {model_name} failed: {e}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
            pickle.dump(spectral_optimal_features, temp_file)
            temp_file_path = temp_file.name

        mlflow.log_artifact(temp_file_path, artifact_path="optimal_features")
        os.remove(temp_file_path)
        logger.info("Optimal features logged as artifact to MLflow.")
    except Exception as e:
        logger.error(f"Error logging optimal features to MLflow: {e}")

    return {
        "Model": spectral_tuned_model,
        "Optimal Features": spectral_optimal_features,
        "Silhouette Score": spectral_silhouette_score,
    }


def train_dbscan_clusterer(
    X_train: pd.DataFrame,
    search_spaces: Dict[str, Any],
    features: List[str],
) -> Dict[str, Any]:
    """
    Trains a DBSCAN clustering model, performs recursive feature elimination to find the optimal set of features, and tunes the model using Bayesian search for hyperparameter optimization.

    Parameters:
    ----------
    X_train : pd.DataFrame
        The training dataset containing all features.
    search_spaces : Dict[str, Any]
        A dictionary defining the search space for each hyperparameter.
    features : List[str]
        The list of feature names to consider for recursive feature elimination.

    Returns:
    -------
    Dict[str, Any]
        A dictionary containing the tuned DBSCAN model, optimal features, silhouette score, cluster labels, and core sample indices.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_name = (
        f"DBSCAN Clusterer Experiment Run - "
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    registered_model_name = "User Profile Segmentation DBSCAN Clusterer"

    with mlflow.start_run(run_name=model_name) as run:
        dbscan_model = DBSCAN()

        (
            dbscan_fitted_model,
            dbscan_optimal_features,
            dbscan_silhouette_score,
        ) = recursive_cluster_feature_elimination(
            dbscan_model,
            X_train,
            features,
        )

        if dbscan_fitted_model is None:
            raise ValueError(
                "The model returned by recursive_cluster_feature_elimination is None."
            )

        if not isinstance(dbscan_fitted_model, DBSCAN):
            raise TypeError(
                f"Expected DBSCAN, got {type(dbscan_fitted_model)} instead."
            )

        if not dbscan_optimal_features:
            raise ValueError(
                "No optimal features were selected by recursive feature elimination."
            )

        dbscan_tuned_model = bayesian_hyperparameter_tuning(
            dbscan_fitted_model,
            search_spaces,
            X_train,
            dbscan_optimal_features,
        )

        cluster_labels = dbscan_tuned_model.labels_
        core_sample_indices = dbscan_tuned_model.core_sample_indices_

        artifact_path = f"{registered_model_name.lower().replace("(", "").replace(")", "").replace(" ", "_")}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            mlflow.sklearn.log_model(
                sk_model=dbscan_tuned_model,
                artifact_path=artifact_path,
            )
        except mlflow.exceptions.MlflowException as e:
            print(f"MLflow error logging model: {e}")

        try:
            mlflow.set_tag("artifact_path", artifact_path)
        except Exception as e:
            print(f"Error logging model tag: {e}")

        try:
            mlflow.log_metrics({"Silhouette Score": dbscan_silhouette_score})
        except Exception as e:
            print(f"Error logging metric: {e}")

        model_uri = f"runs:/{run.info.run_id}/{urllib.parse.quote(model_name)}"
        model_description = (
            f"DBSCAN Clusterer for User Profile Segmentation with "
            f"optimal features: {dbscan_optimal_features} after RCFE."
        )

        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )

            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=registered_model_name,
                version=registered_model.version,
                description=model_description,
            )

            client.set_model_version_tag(
                name=registered_model_name,
                version=registered_model.version,
                key="Trained at",
                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            client.set_registered_model_alias(
                name=registered_model_name,
                version=registered_model.version,
                alias="Staging",
            )
        except MlflowException as e:
            print(f"Model registration for {model_name} failed: {e}")

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
                pickle.dump(dbscan_optimal_features, temp_file)
                temp_file_path = temp_file.name

            mlflow.log_artifact(temp_file_path, artifact_path="optimal_features")
            os.remove(temp_file_path)
            logger.info("Optimal features logged as artifact to MLflow.")
        except Exception as e:
            logger.error(f"Error logging optimal features to MLflow: {e}")

    return {
        "Model": dbscan_tuned_model,
        "Optimal Features": dbscan_optimal_features,
        "Silhouette Score": dbscan_silhouette_score,
        "Cluster Labels": cluster_labels,
        "Core Sample Indices": core_sample_indices,
    }


def train_hierarchical_clusterer(
    X_train: pd.DataFrame,
    search_spaces: Dict[str, Any],
    features: List[str],
) -> Dict[str, Any]:
    """
    Trains an Agglomerative Hierarchical clustering model, performs recursive feature elimination to find the optimal set of features, and tunes the model using Bayesian search for hyperparameter optimization.

    Parameters:
    ----------
    X_train : pd.DataFrame
        The training dataset containing all features.
    search_spaces : Dict[str, Any]
        A dictionary defining the search space for each hyperparameter.
    features : List[str]
        The list of feature names to consider for recursive feature elimination.

    Returns:
    -------
    Dict[str, Any]
        A dictionary containing the tuned Agglomerative Hierarchical model, optimal features, silhouette score, cluster labels, and cluster centroids
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_name = (
        f"Agglomerative Hierarchical Clusterer Experiment Run - "
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    registered_model_name = (
        "User Profile Segmentation Agglomerative Hierarchical Clusterer"
    )

    with mlflow.start_run(run_name=model_name) as run:
        hierarchical_model = AgglomerativeClustering()

        (
            hierarchical_fitted_model,
            hierarchical_optimal_features,
            hierarchical_silhouette_score,
        ) = recursive_cluster_feature_elimination(hierarchical_model, X_train, features)

        hierarchical_tuned_model = bayesian_hyperparameter_tuning(
            hierarchical_fitted_model,
            search_spaces,
            X_train,
            hierarchical_optimal_features,
        )

        # Extract cluster labels for future predictions
        cluster_labels = hierarchical_tuned_model.labels_

        # Compute cluster centroids (mean of features per cluster)
        cluster_centroids = (
            X_train.groupby(cluster_labels).mean().to_numpy()
            if hasattr(hierarchical_tuned_model, "labels_")
            else None
        )

        artifact_path = f"{registered_model_name.lower().replace("(", "").replace(")", "").replace(" ", "_")}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            mlflow.sklearn.log_model(
                sk_model=hierarchical_tuned_model,
                artifact_path=artifact_path,
            )
        except mlflow.exceptions.MlflowException as e:
            print(f"MLflow error logging model: {e}")

        try:
            mlflow.set_tag("artifact_path", artifact_path)
        except Exception as e:
            print(f"Error logging model tag: {e}")

        try:
            mlflow.log_metrics({"Silhouette Score": hierarchical_silhouette_score})
        except Exception as e:
            print(f"Error logging metric: {e}")

        model_uri = f"runs:/{run.info.run_id}/{urllib.parse.quote(model_name)}"
        model_description = (
            f"Agglomerative Hierarchical Clusterer for User Profile Segmentation with "
            f"optimal features: {hierarchical_optimal_features} after RCFE."
        )

        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )

            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=registered_model_name,
                version=registered_model.version,
                description=model_description,
            )

            client.set_model_version_tag(
                name=registered_model_name,
                version=registered_model.version,
                key="Trained at",
                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            client.set_registered_model_alias(
                name=registered_model_name,
                version=registered_model.version,
                alias="Staging",
            )
        except MlflowException as e:
            print(f"Model registration for {model_name} failed: {e}")

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
                pickle.dump(hierarchical_optimal_features, temp_file)
                temp_file_path = temp_file.name

            mlflow.log_artifact(temp_file_path, artifact_path="optimal_features")
            os.remove(temp_file_path)
            logger.info("Optimal features logged as artifact to MLflow.")
        except Exception as e:
            logger.error(f"Error logging optimal features to MLflow: {e}")

        try:
            if hasattr(hierarchical_tuned_model, "labels_"):
                cluster_centroids = (
                    X_train.groupby(hierarchical_tuned_model.labels_).mean().to_numpy()
                )
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pkl"
                ) as temp_file:
                    pickle.dump(cluster_centroids, temp_file)
                    temp_file_path = temp_file.name

                mlflow.log_artifact(temp_file_path, artifact_path="cluster_centroids")
                os.remove(temp_file_path)
                logger.info("Cluster centroids logged as artifact to MLflow.")
        except Exception as e:
            logger.error(f"Error logging cluster centroids to MLflow: {e}")

    return {
        "Model": hierarchical_tuned_model,
        "Optimal Features": hierarchical_optimal_features,
        "Silhouette Score": hierarchical_silhouette_score,
        "Cluster Labels": cluster_labels,
        "Cluster Centroids": cluster_centroids,
    }


def predict_spectral(
    spectral_model: SpectralClustering,
    X_train: pd.DataFrame,
    cluster_labels: np.ndarray,
    X_new: pd.DataFrame,
    k: int = 5,
) -> np.ndarray:
    """
    Predicts cluster assignments for new data points using a KNN classifier
    trained on Spectral Clustering results.

    Parameters
    ----------
    spectral_model : SpectralClustering
        The trained Spectral Clustering model (not directly used in predictions).
    X_train : pd.DataFrame
        The training dataset.
    cluster_labels : np.ndarray
        The cluster labels assigned to training data.
    X_new : pd.DataFrame
        The new data points to classify.
    k : int, optional
        Number of neighbors for the KNN model (default is 5).

    Returns
    -------
    np.ndarray
        Cluster labels for new data points.
    """

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, cluster_labels)

    new_labels = knn.predict(X_new)
    return new_labels


def predict_dbscan(
    dbscan_model: DBSCAN,
    X_train: pd.DataFrame,
    cluster_labels: np.ndarray,
    core_sample_indices: np.ndarray,
    X_new: pd.DataFrame,
) -> np.ndarray:
    """
    Predicts cluster assignments for new data points using a KNN classifier
    trained on DBSCAN core samples.

    Parameters
    ----------
    dbscan_model : DBSCAN
        The trained DBSCAN model.
    X_train : pd.DataFrame
        The training dataset.
    cluster_labels : np.ndarray
        The cluster labels assigned to training data.
    core_sample_indices : np.ndarray
        The indices of the core samples identified by DBSCAN.
    X_new : pd.DataFrame
        The new data points to classify.

    Returns
    -------
    np.ndarray
        Cluster labels for new data points (-1 for noise).
    """

    core_points = X_train.iloc[core_sample_indices]
    core_labels = cluster_labels[core_sample_indices]

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(core_points, core_labels)

    new_labels = knn.predict(X_new)
    distances, _ = knn.kneighbors(X_new)
    noise_mask = distances[:, 0] > dbscan_model.eps
    new_labels[noise_mask] = -1
    return new_labels


def predict_hierarchical(
    hierarchical_model: AgglomerativeClustering,
    X_train: pd.DataFrame,
    cluster_labels: np.ndarray,
    cluster_centroids: Optional[np.ndarray],
    X_new: pd.DataFrame,
) -> np.ndarray:
    """
    Predicts cluster assignments for new data points using nearest centroid matching.

    Parameters
    ----------
    hierarchical_model : AgglomerativeClustering
        The trained hierarchical clustering model.
    X_train : pd.DataFrame
        The training dataset.
    cluster_labels : np.ndarray
        The cluster labels assigned to training data.
    cluster_centroids : Optional[np.ndarray]
        The centroids of each cluster computed after training. If None, centroids will be computed.
    X_new : pd.DataFrame
        The new data points to classify.

    Returns
    -------
    np.ndarray
        Cluster labels for new data points.
    """

    if cluster_centroids is None:
        cluster_centroids = np.array(
            [
                X_train[cluster_labels == i].mean(axis=0)
                for i in np.unique(cluster_labels)
            ]
        )

    new_labels = np.argmin(cdist(X_new, cluster_centroids), axis=1)
    return new_labels


def load_model_from_mlflow(
    bucket_name: str,
    object_name: str,
    run_id: str,
    model_name: str,
    feature_name: str,
) -> Tuple[ClusterMixin, Any]:
    """
    Loads a model and its corresponding features from MLflow artifacts stored in MinIO.

    Parameters
    ----------
    bucket_name : str
        The name of the MinIO bucket where the model and features are stored.
    object_name : str
        The name of the object to load from the bucket.
    run_id : str
        The MLflow run ID where the model and features were logged.
    model_name : str
        The name of the model to load.
    feature_name : str
        The name of the optimal features to load. If None, only the model is loaded.

    Returns
    -------
    Tuple[ClusterMixin, Any]
        The loaded model and features, or None if an error occurs.

    Notes
    -----
    - The model and features are loaded from the specified bucket and object names.
    - If `feature_name` is None, only the model will be loaded.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        s3_client = boto3.client(
            "s3",
            endpoint_url=MINIO_URL,
            aws_access_key_id=MINIO_ROOT_USER,
            aws_secret_access_key=MINIO_ROOT_PASSWORD,
        )

        model_object_key = f"1/{run_id}/artifacts/{model_name}/{object_name}"
        model_response = s3_client.get_object(
            Bucket=bucket_name,
            Key=model_object_key,
        )
        model_data = model_response["Body"].read()
        model = pickle.loads(model_data)

        if feature_name:
            feature_object_key = f"1/{run_id}/artifacts/optimal_features/{feature_name}"
            feature_response = s3_client.get_object(
                Bucket=bucket_name,
                Key=feature_object_key,
            )
            feature_data = feature_response["Body"].read()
            features = pickle.loads(feature_data)
            logger.info(f"Model and features loaded from MinIO successfully.")
            return model, features
        else:
            logger.info(f"Model loaded from MinIO successfully. No features provided.")
            return model
    except Exception as e:
        logger.error(f"Error loading model and features from MinIO: {e}")
        return None, None


def apply_pca(X_optimal):
    """
    Applies Principal Component Analysis (PCA) to reduce the dimensionality of the dataset to 2 components, if the dataset has more than 2 features.

    Parameters
    ----------
    X_optimal : np.ndarray or pd.DataFrame
        The dataset to be transformed. Rows represent samples, and columns represent features.

    Returns
    -------
    np.ndarray
        The transformed dataset with 2 principal components, or the original dataset if it
        has 2 or fewer features.
    """

    if X_optimal.shape[1] > 2:
        pca = PCA(n_components=2)
        return pca.fit_transform(X_optimal)
    else:
        return X_optimal


def apply_tsne(X_optimal):
    """
    Applies t-Distributed Stochastic Neighbor Embedding (t-SNE) to reduce the dimensionality of the dataset to 2 components, if the dataset has more than 2 features.

    Parameters
    ----------
    X_optimal : np.ndarray or pd.DataFrame
        The dataset to be transformed. Rows represent samples, and columns represent features.

    Returns
    -------
    np.ndarray
        The transformed dataset with 2 components using t-SNE, or the original dataset if it
        has 2 or fewer features.

    Notes
    -----
    - t-SNE is typically used for visualization and may not preserve global structure well.
    - Random state is set to 42 for reproducibility.
    """

    if X_optimal.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42)
        return tsne.fit_transform(X_optimal)
    else:
        return X_optimal


def scatter_plot(
    X_reduced: Union[np.ndarray, pd.DataFrame],
    clusters: Union[np.ndarray, List[int]],
    cluster_centers: Optional[np.ndarray] = None,
    title: str = "Scatter Plot",
    xlabel: str = "Index",
    ylabel: str = "Feature Value",
) -> None:
    """
    Generates a scatter plot to visualize clustering results in either 1D or 2D space. The function adapts based on the number of dimensions in the input data (X_reduced).

    Parameters:
    -----------
    X_reduced : numpy.ndarray or pandas.DataFrame
        The reduced input features after dimensionality reduction (e.g., PCA) or selected features.
        Should either be a 1D array (single feature) or a 2D array (two features).
        For 1D data, the points will be plotted against their index.
    clusters : array-like
        The cluster labels for each data point. Used to color the data points according to their cluster.
    cluster_centers : numpy.ndarray, optional
        The coordinates of the cluster centers if available. Applicable for algorithms like KMeans.
        For 1D, cluster centers are plotted on the x-axis index. For 2D, centers are plotted on both axes.
    title : str, optional
        The title of the plot. Default is "Scatter Plot".
    xlabel : str, optional
        Label for the x-axis. Default is "Index" for 1D plots.
    ylabel : str, optional
        Label for the y-axis. Default is "Feature Value" for 1D plots.
    """

    if isinstance(X_reduced, pd.DataFrame):
        X_reduced = X_reduced.values

    if X_reduced.ndim == 2 and X_reduced.shape[1] == 1:
        X_reduced = X_reduced.flatten()

    if X_reduced.ndim == 1:
        plt.scatter(
            range(len(X_reduced)),
            X_reduced,
            c=clusters,
            cmap="viridis",
            marker="o",
            label="Data Points",
        )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # For KMeans if applicable
        if cluster_centers is not None:
            plt.scatter(
                range(len(cluster_centers)),
                cluster_centers,
                s=300,
                c="red",
                marker="x",
                label="Cluster Centers",
            )

    elif X_reduced.ndim == 2 and X_reduced.shape[1] == 2:
        plt.scatter(
            X_reduced[:, 0],
            X_reduced[:, 1],
            c=clusters,
            cmap="viridis",
            marker="o",
            label="Data Points",
        )
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")

        if cluster_centers is not None:
            plt.scatter(
                cluster_centers[:, 0],
                cluster_centers[:, 1],
                s=300,
                c="red",
                marker="x",
                label="Cluster Centers",
            )

    else:
        raise ValueError(
            "The input features must have either one or two dimensions for plotting"
        )

    plt.title(title)
    plt.colorbar(label="Cluster")
    plt.legend()
    plt.show()


def evaluate_clustering_models(
    training_clustering: dict,
    test_clustering: dict,
    X_train_dict: dict,
    X_test_dict: dict,
) -> pd.DataFrame:
    """
    Evaluates the performance of multiple clustering models using Silhouette Score, Davies-Bouldin Score, and Calinski-Harabasz Index, and stores the results in a DataFrame.

    Parameters:
    -----------
    training_clustering (dict)
        Dictionary containing training cluster labels for each model.
    test_clustering (dict)
        Dictionary containing test cluster labels for each model.
    X_train_dict (dict)
        Dictionary containing optimized training datasets for each model.
    X_test_dict (dict)
        Dictionary containing optimized test datasets for each model.

    Returns:
    -----------
    pd.DataFrame
        A DataFrame showing the evaluation metrics for each model.
    """

    results = []

    for model_name in training_clustering.keys():
        train_labels = training_clustering[model_name]
        X_train_optimal = X_train_dict[model_name]

        silhouette_train = silhouette_score(X_train_optimal, train_labels)
        davies_bouldin_train = davies_bouldin_score(X_train_optimal, train_labels)
        calinski_harabasz_train = calinski_harabasz_score(X_train_optimal, train_labels)

        test_labels = test_clustering[model_name]
        X_test_optimal = X_test_dict[model_name]

        silhouette_test = silhouette_score(X_test_optimal, test_labels)
        davies_bouldin_test = davies_bouldin_score(X_test_optimal, test_labels)
        calinski_harabasz_test = calinski_harabasz_score(X_test_optimal, test_labels)

        results.append(
            {
                "Model": model_name,
                "Silhouette Score (Train)": silhouette_train,
                "Silhouette Score (Test)": silhouette_test,
                "Davies-Bouldin Score (Train)": davies_bouldin_train,
                "Davies-Bouldin Score (Test)": davies_bouldin_test,
                "Calinski-Harabasz Score (Train)": calinski_harabasz_train,
                "Calinski-Harabasz Score (Test)": calinski_harabasz_test,
            }
        )

    results_df = pd.DataFrame(results)
    return results_df
