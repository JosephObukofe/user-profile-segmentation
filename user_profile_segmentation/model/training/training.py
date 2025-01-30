import numpy as np
import pandas as pd
import joblib
import mlflow
import logging
from skopt.space import Categorical, Integer, Real
from joblib import Parallel, delayed
from user_profile_segmentation.data.custom import (
    load_data_from_minio,
    train_kmeans_clusterer,
    train_spectral_clusterer,
    train_dbscan_clusterer,
    train_hierarchical_clusterer,
)
from user_profile_segmentation.config.config import (
    MLFLOW_TRACKING_URI,
    TRAINED_KMEANS_FEATURES,
    TRAINED_SPECTRAL_FEATURES,
    TRAINED_DBSCAN_FEATURES,
    TRAINED_HIERARCHICAL_FEATURES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

mlflow.set_experiment("User Profile Segmentation: Model Training")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

X_train = load_data_from_minio(
    bucket_name="processed",
    object_name="X_train_processed.pkl",
)
features = [feature for feature in X_train.columns]

optimal_features = [
    TRAINED_KMEANS_FEATURES,
    TRAINED_SPECTRAL_FEATURES,
    TRAINED_DBSCAN_FEATURES,
    TRAINED_HIERARCHICAL_FEATURES,
]

kmeans_model_search_space = {
    "n_clusters": Integer(5, 6),
    "init": Categorical(["k-means++", "random"]),
    "max_iter": Integer(300, 1000),
}

spectral_model_search_space = {
    "n_clusters": Integer(4, 6),
    "affinity": Categorical(["nearest_neighbors", "rbf"]),
    "gamma": Real(0.5, 1.5),
    "n_neighbors": Integer(8, 15),
}

dbscan_model_search_space = {
    "eps": Real(0.1, 0.2),
    "min_samples": Integer(20, 40),
    "metric": Categorical(["euclidean", "manhattan", "cosine"]),
}

hierarchical_model_search_space = {
    "n_clusters": Integer(5, 6),
    "linkage": Categorical(["ward", "complete", "average", "single"]),
    "metric": Categorical(["euclidean"]),
}


def safe_task(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logging.error(f"Error in {func.__name__}: {e}", exc_info=True)
        return None


def parallel_model_training():
    tasks = [
        delayed(safe_task)(
            train_kmeans_clusterer,
            X_train,
            kmeans_model_search_space,
            features,
        ),
        delayed(safe_task)(
            train_spectral_clusterer,
            X_train,
            spectral_model_search_space,
            features,
        ),
        delayed(safe_task)(
            train_dbscan_clusterer,
            X_train,
            dbscan_model_search_space,
            features,
        ),
        delayed(safe_task)(
            train_hierarchical_clusterer,
            X_train,
            hierarchical_model_search_space,
            features,
        ),
    ]

    trained_models_output = Parallel(n_jobs=4, verbose=10)(tasks)
    for i, model_output in enumerate(trained_models_output):
        if model_output:
            logging.info(f"Model {i + 1} Training Successful")
            logging.info(f"Silhouette Score: {model_output['Silhouette Score']}")
        else:
            logging.warning(f"Model {i + 1} Training Failed")
    return trained_models_output


if __name__ == "__main__":
    trained_models = parallel_model_training()

    for i, model_output in enumerate(trained_models):
        if model_output:
            joblib.dump(model_output["Optimal Features"], optimal_features[i])
