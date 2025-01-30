import shap
import joblib
import numpy as np
import pandas as pd
from skopt.space import Categorical, Integer, Real
from sklearn.model_selection import train_test_split
from user_profile_segmentation.data.custom import (
    read_csv_from_minio,
    preprocess_training_set,
    preprocess_test_set,
    upload_data_to_minio,
)


df = read_csv_from_minio(bucket_name="data", object_name="users_data.csv")
df.drop(columns=["user_id", "cluster"], inplace=True)
X_train, X_test = train_test_split(df, test_size=0.2)

sparse_features = ["data_quality"]

skewed_features = [
    "loan_score",
    "device_rating",
    "ltv_rate",
    "bureau_score",
    "total_tenure",
    "months_active",
    "usage_score",
    "airtime_score",
]

X_train_processed, preprocessor = preprocess_training_set(
    X=X_train,
    sparse_features=sparse_features,
    skewed_features=skewed_features,
)

X_test_processed = preprocess_test_set(
    X=X_test,
    preprocessor=preprocessor,
    sparse_features=sparse_features,
    skewed_features=skewed_features,
)

upload_data_to_minio(
    dataframe=X_train_processed,
    bucket_name="processed",
    object_name="X_train_processed.pkl",
)

upload_data_to_minio(
    dataframe=X_test_processed,
    bucket_name="processed",
    object_name="X_test_processed.pkl",
)
