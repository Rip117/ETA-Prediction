import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")


NUMERIC_COLS = ["prev_delay", "trend", "gap_min", "distance_km", "num_stops",
                "travel_min", "hour", "n_run_days", "avg_speed_kmph"]
CATEGORICAL_COLS = ["type_label", "weekday"]
TARGET = "delta"


def count_run_days(value) -> float:
    """'Daily' -> 7, 'Mon,Wed,Fri' -> 3, 'Tue' -> 1"""
    if pd.isna(value):
        return np.nan
    value = str(value)
    if "daily" in value.lower():
        return 7
    return len([d for d in value.split(",") if d.strip()])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["n_run_days"] = df["runs_days"].apply(count_run_days)
    df["avg_speed_kmph"] = df["distance_km"] / (df["travel_min"] / 60)
    df["weekday"] = df["weekday"].astype(str)      # 0..6 are labels, not quantities
    return df


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_preprocessor(self) -> ColumnTransformer:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                     min_frequency=10,
                                     sparse_output=False)),
        ])
        return ColumnTransformer([
            ("num", num_pipeline, NUMERIC_COLS),
            ("cat", cat_pipeline, CATEGORICAL_COLS),
        ])

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            train_df = add_features(pd.read_csv(train_path, dtype={"train_no": str}))
            test_df = add_features(pd.read_csv(test_path, dtype={"train_no": str}))
            logging.info("Read train and test data, features added")

            preprocessor = self.get_preprocessor()

            X_train = train_df[NUMERIC_COLS + CATEGORICAL_COLS]
            X_test = test_df[NUMERIC_COLS + CATEGORICAL_COLS]
            y_train = train_df[TARGET]
            y_test = test_df[TARGET]

            # fit on TRAIN only; the test set must stay unseen
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)
            logging.info(f"Transformed shapes: {X_train_arr.shape}, {X_test_arr.shape}")

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            save_object(self.config.preprocessor_path, preprocessor)
            logging.info("Preprocessor saved")

            return train_arr, test_arr, self.config.preprocessor_path
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.Components.data_ingestion import DataIngestion

    train_path, test_path = DataIngestion().initiate_data_ingestion()
    train_arr, test_arr, _ = DataTransformation().initiate_data_transformation(
        train_path, test_path)
    print(train_arr.shape, test_arr.shape)