import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    delays_path: str = os.path.join("data", "delay_history.csv")
    trains_path: str = os.path.join("railpull", "data", "out", "trains.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    max_delay_min: int = 720        # README: delays above ~12h are ghost data
    max_gap_min: int = 60           # drop pairs with a hole in the collection
    test_fraction: float = 0.2
    min_pair_snapshots: int = 3     # below this, fall back to a random split


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def build_dataset(self) -> pd.DataFrame:
        cfg = self.ingestion_config

        delays = pd.read_csv(cfg.delays_path, dtype={"train_no": str},
                             parse_dates=["snapshot_time"])
        logging.info(f"Loaded {len(delays)} delay rows, "
                     f"{delays['snapshot_time'].nunique()} snapshots")

        # keep running trains with believable delays, one row per train per snapshot
        delays = delays[(delays["cancelled"] == 0)
                        & (delays["delay_min"] <= cfg.max_delay_min)]
        delays = delays.drop_duplicates(["train_no", "snapshot_time"])
        delays = delays.sort_values(["train_no", "snapshot_time"]).copy()

        # history: the train's previous two appearances
        g = delays.groupby("train_no")
        delays["prev_delay"] = g["delay_min"].shift(1)
        delays["prev_time"] = g["snapshot_time"].shift(1)
        delays["prev_delay2"] = g["delay_min"].shift(2)
        prev_time2 = g["snapshot_time"].shift(2)

        delays["gap_min"] = (delays["snapshot_time"] - delays["prev_time"]
                             ).dt.total_seconds() / 60
        gap2 = (delays["prev_time"] - prev_time2).dt.total_seconds() / 60

        # only keep pairs without a collection hole
        delays = delays[delays["gap_min"].notna()
                        & (delays["gap_min"] <= cfg.max_gap_min)].copy()
        gap2 = gap2.loc[delays.index]

        # second lag is optional: if missing or too old, assume "no trend"
        delays["prev_delay2"] = delays["prev_delay2"].where(gap2 <= cfg.max_gap_min)
        delays["prev_delay2"] = delays["prev_delay2"].fillna(delays["prev_delay"])
        delays["trend"] = delays["prev_delay"] - delays["prev_delay2"]

        delays["delta"] = delays["delay_min"] - delays["prev_delay"]   # the target
        logging.info(f"{len(delays)} usable (previous -> now) pairs")

        # attach timetable facts about the train
        trains = pd.read_csv(cfg.trains_path, dtype=str).rename(
            columns={"number": "train_no"})
        df = delays.merge(trains, on="train_no", how="inner")
        logging.info(f"{len(delays) - len(df)} pairs had no timetable match")

        df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
        df["num_stops"] = pd.to_numeric(df["num_stops"], errors="coerce")
        tt = df["travel_time"].str.split(":", expand=True)
        df["travel_min"] = (pd.to_numeric(tt[0], errors="coerce") * 60
                            + pd.to_numeric(tt[1], errors="coerce"))

        df["hour"] = df["snapshot_time"].dt.hour
        df["weekday"] = df["snapshot_time"].dt.weekday   # 0 = Monday

        return df.dropna(subset=["distance_km", "num_stops", "travel_min"])

    def split(self, df: pd.DataFrame):
        cfg = self.ingestion_config
        times = sorted(df["snapshot_time"].unique())

        if len(times) >= cfg.min_pair_snapshots:
            cutoff = times[int(len(times) * (1 - cfg.test_fraction))]
            train_set = df[df["snapshot_time"] < cutoff]
            test_set = df[df["snapshot_time"] >= cutoff]
            logging.info(f"Time-based split: train < {cutoff} <= test")
        else:
            logging.warning("Too few snapshots for a time split, using a random split")
            train_set, test_set = train_test_split(
                df, test_size=cfg.test_fraction, random_state=42)

        # the number every model has to beat
        persistence_mae = test_set["delta"].abs().mean()
        logging.info(f"Train rows: {len(train_set)}, test rows: {len(test_set)}")
        logging.info(f"Persistence MAE on test set: {persistence_mae:.2f} min")
        return train_set, test_set

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            cfg = self.ingestion_config
            df = self.build_dataset()
            logging.info(f"Built dataset with {df.shape[0]} rows, {df.shape[1]} columns")

            os.makedirs(os.path.dirname(cfg.train_data_path), exist_ok=True)
            df.to_csv(cfg.raw_data_path, index=False)

            train_set, test_set = self.split(df)
            train_set.to_csv(cfg.train_data_path, index=False)
            test_set.to_csv(cfg.test_data_path, index=False)

            logging.info("Ingestion completed")
            return cfg.train_data_path, cfg.test_data_path
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    DataIngestion().initiate_data_ingestion()