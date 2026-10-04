import os
import sys
from dataclasses import dataclass

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts", "model.pkl")


PERSISTENCE = "Persistence (delta = 0)"


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            models = {
                PERSISTENCE: DummyRegressor(strategy="constant", constant=0.0),
                "Median delta": DummyRegressor(strategy="median"),
                "Ridge": Ridge(alpha=1.0),
                "Random Forest": RandomForestRegressor(
                    n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1),
                "Gradient Boosting": GradientBoostingRegressor(
                    loss="absolute_error", random_state=42),
            }

            # error of predicting the change == error of forecasting the delay (minutes)
            results = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                results[name] = mean_absolute_error(y_test, model.predict(X_test))
                logging.info(f"{name}: MAE = {results[name]:.2f} min")

            persistence_mae = results[PERSISTENCE]
            candidates = {k: v for k, v in results.items()
                          if k not in (PERSISTENCE, "Median delta")}
            best_name = min(candidates, key=candidates.get)

            if candidates[best_name] >= persistence_mae:
                logging.warning("No model beat persistence")

            save_object(self.config.trained_model_path, models[best_name])
            logging.info(f"Best model: {best_name}; saved")
            return results, best_name

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.Components.data_ingestion import DataIngestion
    from src.Components.data_transformation import DataTransformation

    train_path, test_path = DataIngestion().initiate_data_ingestion()
    train_arr, test_arr, _ = DataTransformation().initiate_data_transformation(
        train_path, test_path)
    results, best = ModelTrainer().initiate_model_trainer(train_arr, test_arr)

    for name, mae in results.items():
        print(f"{name:26s} MAE = {mae:6.2f} min")
    print("Best model:", best)