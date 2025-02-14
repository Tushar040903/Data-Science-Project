import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib

from src.datascience.constant import *
from src.datascience.utils.comman import read_yaml, create_directories, save_json
from src.datascience.entity.config_entity import ModelEvaluationConfig

# os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/tusharbhardwaj9873010398/Data-Science-Project.mlflow"
# os.environ["MLFLOW_TRACKING_USERNAME"] = "tusharbhardwaj9873010398"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "08bbad566a9e25e40f9109c5d90704bbc8974a56"


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig): 
        self.config = config

    def evaluate_model(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop(columns=[self.config.target_column])
        test_y = test_data[self.config.target_column]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)
            rmse, mae, r2 = self.evaluate_model(test_y, predicted_qualities)
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="model_v1")
            else:
                mlflow.sklearn.log_model(model, "model")
