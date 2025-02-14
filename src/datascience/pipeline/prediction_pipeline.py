import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        self.preprocessor = pickle.load(Path('artifacts/data_transformation/preprocessor.pkl'))
        self.expected_columns = [
            "Delivery_person_Age",
            "Delivery_person_Ratings",
            "Restaurant_latitude",
            "Restaurant_longitude",
            "Delivery_location_latitude",
            "Delivery_location_longitude",
            "Type_of_order",
            "Type_of_vehicle",
            "temperature",
            "humidity",
            "precipitation",
            "weather_description",
            "Traffic_Level",
            "Distance (km)"
        ]

    def predict(self, data):
        # Flatten nested input if needed
        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
            data = data[0]
            input_df = pd.DataFrame([data], columns=self.expected_columns)
            transformed_data = self.preprocessor.transform(input_df)
            prediction = self.model.predict(transformed_data)
        return prediction
