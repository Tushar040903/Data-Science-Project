import os
from src.datascience import logger
from sklearn.model_selection import train_test_split
import pandas as pd

from src.datascience.entity.config_entity import (DataTransformationConfig)


from dataclasses import dataclass
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import logging

logger = logging.getLogger(__name__)

@dataclass
class DataTransformationConfig:
    root_dir: Path
    input_file: Path

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_splitting(self):
        data = pd.read_csv(self.config.input_file)
        for col in ["ID", "Delivery_person_ID"]:
            if col in data.columns:
                data.drop(col, axis=1, inplace=True)
        target_column = "TARGET"
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        numeric_features = X_train.select_dtypes(include=["float64", "int"]).columns.tolist()
        categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
            ]
        )
        
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
        all_feature_names = numeric_features + cat_feature_names.tolist()
        
        X_train_df = pd.DataFrame(
            X_train_transformed.toarray() if hasattr(X_train_transformed, "toarray") else X_train_transformed,
            columns=all_feature_names
        )
        X_train_df[target_column] = y_train.reset_index(drop=True)
        
        X_test_df = pd.DataFrame(
            X_test_transformed.toarray() if hasattr(X_test_transformed, "toarray") else X_test_transformed,
            columns=all_feature_names
        )
        X_test_df[target_column] = y_test.reset_index(drop=True)
        
        os.makedirs(self.config.root_dir, exist_ok=True)
        train_path = os.path.join(self.config.root_dir, "train.csv")
        test_path = os.path.join(self.config.root_dir, "test.csv")
        X_train_df.to_csv(train_path, index=False)
        X_test_df.to_csv(test_path, index=False)
        
        pipeline_path = os.path.join(self.config.root_dir, "preprocessor.pkl")
        joblib.dump(preprocessor, pipeline_path)
        
        logger.info("Train and test data split and transformed successfully")
        logger.info(f"Train shape: {X_train_df.shape}")
        logger.info(f"Test shape: {X_test_df.shape}")
        print(X_train_df.shape)
        print(X_test_df.shape)
