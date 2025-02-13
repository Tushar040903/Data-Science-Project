import os
from src.datascience import logger
from sklearn.model_selection import train_test_split
import pandas as pd

from src.datascience.entity.config_entity import (DataTransformationConfig)


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    ## Note we can add diff data transformation techniques such as scaler, PCA, etc.
    def train_test_splitting(self):
        data = pd.read_csv(self.config.input_file)  # FIXED attribute name

        # Splitting the data
        train, test = train_test_split(data, test_size=0.2, random_state=42)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)  # FIXED os.Path.join
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Train and test data split successfully")
        logger.info(f"Train shape: {train.shape}")
        logger.info(f"Test shape: {test.shape}")

        print(train.shape)
        print(test.shape)