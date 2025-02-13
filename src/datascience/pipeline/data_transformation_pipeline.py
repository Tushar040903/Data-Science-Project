from src.datascience.config.congiguration import ConfigurationManager
from src.datascience.components.data_transformation import DataTransformation
from pathlib import Path
from src.datascience import logger

STAGE_NAME = "Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        try:
            with open(Path("artifacts\data_validation\status.txt"), 'r') as f:
                status = f.read().split(" ")[-1]
                if status == "True":
                    config = ConfigurationManager()
                    data_transformation_config = config.get_data_transformation_config()
                    data_transformation = DataTransformation(config=data_transformation_config)
                    data_transformation.train_test_splitting()
                else: 
                    raise Exception("Your data Scheme is not valid")
        except Exception as e:
            print(e)

if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.initiate_data_transformation()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.error(e)