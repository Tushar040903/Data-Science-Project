from urllib.request import urlretrieve
import os
import zipfile
from src.datascience import logger

from src.datascience.entity.config_entity import (DataIngestionConfig)

class DataIngestion:
    def __init__(self, config):
        self.config = config

    ## downloading the zip file
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, header = urlretrieve(
                url=self.config.source_url,
                filename=self.config.local_data_file
            )
            logger.info(f"{filename} downloaded with header: {header}")
        else:
            logger.info(f"{self.config.local_data_file} already exists")

    ## unzipping the file
    def unzip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
