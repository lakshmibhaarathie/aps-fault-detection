import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sensor.entity import config_entity, artifact_entity
from sensor.exception import SensorException
from sensor.utils import get_mongoDB_collection


class DataIngestion:
    def __init__(self, data_ingestion_config: config_entity.DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise SensorException(e)

    def initiate_data_ingestion(self):
        try:
            df: pd.DataFrame = get_mongoDB_collection(database_name=self.data_ingestion_config.database_name,
                                                      collection_name=self.data_ingestion_config.collection_name)
            df.replace({"na": np.NAN}, inplace=True)

            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_path)
            os.makedirs(feature_store_dir, exist_ok=True)
            train_file_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(train_file_dir, exist_ok=True)

            train_df, test_df = train_test_split(df, test_size=self.data_ingestion_config.test_size,random_state=3)
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_path, index=False)
            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path, index=False)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path, index=False)

            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_path=self.data_ingestion_config.feature_store_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path)

            return data_ingestion_artifact
        except Exception as e:
            raise SensorException(e)
