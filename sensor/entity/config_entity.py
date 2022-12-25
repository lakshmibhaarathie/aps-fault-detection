import os
from datetime import datetime
from sensor.exception import SensorException

FILE_NAME = "sensor.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TARGET_COLUMN = "class"
REPORT_FILE_NAME = "report.yaml"
BASE_FILE_NAME1 = r"E:\\pythonProject1\\aps_failure_training_set1.csv"
BASE_FILE_NAME = "aps_failure_training_set1.csv"
TRANSFORMER_OBJECT_FILE_NAME = "transformer.pkl"
TARGET_ENCODER_OBJECT_FILE_NAME = "target_encoder.pkl"
MODEL_NAME = "model.pkl"


class TraininPipelineConfig:
    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(), "artifacts", f"{datetime.now().strftime('%d%m%Y__%H%M%S')}")
        except Exception as e:
            raise SensorException(e)


class DataIngestionConfig:
    database_name = "sensor"
    collection_name = "aps"
    test_size = 0.3

    def __init__(self, training_pipeline_config: TraininPipelineConfig):
        try:
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, "data_ingestion")
            self.feature_store_path = os.path.join(self.data_ingestion_dir, "feature_store", FILE_NAME)
            self.train_file_path = os.path.join(self.data_ingestion_dir, "dataset", TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_ingestion_dir, "dataset", TEST_FILE_NAME)
        except Exception as e:
            raise SensorException(e)

    def to_dict(self):
        try:
            return self.__dict__
        except Exception as e:
            raise SensorException(e)


class DataValidationConfig:
    missing_threshold = 0.3

    def __init__(self, training_pipeline_config: TraininPipelineConfig):
        try:
            self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_validation")
            self.report_file_path = os.path.join(self.data_validation_dir, REPORT_FILE_NAME)
            self.base_file_path = os.path.join(BASE_FILE_NAME)
        except Exception as e:
            raise SensorException(e)


class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TraininPipelineConfig):
        try:
            self.data_tranformation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_transformation")
            self.transformer_path = os.path.join(self.data_tranformation_dir, "transformer",
                                                 TRANSFORMER_OBJECT_FILE_NAME)
            self.transformed_train_path = os.path.join(self.data_tranformation_dir, "transformed",
                                                       TRAIN_FILE_NAME.replace("csv", "npz"))
            self.transformed_test_path = os.path.join(self.data_tranformation_dir, "transformed",
                                                      TEST_FILE_NAME.replace("csv", "npz"))
            self.target_encoder_path = os.path.join(self.data_tranformation_dir, "target_encoder",
                                                    TARGET_ENCODER_OBJECT_FILE_NAME)
        except Exception as e:
            raise SensorException(e)


class ModelTrainerConfig:
    expected_score = 0.8
    overfitting_threshold = 0.2

    def __init__(self, training_pipeline_config: TraininPipelineConfig):
        try:
            self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, "model_trainer")
            self.trained_model_path = os.path.join(self.model_trainer_dir, MODEL_NAME)
        except Exception as e:
            raise SensorException(e)


class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config: TraininPipelineConfig):
        try:
            self.change_threshold = 0.01
        except Exception as e:
            raise SensorException(e)


class ModelPusherConfig:
    def __init__(self, training_pipeline_config: TraininPipelineConfig):
        try:
            self.model_pusher_dir = os.path.join(training_pipeline_config.artifact_dir, "model_pusher")
            self.saved_model_pusher_dir = os.path.join(self.model_pusher_dir, "saved_model")
            self.pusher_model_path = os.path.join(self.saved_model_pusher_dir, MODEL_NAME)
            self.pusher_transformer_path = os.path.join(self.saved_model_pusher_dir, TRANSFORMER_OBJECT_FILE_NAME)
            self.pusher_target_encoder_path = os.path.join(self.saved_model_pusher_dir, TARGET_ENCODER_OBJECT_FILE_NAME)
            self.model_registry_dir = os.path.join("model_registry")

        except Exception as e:
            raise SensorException(e)
