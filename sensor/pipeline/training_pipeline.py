from sensor.exception import SensorException
from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import DataValidation
from sensor.components.data_transformation import DataTransformation
from sensor.components.model_trainer import ModelTrainer
from sensor.components.model_evaluation import ModelEvaluation
from sensor.components.model_pusher import ModelPusher
from sensor.entity import config_entity, artifact_entity


def start_training_pipeline():
    try:
        training_pipline_config = config_entity.TraininPipelineConfig()
        # data ingestion
        data_ingestion_config = config_entity.DataIngestionConfig(training_pipeline_config=training_pipline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        # data validation
        data_validation_config = config_entity.DataValidationConfig(training_pipeline_config=training_pipline_config)
        data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                         data_validation_config=data_validation_config)
        data_validation_artifact = data_validation.initiate_data_validation()

        # data transformation
        data_transformation_config = config_entity.DataTransformationConfig(
            training_pipeline_config=training_pipline_config)
        data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                 data_transformation_config=data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_tranformation()

        # model training
        model_trainer_config = config_entity.ModelTrainerConfig(training_pipeline_config=training_pipline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config,
                                     data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()

        # model evaluation
        model_eval_config = config_entity.ModelEvaluationConfig(training_pipeline_config=training_pipline_config)
        model_evaluation = ModelEvaluation(data_ingestion_artifact=data_ingestion_artifact,
                                           data_transformation_artifact=data_transformation_artifact,
                                           model_trainer_artifact=model_trainer_artifact,
                                           model_evaluation_config=model_eval_config)
        model_eval_artifact = model_evaluation.initiate_model_evaluation()

        # model pusher
        model_pusher_config = config_entity.ModelPusherConfig(training_pipeline_config=training_pipline_config)
        model_pusher = ModelPusher(data_transformation_artifact=data_transformation_artifact,
                                   model_trainer_artifact=model_trainer_artifact,
                                   model_eval_artifact=model_eval_artifact, model_pusher_config=model_pusher_config)
        model_pusher_artifact = model_pusher.initiate_model_pusher()
    except Exception as e:
        raise SensorException(e)
