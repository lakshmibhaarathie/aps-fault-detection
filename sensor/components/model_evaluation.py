import pandas as pd

from sensor.entity import config_entity, artifact_entity
from sensor import utils
from sensor.model_resolver import ModelResolver
from sensor.exception import SensorException
from sklearn.metrics import f1_score


class ModelEvaluation:
    def __init__(self, model_evaluation_config: config_entity.ModelEvaluationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact: artifact_entity.ModelTrainerArtifact):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise SensorException(e)

    def initiate_model_evaluation(self):
        try:

            latest_dir_path = self.model_resolver.get_latest_registry_dir()

            if latest_dir_path is None:
                model_evaluation_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
                                                                                    improved_accuracy=None)
                return model_evaluation_artifact

            test_df_path = self.data_ingestion_artifact.test_file_path
            test_df = pd.read_csv(test_df_path)
            target = test_df[config_entity.TARGET_COLUMN]

            # getiing score for previously accepted model
            prev_transformer_path = self.model_resolver.get_latest_registry_transformer()
            prev_target_encoder_path = self.model_resolver.get_latest_registry_target_encoder()
            prev_model_path = self.model_resolver.get_latest_registry_model()

            prev_transformer_obj = utils.load_file_object(file_path=prev_transformer_path)
            prev_target_encoder_obj = utils.load_file_object(file_path=prev_target_encoder_path)
            prev_model_obj = utils.load_file_object(file_path=prev_model_path)

            input_feature_names = list(prev_transformer_obj.feature_names_in_)
            input_arr = prev_transformer_obj.transform(test_df[input_feature_names])
            y_true = prev_target_encoder_obj.transform(target)

            y_pred = prev_model_obj.predict(input_arr)
            prev_model_score = f1_score(y_true, y_pred)

            # getting score for the current model
            curr_transformer_path = self.data_transformation_artifact.transformer_object_path
            curr_target_encoder_path = self.data_transformation_artifact.target_encoder_path
            curr_model_path = self.model_trainer_artifact.trained_model_path

            curr_transformer_obj = utils.load_file_object(file_path=curr_transformer_path)
            curr_target_encoder_obj = utils.load_file_object(file_path=curr_target_encoder_path)
            curr_model_obj = utils.load_file_object(file_path=curr_model_path)

            input_feature_names = list(curr_transformer_obj.feature_names_in_)
            input_arr = curr_transformer_obj.transform(test_df[input_feature_names])
            y_true = curr_target_encoder_obj.transform(target)

            y_pred = curr_model_obj.predict(input_arr)
            curr_model_score = f1_score(y_true, y_pred)
            print(f"prev_model_score: {prev_model_score} current_model_score: {curr_model_score}")

            # validating both models
            improved_accuracy = curr_model_score - prev_model_score
            if round(curr_model_score,3) < round(prev_model_score):
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=False,
                                                                              improved_accuracy=improved_accuracy)
                return model_eval_artifact

            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
                                                                          improved_accuracy=improved_accuracy)
            return model_eval_artifact
        except Exception as e:
            raise SensorException(e)
