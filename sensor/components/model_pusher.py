from sensor.entity import config_entity, artifact_entity
from sensor.exception import SensorException
from sensor import utils, model_resolver


class ModelPusher:
    def __init__(self, model_pusher_config: config_entity.ModelPusherConfig,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact: artifact_entity.ModelTrainerArtifact,
                 model_eval_artifact: artifact_entity.ModelEvaluationArtifact):
        try:
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_eval_artifact = model_eval_artifact
            self.model_resolver = model_resolver.ModelResolver(
                model_registry=self.model_pusher_config.model_registry_dir)
        except Exception as e:
            raise SensorException(e)

    def initiate_model_pusher(self):
        try:
            if self.model_eval_artifact.is_model_accepted == False:
                # get saved objects
                transformer = utils.load_file_object(
                    file_path=self.data_transformation_artifact.transformer_object_path)
                target_encoder = utils.load_file_object(file_path=self.data_transformation_artifact.target_encoder_path)
                model = utils.load_file_object(file_path=self.model_trainer_artifact.trained_model_path)

                #  dump in model pusher directory
                utils.save_file_object(file_path=self.model_pusher_config.pusher_transformer_path, obj=transformer)
                utils.save_file_object(file_path=self.model_pusher_config.pusher_target_encoder_path,
                                       obj=target_encoder)
                utils.save_file_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)
                return None



            # get saved objects
            transformer = utils.load_file_object(file_path=self.data_transformation_artifact.transformer_object_path)
            target_encoder = utils.load_file_object(file_path=self.data_transformation_artifact.target_encoder_path)
            model = utils.load_file_object(file_path=self.model_trainer_artifact.trained_model_path)

            #  dump in model pusher directory
            utils.save_file_object(file_path=self.model_pusher_config.pusher_transformer_path, obj=transformer)
            utils.save_file_object(file_path=self.model_pusher_config.pusher_target_encoder_path, obj=target_encoder)
            utils.save_file_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)

            # get model registry path
            transformer_path = self.model_resolver.new_model_registry_transformer_path()
            target_encoder_path = self.model_resolver.new_model_registry_target_encoder_path()
            model_path = self.model_resolver.new_model_registry_model_path()

            # dump in model registry
            utils.save_file_object(file_path=transformer_path, obj=transformer)
            utils.save_file_object(file_path=target_encoder_path, obj=target_encoder)
            utils.save_file_object(file_path=model_path, obj=model)

            model_pusher_artifact = artifact_entity.ModelPusherArtifact(
                pusher_model_dir=self.model_pusher_config.saved_model_pusher_dir,
                model_registry_dir=self.model_pusher_config.model_registry_dir)
            return model_pusher_artifact
        except Exception as e:
            raise SensorException(e)
