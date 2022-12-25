import os
from sensor.exception import SensorException
from sensor.entity import config_entity


class ModelResolver:
    transformer_dir = "transformer"
    target_encoder_dir = "target_encoder"
    model_dir = "model"

    def __init__(self, model_registry:str = "model_registry"):
        self.model_registry = model_registry
        os.makedirs(self.model_registry, exist_ok=True)

    def get_latest_registry_dir(self):
        try:
            dir_names = os.listdir(self.model_registry)
            if not dir_names:
                return None
            dir_names = list(map(int, dir_names))
            latest_dir_name = max(dir_names)
            latest_dir_path = os.path.join(self.model_registry, f"{latest_dir_name}")
            return latest_dir_path
        except Exception as e:
            raise SensorException(e)

    def get_latest_registry_transformer(self):
        try:
            latest_dir_path = self.get_latest_registry_dir()
            if not latest_dir_path:
                return None
            latest_registry_transformer_path = os.path.join(latest_dir_path, self.transformer_dir,
                                                            config_entity.TRANSFORMER_OBJECT_FILE_NAME)
            return latest_registry_transformer_path
        except Exception as e:
            raise SensorException(e)

    def get_latest_registry_target_encoder(self):
        try:
            latest_dir_path = self.get_latest_registry_dir()
            if not latest_dir_path:
                return None
            latest_registry_target_encoder_path = os.path.join(latest_dir_path, self.target_encoder_dir,
                                                               config_entity.TARGET_ENCODER_OBJECT_FILE_NAME)
            return latest_registry_target_encoder_path
        except Exception as e:
            raise SensorException(e)

    def get_latest_registry_model(self):
        try:
            latest_dir_path = self.get_latest_registry_dir()
            if not latest_dir_path:
                return None
            latest_registry_model_path = os.path.join(latest_dir_path, self.model_dir,
                                                      config_entity.MODEL_NAME)
            return latest_registry_model_path
        except Exception as e:
            raise SensorException(e)

    def new_model_registry_dir(self):
        try:
            latest_dir = self.get_latest_registry_dir()
            if not latest_dir:
                new_model_registry_dir = os.path.join(self.model_registry, f"{0}")
                return new_model_registry_dir
            latest_dir_number = int(os.path.basename(latest_dir))
            new_model_registry_dir = os.path.join(self.model_registry, f"{latest_dir_number + 1}")
            return new_model_registry_dir
        except Exception as e:
            raise SensorException(e)

    def new_model_registry_model_path(self):
        try:
            new_model_registry_dir = self.new_model_registry_dir()
            new_model_path = os.path.join(new_model_registry_dir, self.model_dir, config_entity.MODEL_NAME)
            return new_model_path
        except Exception as e:
            raise SensorException(e)

    def new_model_registry_transformer_path(self):
        try:
            new_model_registry_dir = self.new_model_registry_dir()
            new_transformer_path = os.path.join(new_model_registry_dir, self.transformer_dir,
                                                config_entity.TRANSFORMER_OBJECT_FILE_NAME)
            return new_transformer_path
        except Exception as e:
            raise SensorException(e)

    def new_model_registry_target_encoder_path(self):
        try:
            new_model_registry_dir = self.new_model_registry_dir()
            new_taget_encoder_path = os.path.join(new_model_registry_dir, self.target_encoder_dir,
                                                  config_entity.TARGET_ENCODER_OBJECT_FILE_NAME)
            return new_taget_encoder_path
        except Exception as e:
            raise SensorException(e)
