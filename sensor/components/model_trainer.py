import pandas as pd

from sensor.entity import config_entity, artifact_entity
from sensor import utils
from sensor.exception import SensorException
from xgboost import XGBClassifier
from sklearn.metrics import f1_score


class ModelTrainer:
    def __init__(self, model_trainer_config: config_entity.ModelTrainerConfig,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    def train_model(self, feature: pd.DataFrame, target: pd.DataFrame) -> object:
        try:
            model = XGBClassifier()
            model.fit(feature, target)
            return model
        except Exception as e:
            raise SensorException(e)

    def initiate_model_trainer(self, ) -> artifact_entity.ModelTrainerArtifact:
        try:
            train_df = utils.load_numpy_array(file_path=self.data_transformation_artifact.transformed_train_path)
            test_df = utils.load_numpy_array(file_path=self.data_transformation_artifact.transformed_test_path)

            X_train, y_train = train_df[:, :-1], train_df[:,-1]
            X_test, y_test = test_df[:, :-1], test_df[:,-1]

            model = self.train_model(feature=X_train, target=y_train)

            y_train_pred = model.predict(X_train)
            f1_train_score = f1_score(y_train, y_train_pred)

            y_test_pred = model.predict(X_test)
            f1_test_score = f1_score(y_test, y_test_pred)

            diff = abs(f1_train_score - f1_test_score)

            if f1_test_score < self.model_trainer_config.expected_score:
                raise Exception(
                    f"The current model accuracy:{f1_test_score} is not matching the expected accuracy{self.model_trainer_config.expected_score}")
            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(
                    f"The difference between training accuracy and test accuracy {diff} is greater than the expected threshold.")

            utils.save_file_object(file_path=self.model_trainer_config.trained_model_path, obj=model)

            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.trained_model_path,
                f1_train_score=f1_train_score, f1_test_score=f1_test_score)
            return model_trainer_artifact

        except Exception as e:
            raise SensorException(e)
