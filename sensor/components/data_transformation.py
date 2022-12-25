import numpy as np
import pandas as pd
from sensor.entity import config_entity, artifact_entity
from sensor.exception import SensorException
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
from sensor import config
from sensor import utils


class DataTransformation:
    def __init__(self, data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
                 data_transformation_config: config_entity.DataTransformationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise SensorException(e)

    @classmethod
    def get_data_tranformer_object(cls) -> Pipeline:
        try:
            simple_imputer = SimpleImputer(strategy="constant", fill_value=0)
            robust_scaler = RobustScaler()
            pipeline = Pipeline(steps=[
                ("Imputer", simple_imputer),
                ("Robust Scaler", robust_scaler)
            ])
            return pipeline
        except Exception as e:
            raise SensorException(e)

    def initiate_data_tranformation(self, ) -> artifact_entity.DataTransformationArtifact:
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            TARGET_COLUMN = config_entity.TARGET_COLUMN
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis=1)

            transformation_pipeline = self.get_data_tranformer_object()
            transformation_pipeline.fit(input_feature_train_df)

            input_feature_train_df = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_df = transformation_pipeline.transform(input_feature_test_df)

            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)

            target_feature_train_df = label_encoder.transform(target_feature_train_df)
            target_feature_test_df = label_encoder.transform(target_feature_test_df)

            smt = SMOTETomek(sampling_strategy="not minority",random_state=3)

            input_feature_train_df, target_feature_train_df = smt.fit_resample(input_feature_train_df,
                                                                               target_feature_train_df)

            input_feature_test_df, target_feature_test_df = smt.fit_resample(input_feature_test_df, target_feature_test_df)

            train_arr = np.c_[input_feature_train_df, target_feature_train_df]
            test_arr = np.c_[input_feature_test_df, target_feature_test_df]

            utils.save_numpy_array(file_path=self.data_transformation_config.transformed_train_path, data=train_arr)
            utils.save_numpy_array(file_path=self.data_transformation_config.transformed_test_path, data=test_arr)

            utils.save_file_object(file_path=self.data_transformation_config.transformer_path, obj=transformation_pipeline)
            utils.save_file_object(file_path=self.data_transformation_config.target_encoder_path, obj=label_encoder)

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transformed_train_path=self.data_transformation_config.transformed_train_path,
                transformed_test_path=self.data_transformation_config.transformed_test_path,
                transformer_object_path=self.data_transformation_config.transformer_path,
                target_encoder_path=self.data_transformation_config.target_encoder_path)
            return data_transformation_artifact
        except Exception as e:
            raise SensorException(e)