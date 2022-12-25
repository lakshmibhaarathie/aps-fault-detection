import os
from typing import Optional
import pandas as pd
import numpy as np
from sensor import utils
from scipy.stats import ks_2samp
from sensor.entity import config_entity, artifact_entity
from sensor.exception import SensorException


class DataValidation:
    def __init__(self, data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
                 data_validation_config: config_entity.DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.validation_report = dict()
        except Exception as e:
            raise SensorException(e)

    def drop_missing_columns(self, df: pd.DataFrame, threshold: float, report_key_name: str) -> Optional[pd.DataFrame]:
        try:
            df.replace(to_replace="na", value=np.NAN, inplace=True)
            null_report = df.isna().sum() / df.shape[0]
            drop_column_names = null_report[null_report > threshold].index
            df.drop(list(drop_column_names), axis=1, inplace=True)
            self.validation_report[report_key_name] = list(drop_column_names)
            if len(df.columns) == 0:
                return None
            return df
        except Exception as e:
            raise SensorException(e)

    def is_required_column_exists(self, base_df: pd.DataFrame, current_df: pd.DataFrame, report_key_name: str,
                                  curr_df_name: str):
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns
            missing_columns = list()
            for base_column in base_columns:
                if base_column not in current_columns:
                    missing_columns.append(base_column)

            if missing_columns:
                self.validation_report[report_key_name] = missing_columns
                raise SensorException(e)
            self.validation_report[report_key_name] = f"All the required columns are present in the {curr_df_name}."
            return True
        except Exception as e:
            raise SensorException(e)
    def drift_report(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str):
        try:
            drift_report = dict()
            base_columns = base_df.columns
            for base_column in base_columns:
                base_data, current_data = base_df[base_column], current_df[base_column]

                sample_distribution = ks_2samp(base_data,current_data)

                if sample_distribution.pvalue > 0.05:
                    drift_report[base_column] = {"pvalue":float(sample_distribution.pvalue),
                                                 "same_distribution":True}
                else:
                    drift_report[base_column] = {"pvalue":float(sample_distribution.pvalue),
                                                 "same_distribution":False}
            self.validation_report[report_key_name] = drift_report
        except Exception as e:
            raise SensorException(e)

    def initiate_data_validation(self):
        try:
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            missing_threshold = self.data_validation_config.missing_threshold
            base_df = self.drop_missing_columns(df=base_df, threshold=missing_threshold,
                                                report_key_name="base_df_drop_missing_value_columns")
            train_df = self.drop_missing_columns(df=train_df, threshold=missing_threshold,
                                                 report_key_name="train_df_drop_missing_value_columns")
            test_df = self.drop_missing_columns(df=test_df, threshold=missing_threshold,
                                                report_key_name="test_df_drop_missing_value_columns")
            train_df_column_status = self.is_required_column_exists(base_df=base_df, current_df=train_df,
                                                                    report_key_name="is_required_columns_exists_train_df",
                                                                    curr_df_name="train_df")
            test_df_column_status = self.is_required_column_exists(base_df=base_df, current_df=test_df,
                                                                   report_key_name="is_required_columns_exists_test_df",
                                                                   curr_df_name="test_df")
            exclude_column = [config_entity.TARGET_COLUMN]

            base_df = utils.convert_column_values(df=base_df,exclude_columns=exclude_column)
            train_df = utils.convert_column_values(df=train_df,exclude_columns=exclude_column)
            test_df = utils.convert_column_values(df=test_df,exclude_columns=exclude_column)
            if train_df_column_status:
                self.drift_report(base_df=base_df,current_df=train_df,report_key_name="drift_report_train_df")
            if test_df_column_status:
                self.drift_report(base_df=base_df, current_df=test_df,report_key_name="drift_report_test_df")

            # prepare validation report
            utils.write_yaml_file(report_file=self.validation_report,
                                  report_file_path=self.data_validation_config.report_file_path)
            data_validation_artifact = artifact_entity.DataValidationArtifact(
                report_file_path=self.data_validation_config.report_file_path)
            return data_validation_artifact
        except Exception as e:
            raise SensorException(e)
