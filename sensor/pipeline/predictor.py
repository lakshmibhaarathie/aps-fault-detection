import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sensor import utils
from sensor.exception import SensorException

class PredDataValidation:
    missing_threshold = 0.3
    def __init__(self):
        self.validation_report = dict()

    def drop_missing_value_columns(self,df:pd.DataFrame,report_key_name:str):
        try:
            null_report = df.isna().sum() / df.shape[0]
            drop_column_names = null_report[null_report>self.missing_threshold].index
            df.drop(list(drop_column_names),axis=1,inplace=True)
            self.validation_report[report_key_name] = list(drop_column_names)
            if len(df) ==0:
                return None
            return df
        except Exception as e:
            raise SensorException(e)

    def is_required_columns_exists(self,base_df:pd.DataFrame,pred_df:pd.DataFrame,report_key_name:str):
        try:
            base_columns = base_df.columns
            pred_columns = pred_df.columns
            missing_columns = list()
            for base_column in base_columns:
                if base_column not in pred_columns:
                    missing_columns.append(base_column)

            if missing_columns:
                self.validation_report[report_key_name] = missing_columns
                return False
            self.validation_report[report_key_name] = "All the required columns exists"
            return True
        except Exception as e:
            raise SensorException(e)
    def data_drift(self,base_df:pd.DataFrame,pred_df:pd.DataFrame,report_key_name:str):
        drift_report = dict()
        base_columns = base_df.columns
        for base_column in base_columns:
            base_data,pred_data = base_df[base_column], pred_df[base_column]

            sample_distribution = ks_2samp(base_data,pred_data)

            if sample_distribution.pvalue>0.05
                drift_report[base_column] = {"pvalue":sample_distribution.pvalue,
                                             "same_distribution":True}
            else:
                drift_report[base_column] = {"pvalue":sample_distribution.pvalue,
                                             "same_distribution":False}
        self.validation_report[report_key_name] = drift_report

    def initiate_data_validation(self):

        base_df = pd.read_csv(base_df_path)
        pred_df = pd.read_csv(pred_df_path)

        base_df = self.drop_missing_value_columns(df=base_df,report_key_name="drop_missing_value_columns_base_df")
        pred_df = self.drop_missing_value_columns(df=pred_df,report_key_name="drop_missing_value_columns_pred_df")

        columns_status = self.is_required_columns_exists(base_df=base_df,pred_df=pred_df,report_key_name="is_required_columns_exists")

        exclude_column = ["class"]
        base_df = utils.convert_column_values(df=base_df,exclude_columns=exclude_column)
        pred_df = utils.convert_column_values(df=pred_df,exclude_columns=exclude_column)
        if columns_status:
            self.data_drift(base_df=base_df,pred_df=pred_df,report_key_name="check_data_drift")

        report_file= self.validation_report













