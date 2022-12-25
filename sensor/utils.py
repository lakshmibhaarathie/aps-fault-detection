import os.path
import yaml
import dill
import pandas as pd
import numpy as np
from typing import List
from sensor.exception import SensorException
from sensor.config import mongo_client


def get_mongoDB_collection(database_name: str, collection_name: str) -> pd.DataFrame:
    """
    Description:
        This function get data from mongoDB collections and converts it into pandas DataFrame
    :return: pandas.DataFrame
    """
    try:
        database_name = database_name
        collection_name = collection_name
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        df.drop("_id", axis=1, inplace=True)
        df.replace({"na": np.NAN}, inplace=True)
        return df
    except Exception as e:
        raise SensorException(e)


def convert_column_values(df: pd.DataFrame, exclude_columns: List) -> pd.DataFrame:
    try:
        for column in df.columns:
            if column not in exclude_columns:
                df[column] = df[column].astype(float)
        return df
    except Exception as e:
        raise SensorException(e)


def write_yaml_file(report_file_path: str, report_file: dict):
    try:
        report_file_dir = os.path.dirname(report_file_path)
        os.makedirs(report_file_dir, exist_ok=True)
        with open(report_file_path, "w") as file_writer:
            yaml.dump(report_file, file_writer)
    except Exception as e:
        raise SensorException(e)


def save_numpy_array(file_path: str, data: np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, data)
    except Exception as e:
        raise SensorException(e)


def save_file_object(file_path: str, obj: object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise SensorException(e)


def load_numpy_array(file_path: str):
    try:
        with open(file_path, "rb") as data:
            return np.load(data)
    except Exception as e:
        raise SensorException(e)


def load_file_object(file_path: str):
    try:
        with open(file_path, "rb") as obj:
            return dill.load(obj)
    except Exception as e:
        raise SensorException(e)
