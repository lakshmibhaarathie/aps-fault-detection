# this should be done by data engineer.
import os
import pymongo
import pandas as pd
import numpy as np
import json
from sensor.exception import SensorException

client = pymongo.MongoClient("mongodb+srv://aakash:aakash@cluster0.4gxtdut.mongodb.net/?retryWrites=true&w=majority")


DATA_FILE_PATH = os.path.join('aps_failure_training_set1.csv')
DATABASE_NAME = 'sensor'
COLLECTION_NAME = 'aps'

if __name__ == '__main__':
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        # replace null values
        df.replace({"na":np.NaN},inplace=True)
        # remove index
        df.reset_index(drop=True, inplace=True)
        # convert csv --> json
        json_records = df.T.to_json()
        json_records = list(json.loads(json_records).values())

        # upload the json_file --> mongoDB
        client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_records)
        print("Data dump success...!")
    except Exception as e:
        raise SensorException(e)
