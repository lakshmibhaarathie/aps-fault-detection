from sensor.pipeline.training_pipeline import start_training_pipeline
from sensor.pipeline.batch_prediction import start_batch_prediction
from sensor.exception import SensorException
import os

file_path = os.path.join("aps_failure_training_set1.csv")

if __name__ == "__main__":
    try:
        start_training_pipeline()
        start_batch_prediction(file_path=file_path)
    except Exception as e:
        raise SensorException(e)
