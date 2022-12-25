import os
import json
import pendulum
from airflow import DAG
from asyncio import tasks
from textwrap import dedent
from airflow.operators.python import PythonOperator
from sensor.pipeline.batch_prediction import start_batch_prediction

with DAG("sensor_training", default_args={"retries": 2},
         description="Sensor Fault Detection", catchup=False,
         schedule_intrval="@weekly", tags=["example"],
         start_data=pendulum.datetime(2022, 12, 24, tz="UTC")) as dag:
    def download_files(**kwargs):
        bucket_name = os.getenv("BUCKET NAME")
        input_dir = "/app/input_files"
        # create directory
        os.makedirs(input_dir, exist_ok=True)
        os.system(f"aws s3 sync s3://{bucket_name}/input_files /app/input_files")


    def batch_prediction(**kwargs):
        input_dir = "/app/input_files"
        for filename in os.listdir(input_dir):
            start_batch_prediction(file_path=os.path.join(input_dir
                                                          , filename))


    def sync_prediction_to_s3_bucket(**kwargs):
        bucket_name = "BUCKET NAME"
        os.system(f"aws s3 sync /app/prediction s3://{bucket_name}/prediction_files")


    downoad_input_file = PythonOperator(task_id="download_file",
                                        python_callable=download_files)
    generate_prediction_files = PythonOperator(task_id="Prediction",
                                               python_callable=batch_prediction)
    upload_prediction_files = PythonOperator(task_id="upload_prediction_files",
                                             python_callable=sync_prediction_to_s3_bucket)
    downoad_input_file >> generate_prediction_files >> upload_prediction_files
