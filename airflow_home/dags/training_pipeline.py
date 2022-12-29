import os
import json
import pendulum
from airflow import DAG
from asyncio import tasks
from textwrap import dedent
from airflow.operators.python import PythonOperator


with DAG("sensor_training", default_args={"retries": 2},
         description="Sensor Fault Detection", schedule_interval="@weekly",
         start_date=pendulum.datetime(2022, 12, 24, tz="UTC"), catchup=False,
         tags=['example']) as dag:
    def training(**kwargs):
        from sensor.pipeline.training_pipeline import start_training_pipeline
        start_training_pipeline()


    def sync_artifact_to_s3_bucket(**kwargs):
        bucket_name = os.getenv("BUCKET_NAME")
        os.system(f"aws s3 sync /app/artifacts s3://{bucket_name}/artifacts")
        os.system(f"aws s3 sync /app/model_registry s3://{bucket_name}/model_registry")


    trining_pipeline = PythonOperator(task_id="train_pipeline",
                                      python_callable=training)

    sync_data_to_s3 = PythonOperator(task_id="sync_data_to_s3",
                                     python_callable=sync_artifact_to_s3_bucket)

    trining_pipeline >> sync_data_to_s3
