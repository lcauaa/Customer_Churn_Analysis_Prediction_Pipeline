import logging
import os
from datetime import datetime

import boto3
from dotenv import load_dotenv

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

load_dotenv("/opt/airflow/.env")

AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")

local_input = "/opt/airflow/data/tableau_ready_customer_churn.csv"
local_output = "/opt/airflow/data/customer_churn_predictions.csv"
remote_input_key = "tableau_ready_customer_churn.csv"
remote_output_key = "customer_churn_predictions.csv"


def download_from_s3():
    logging.warning(f"Loaded BUCKET_NAME: {BUCKET_NAME}")
    logging.warning("Starting S3 download...")

    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_KEY,
            aws_secret_access_key=AWS_SECRET,
            region_name=AWS_REGION,
        )

        logging.warning(f"Downloading from: s3://{BUCKET_NAME}/{remote_input_key}")
        s3.download_file(BUCKET_NAME, remote_input_key, local_input)

        logging.warning(f"File downloaded to {local_input}")

    except Exception as e:
        logging.error(f"Download failed: {str(e)}")
        raise e


def upload_to_s3():
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name=AWS_REGION,
    )
    s3.upload_file(local_output, BUCKET_NAME, remote_output_key)


default_args = {
    "start_date": datetime(2025, 5, 21),
    "retries": 0,
}

with DAG(
    "churn_prediction_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:
    download_task = PythonOperator(
        task_id="download_from_s3", python_callable=download_from_s3
    )

    run_pipeline = BashOperator(
        task_id="run_ml_pipeline",
        bash_command="python3 /opt/airflow/scripts/churn_pipeline.py",
    )

    upload_task = PythonOperator(task_id="upload_to_s3", python_callable=upload_to_s3)

    download_task >> run_pipeline >> upload_task
