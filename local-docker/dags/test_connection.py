from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mlflow
import os
import random

load_dotenv() 

def test_mlflow_connection():
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("MINIO_ACCESS_KEY")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("MINIO_SECRET_ACCESS_KEY")

    mlflow.set_tracking_uri("postgresql://mlflow:" + os.environ.get("POSTGRES_PASSWORD_MLFLOW") + "@mlflowdb/mlflow")
    mlflow.set_experiment("test_connection_airflow")

    with mlflow.start_run() as run:
        mlflow.log_param("test_param", random.randint(1, 100))
        mlflow.log_metric("test_metric", random.random())
        with open("test_artifact.txt", "w") as f:
            f.write("This is a test artifact from Airflow.")
        mlflow.log_artifact("test_artifact.txt")
        print(f"Run ID: {run.info.run_id}")

    print("MLflow connection test from Airflow completed.")

with DAG(
    dag_id='test_mlflow_connection_dag',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mlflow', 'minio', 'postgres'],
) as dag:
    test_task = PythonOperator(
        task_id='test_mlflow_connection',
        python_callable=test_mlflow_connection,
    )