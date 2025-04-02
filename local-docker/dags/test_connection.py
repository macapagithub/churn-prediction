from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.exceptions import AirflowException
from datetime import datetime, timedelta
import mlflow
import os
import random
import logging
from dotenv import load_dotenv

# Cargar variables de entorno desde archivo .env
# Mejor aplicar esto en la configuración de Airflow en vez de en cada DAG
load_dotenv()

# Definir parámetros predeterminados para el DAG
default_args = {
    'owner': 'data_science_team',
    'depends_on_past': False,
    'email': ['alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

@dag(
    dag_id='test_mlflow_connection_dag',
    default_args=default_args,
    description='DAG for test the connection between MinIO, MLflow and Airflow',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mlflow', 'minio', 'test'],
)
def mlflow_connection_dag():
    """
    DAG para probar la integración de Airflow con MLflow y MinIO como almacenamiento de artefactos.
    Este DAG verifica que todas las credenciales y conexiones estén configuradas correctamente.
    """
    
    REQUIRED_VARS = [
        'MLFLOW_TRACKING_URI', 'MINIO_ACCESS_KEY', 'MINIO_SECRET_ACCESS_KEY',
        'MLFLOW_S3_ENDPOINT_URL'
    ]
    
    def get_variable(var_name, required=True):
        """Obtiene el valor de una variable de Airflow y verifica si es obligatoria."""
        value = Variable.get(var_name, default_var=None)
        if required and not value:
            raise AirflowException(f"La variable {var_name} no está configurada correctamente.")
        return value
    
    @task(task_id='get_connection_parameters')
    def get_connection_parameters():
        """Obtiene y verifica los parámetros de conexión desde Airflow Variables."""
        return {
            'aws_access_key': get_variable("MINIO_ACCESS_KEY"),
            'aws_secret_access_key': get_variable("MINIO_SECRET_ACCESS_KEY"),
            'mlflow_s3_endpoint_url': get_variable("MLFLOW_S3_ENDPOINT_URL"),
            'tracking_uri': get_variable("MLFLOW_TRACKING_URI")
        }
        print("Print the first step")
        logging.info("Parámetros de conexión obtenidos y verificados correctamente.")
    
    @task(task_id='test_mlflow_connection')
    def test_mlflow_connection(connection_params):
        """Realiza una prueba de conexión con MLflow utilizando los parámetros proporcionados."""
        try:
            mlflow.set_tracking_uri(connection_params['tracking_uri'])
            
            experiment_name = "test_connection_airflow"
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run() as run:
                test_param_value = random.randint(1, 100)
                mlflow.log_param("test_param", test_param_value)
                
                test_metric_value = random.random()
                mlflow.log_metric("test_metric", test_metric_value)
                
                artifact_path = "/tmp/test_artifact.txt"
                with open(artifact_path, "w") as f:
                    f.write(f"Este es un artefacto de prueba creado por Airflow en {datetime.now()}")
                mlflow.log_artifact(artifact_path)
                logging.info(f"Artefacto registrado en MLflow: {artifact_path}")
                os.remove(artifact_path)
                
                run_id = run.info.run_id
                logging.info(f"Ejecución de MLflow completada con éxito. Run ID: {run_id}")
                
                return {
                    "run_id": run_id,
                    "experiment_name": experiment_name,
                    "test_param": test_param_value,
                    "test_metric": test_metric_value
                }
                
        except Exception as e:
            logging.error(f"Error en la conexión con MLflow: {str(e)}")
            raise AirflowException(f"Error en conexión MLflow: {str(e)}")
    
    @task(task_id='validate_results')
    def validate_results(mlflow_results):
        """Valida los resultados de la ejecución de MLflow."""
        if not mlflow_results.get("run_id"):
            raise AirflowException("No se generó un run_id válido en MLflow")
            
        logging.info(f"Resultados de la prueba MLflow validados correctamente:")
        logging.info(f"- Experimento: {mlflow_results['experiment_name']}")
        logging.info(f"- Run ID: {mlflow_results['run_id']}")
        logging.info(f"- Parámetro de prueba: {mlflow_results['test_param']}")
        logging.info(f"- Métrica de prueba: {mlflow_results['test_metric']}")
        
        return True
    
    connection_params = get_connection_parameters()
    mlflow_results = test_mlflow_connection(connection_params)
    validation = validate_results(mlflow_results)
    
    
    connection_params >> mlflow_results >> validation


mlflow_dag = mlflow_connection_dag()