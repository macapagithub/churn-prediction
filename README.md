# Churn Prediction Project with XGBoost Airflow and MLflow

This project implements a machine learning pipeline to predict customer churn using Apache Airflow for orchestration and MLflow for experiment tracking and model management.

## Project Description

The main goal of this project is to build a robust predictive model that can identify customers with a high probability of leaving a service. The pipeline includes the following stages:

1.  **Data Ingestion:** Loading customer data from a specified source.
2.  **Data Preprocessing:** Dividing the data into training and test sets, and identifying numerical and categorical features.
3.  **Model Training and Optimization:** Training an XGBoost model and optimizing its hyperparameters using RandomizedSearchCV. Experiments are tracked with MLflow.
4.  **Model Evaluation:** Evaluating the performance of the best model on the test set, calculating metrics like F1-score and accuracy.

## Prerequisites

Before you begin, ensure you have the following installed:

* **Docker:** Required to run the service containers (Airflow, PostgreSQL, Redis, MinIO, MLflow).
* **Docker Compose:** Used to define and manage the multi-container application.
* **Python 3.x:** To run scripts or interact with the development environment (optional, as everything runs in containers).

## Installation and Setup

1.  **Clone the Repository (Optional):** If this project is in a repository, clone it to your local machine.
2.  **Configure Environment Variables:**
    * Create a `.env` file in the project root.
    * Define the following necessary environment variables. Make sure to replace the example values with your own configurations.

        ```env
        POSTGRES_PASSWORD=your_postgres_password
        AIRFLOW_WWW_USER_PASSWORD=your_airflow_password
        EMAIL_ADMIN=[email address removed] 
        AIRFLOW_UID=50000 # Or the UID you want for the Airflow user
        MINIO_ROOT_USER=minioadmin
        MINIO_ROOT_PASSWORD=minioadmin
        MINIO_ACCESS_KEY=minio access key
        MINIO_SECRET_ACCESS_KEY=minio secret key
        MINIO_PORT=9000
        MINIO_BUCKET_NAME=mlflow # Name of the bucket for MLflow artifacts
        POSTGRES_PASSWORD_MLFLOW=mlflow_password # Password for the MLflow database (if enabled)
        MLFLOW_TRACKING_URI=http://tracking_server:5000
        ```

3.  **Start Services with Docker Compose:**
    Navigate to the project root directory and run the command:

    ```bash
    cd local-docker
    ```
    
    where the `docker-compose.yml` file is located and run the following command:

    ```bash
    docker-compose up -d --build
    ```

    This command will build the necessary images and start the containers in the background.

## Usage

Once the services are running, you can interact with the project as follows:

1.  **Access Airflow UI:** Open your web browser and go to `http://localhost:8080`. Log in with the credentials configured in the environment variables (`AIRFLOW_WWW_USER_PASSWORD`, the user is `airflow`).

2.  **Access MLflow UI:** Open your web browser and go to `http://localhost:5000`. Here you can view experiments, runs, and registered models.

3.  **Access MinIO Console:** Open your web browser and go to `http://localhost:9001`. Log in with the credentials configured in the environment variables (`MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`). Here you can view the buckets created, in particular for MLflow.

4.  **Load Data:** Ensure that the churn data file (`churn_data.csv`) is located at the path specified in the DAG (`/path/to/churn_data.csv`). You can mount a volume or modify the path in the DAG as needed. The DAG expects a semicolon-separated CSV file with a comma as the decimal separator and containing at least the columns `churn` and `infobase`.

5.  **Activate the DAG:** In the Airflow UI, find the DAG named `churn_prediction_pipeline` and turn it on. You can trigger a manual run by clicking the "Trigger DAG" button.

6.  **Monitor Execution:** Observe the progress of the DAG in the Airflow UI. Each task in the pipeline will execute sequentially.

7.  **View Results in MLflow:** Once the DAG completes, go to the MLflow UI to see the details of the run, including:
    * Optimized model parameters.
    * Performance metrics (F1-score, accuracy).
    * Artifacts such as the confusion matrix and classification report.
    * The registered XGBoost model.

## MLflow Integration

This project utilizes MLflow for:

* **Experiment Tracking:** Recording the parameters, metrics, and artifacts of each model training run.
* **Model Management:** Saving the best trained model in the MLflow Model Registry.

The `MLFLOW_TRACKING_URI` environment variable in the `docker-compose.yml` file configures the connection between Airflow and the MLflow tracking server.

## Airflow DAG Description (`churn_prediction_dag.py`)

The `churn_prediction_pipeline` DAG defines the workflow for churn prediction. Its main tasks are:

* **`load_data`:** Loads the data from the specified path.
* **`preprocess_data`:** Splits the data and prepares the features.
* **`training_optimization.training_opt`:** Trains and optimizes the XGBoost model using `RandomizedSearchCV` and logs the results in MLflow.
* **`evaluate_on_test`:** Evaluates the best model on the test set and logs the metrics in MLflow.

## Important Environment Variables

The following environment variables configured in the `.env` file are crucial for the project's operation:

* `POSTGRES_PASSWORD`: Password for the Airflow PostgreSQL database.
* `AIRFLOW_WWW_USER_PASSWORD`: Password for the Airflow user interface.
* `EMAIL_ADMIN`: Email address of the Airflow administrator.
* `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD`: Credentials for the MinIO console.
* `MINIO_ACCESS_KEY` and `MINIO_SECRET_ACCESS_KEY`: Credentials to access the MinIO bucket.
* `MLFLOW_TRACKING_URI`: URL of the MLflow tracking server (`localhost:5000`).

## Potential Improvements

* Implement a more robust data validation stage.
* Configure alerts and notifications in Airflow for pipeline failures.
* Automate the registration and deployment of the best model to a production environment.
* Explore different classification models and compare their performance.
* Implement unit and integration tests for the DAG and tasks.
* Use a secrets management system for credentials in a production environment.
* Monitoring.

Thank you for reviewing this project!