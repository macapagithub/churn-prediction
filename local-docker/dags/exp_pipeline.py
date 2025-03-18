from airflow import DAG
from airflow.operators.python_operator import PythonOperator

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from matplotlib import pyplot as plt
import seaborn as sns

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
#from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from sklearn.preprocessing import (StandardScaler, OneHotEncoder)
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv() 

default_args = {
    'owner': 'data_science_team',
    'depends_on_past': False,
    'email': ['test@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 3, 10),
}


#Configuration mlflow
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MLFLOW_EXPERIMENT_NAME = 'churn_prediction'
MODEL_REGISTRY_NAME = "churn_xgboost_model"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

dag = DAG(
    'churn_prediction_pipeline',
    default_args=default_args,
    description='Churn prediction pipeline with mlflow',
    schedule_interval=timedelta(days=1),
    tags = ['churn', 'machine-learning', 'xgboost', 'mlflow']
)

def load_data(path_data: str, **kwargs) -> pd.DataFrame:
    """
    Load the data from the bucket.
    """
    ti = kwargs['ti']
    ##################
    # path_data = Variable.get("churn_data_path", default_var="/path/to/churn_data.csv") #Other way to get the path
    
    try:
        # Cargar los datos
        df = pd.read_csv(path_data, sep=';', decimal=",")
        print(f"Uploaded data: {df.shape[0]} rows and {df.shape[1]} columns")
        
        missing_values = df.isnull().sum().sum()
        print(f"Missing values in the dataset: {missing_values}")
        
        required_columns = ['churn', 'infobase']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        df.to_parquet('/tmp/churn_data_validated.parquet', index=False)
        
        # Registrar métricas básicas
        ti.xcom_push(key='dataset_shape', value=df.shape)
        ti.xcom_push(key='missing_values', value=missing_values)
        
        return True
        
    except Exception as e:
        print(f"Error al cargar datos: {str(e)}")
        raise



def preprocess_data(**kwargs):
    """
    Divide the data into training and test sets, and identifies the numerical and categorical features.
    """

    ti = kwargs['ti']
    df = pd.read_parquet('/tmp/churn_data_validated.parquet')
    
    X = df.drop(columns=['churn', 'infobase'])
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

    num_features = [var for var in X_train.columns if X_train[var].dtypes != object]
    cat_features = [var for var in X_train.columns if X_train[var].dtypes == object]
    #Other way to get the variables
    # num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    # cat_features = X_train.select_dtypes(include=[object]).columns.tolist()

    X_train.to_parquet('/tmp/X_train.parquet', index=False)
    X_test.to_parquet('/tmp/X_test.parquet', index=False)
    y_train.to_parquet('/tmp/y_train.parquet', index=False)
    y_test.to_parquet('/tmp/y_test.parquet', index=False)

    pd.Series(num_features).to_csv('/tmp/num_variables.csv', index=False)
    pd.Series(cat_features).to_csv('/tmp/cat_variables.csv', index=False)
    
    ti.xcom_push(key='num_features', value=num_features)
    ti.xcom_push(key='cat_features', value=cat_features)
    ti.xcom_push(key='train_shape', value=X_train.shape)
    
    return True


def create_pipeline(**kwargs):
    """
    Create the pipeline with the preprocessing steps and the model.
    """

    ti = kwargs['ti']
    X_train = pd.read_parquet('/tmp/X_train.parquet')
    y_train = pd.read_parquet('/tmp/y_train.parquet')
    num_features = pd.read_csv('/tmp/num_variables.csv')['0'].tolist()
    cat_features = pd.read_csv('/tmp/cat_variables.csv')['0'].tolist()
    
    # Preprocessing
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    transformation = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ]
    )

    pd.Series({
        'num_features': str(num_features),
        'cat_features': str(cat_features)
    }).to_csv('/tmp/transformation_info.csv', index=False)
    
    return True


def train_model(**kwargs):
    """
    Train base model and optimize hyperparameters.
    Model and metrics are logged in mlflow.
    """ 

    ti = kwargs['ti']

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X_train = pd.read_parquet('/tmp/X_train.parquet')
    y_train = pd.read_parquet('/tmp/y_train.parquet')
    X_test = pd.read_parquet('/tmp/X_test.parquet')
    y_test = pd.read_parquet('/tmp/y_test.parquet')

    num_features = pd.read_csv('/tmp/num_variables.csv')['0'].tolist()
    cat_features = pd.read_csv('/tmp/cat_variables.csv')['0'].tolist()
    transformation_info = pd.read_csv('/tmp/transformation_info.csv')

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    transformation = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ]
    )


    with mlflow.start_run(run_name='xgboost_opt') as run:
        
        mlflow.set_tag('model', 'XGBoost')
        mlflow.set_tag('task', 'classification')
        mlflow.set_tag('dataset', 'churn')

        model = XGBClassifier(random_state=12)
        
        classifier_xgb = Pipeline(steps=[
            ('preprocessor', transformation),
            ('classifier', model)
        ])

        param_dist = {
            'classifier__n_estimators': np.arange(50, 500, 50),
            'classifier__max_depth': np.arange(3, 10),
            'classifier__min_child_weight': np.arange(1, 7),
            'classifier__learning_rate': np.linspace(0.01, 0.5, 10),
            'classifier__subsample': np.linspace(0.5, 1, 10),
            'classifier__colsample_bytree': np.linspace(0.5, 1, 10),
            'classifier__gamma': np.linspace(0, 1, 10),
        }

        mlflow.log_params(param_dist)

        search = RandomizedSearchCV(
            classifier_xgb,
            param_distributions=param_dist,
            n_iter=30,
            scoring='f1',
            cv=3,
            n_jobs=-1,
            random_state=12,
            verbose=1,
            return_train_score=True
        )

        search.fit(X_train, y_train)

        best_params = search.best_params_
        best_estimator = search.best_estimator_
        best_score = search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric('f1', best_score)
        
        y_pred = best_estimator.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        

        mlflow.log_metric('f1_test', f1)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('confusion_matrix', confusion)
        mlflow.log_metric('classification_report', report)

        confusion = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 8))
        sns.heatmap(confusion, annot=True, fmt="d", cmap='Blues', cbar=False)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion matrix')
        plt.savefig('/tmp/confusion_matrix.png')
        
        mlflow.log_artifact('/tmp/confusion_matrix.png')
        plt.close()

        
        
        report = classification_report(y_test, y_pred)
        with open('/tmp/classification_report.txt', 'w') as f:
            f.write(report)
        mlflow.log_artifact('/tmp/classification_report.txt')

        mlflow.sklearn.log_model(best_estimator, 'model', MODEL_REGISTRY_NAME)

        ti.xcom_push(key= 'run_id', value=run.info.run_id)
        ti.xcom_push(key= 'best_params', value=best_params)
        ti.xcom_push(key= 'metrics', value={'f1': f1, 'accuracy': accuracy})

        import joblib
        joblib.dump(best_estimator, '/tmp/best_model.joblib')

    return True

def evaluate_model(**kwargs):
    """
    Evaluate the model with the test data.
    """

    ti = kwargs['ti']
    run_id = ti.xcom_pull(task_ids='train_model', key='run_id')
    best_params = ti.xcom_pull(task_ids='train_model', key='best_params')
    metrics = ti.xcom_pull(task_ids='train_model', key='metrics')

    import joblib
    best_model = joblib.load('/tmp/best_model.joblib')

    X_test = pd.read_parquet('/tmp/X_test.parquet')
    y_test = pd.read_parquet('/tmp/y_test.parquet')

    with mlflow.start_run(run_id=run_id, nested=True) as run:
        y_pred_test = best_model.predict(X_test)
        test_f1 = f1_score(y_test, y_pred_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        mlflow.log_metric('test_f1_test', test_f1)
        mlflow.log_metric('test_accuracy', test_accuracy)

        test_confusion = confusion_matrix(y_test, y_pred_test)

        plt.figure(figsize=(8, 8))
        sns.heatmap(test_confusion, annot=True, fmt="d", cmap='Blues', cbar=False)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion matrix (Test)')
        plt.savefig('/tmp/test_confusion_matrix.png')
        
        mlflow.log_artifact('/tmp/test_confusion_matrix.png')
        plt.close()

        test_report = classification_report(y_test, y_pred_test)
        with open('/tmp/test_classification_report.txt', 'w') as f:
            f.write(test_report)
        
        mlflow.log_artifact('/tmp/test_classification_report.txt', 'test_classification_report')

        ti.xcom_push(key='test_metrics', value={'test_f1_score': test_f1, 'test_accuracy': test_accuracy})
        
        print(f"Results for the test data: \nF1: {test_f1:.4f} \nAccuracy: {test_accuracy:.4f}")

    return True


data_ingestion = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    op_args=['/path/to/churn_data.csv'],
    provide_context=True,
    dag=dag
)

data_preprocessing = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    provide_context=True,
    dag=dag
)

with TaskGroup(group_id='training_optimization', dag=dag) as training_group:
    train_task = PythonOperator(
        task_id='training_opt',
        python_callable=train_model,
        provide_context=True
    )

test_evaluation = PythonOperator(
    task_id='evaluate_on_test',
    python_callable=evaluate_model,
    provide_context=True,
    dag=dag
)

data_ingestion >> data_preprocessing >> training_group >> test_evaluation
