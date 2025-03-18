import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import joblib
from exp_pipeline import load_data, preprocess_data, create_pipeline, train_model, evaluate_model

class TestExpPipeline(unittest.TestCase):

    @patch('exp_pipeline.pd.read_csv')
    @patch('exp_pipeline.pd.DataFrame.to_parquet')
    def test_load_data(self, mock_to_parquet, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            'churn': [0, 1],
            'infobase': [1, 2],
            'feature1': [10, 20]
        })
        mock_ti = MagicMock()
        kwargs = {'ti': mock_ti}
        result = load_data('/path/to/churn_data.csv', **kwargs)
        self.assertTrue(result)
        mock_to_parquet.assert_called_once()

    @patch('exp_pipeline.pd.read_parquet')
    @patch('exp_pipeline.pd.Series.to_csv')
    def test_preprocess_data(self, mock_to_csv, mock_read_parquet):
        mock_read_parquet.return_value = pd.DataFrame({
            'churn': [0, 1],
            'infobase': [1, 2],
            'feature1': [10, 20],
            'feature2': ['A', 'B']
        })
        mock_ti = MagicMock()
        kwargs = {'ti': mock_ti}
        result = preprocess_data(**kwargs)
        self.assertTrue(result)
        mock_to_csv.assert_called()

    @patch('exp_pipeline.pd.read_parquet')
    @patch('exp_pipeline.pd.Series.to_csv')
    def test_create_pipeline(self, mock_to_csv, mock_read_parquet):
        mock_read_parquet.return_value = pd.DataFrame({
            'feature1': [10, 20],
            'feature2': ['A', 'B']
        })
        mock_ti = MagicMock()
        kwargs = {'ti': mock_ti}
        result = create_pipeline(**kwargs)
        self.assertTrue(result)
        mock_to_csv.assert_called()

    @patch('exp_pipeline.mlflow.start_run')
    @patch('exp_pipeline.RandomizedSearchCV.fit')
    @patch('exp_pipeline.pd.read_parquet')
    @patch('exp_pipeline.pd.read_csv')
    @patch('exp_pipeline.joblib.dump')
    def test_train_model(self, mock_dump, mock_read_csv, mock_read_parquet, mock_fit, mock_start_run):
        mock_read_parquet.return_value = pd.DataFrame({
            'feature1': [10, 20],
            'feature2': ['A', 'B']
        })
        mock_read_csv.return_value = pd.DataFrame({'0': ['feature1', 'feature2']})
        mock_fit.return_value = MagicMock(best_params_={}, best_estimator_=MagicMock(), best_score_=0.9)
        mock_ti = MagicMock()
        kwargs = {'ti': mock_ti}
        result = train_model(**kwargs)
        self.assertTrue(result)
        mock_dump.assert_called_once()

    @patch('exp_pipeline.mlflow.start_run')
    @patch('exp_pipeline.joblib.load')
    @patch('exp_pipeline.pd.read_parquet')
    def test_evaluate_model(self, mock_read_parquet, mock_load, mock_start_run):
        mock_read_parquet.return_value = pd.DataFrame({
            'feature1': [10, 20],
            'feature2': ['A', 'B']
        })
        mock_load.return_value = MagicMock(predict=MagicMock(return_value=[0, 1]))
        mock_ti = MagicMock()
        kwargs = {'ti': mock_ti}
        result = evaluate_model(**kwargs)
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
