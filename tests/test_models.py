"""
Tests for Model Modules
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator


class TestModelTrainer:
    """Tests for ModelTrainer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = (X[:, 0] > 0).astype(int)
        return X, y
    
    def test_init(self, tmp_path):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(artifacts_path=str(tmp_path))
        assert os.path.exists(tmp_path)
    
    def test_get_model(self, tmp_path):
        """Test model retrieval."""
        trainer = ModelTrainer(artifacts_path=str(tmp_path))
        
        model = trainer.get_model("random_forest")
        assert model is not None
    
    def test_train(self, tmp_path, sample_data):
        """Test model training."""
        X, y = sample_data
        trainer = ModelTrainer(artifacts_path=str(tmp_path))
        
        model = trainer.train(X, y, model_name="random_forest")
        
        assert model is not None
        assert trainer.training_info is not None
    
    def test_predict(self, tmp_path, sample_data):
        """Test prediction."""
        X, y = sample_data
        trainer = ModelTrainer(artifacts_path=str(tmp_path))
        trainer.train(X, y)
        
        predictions = trainer.predict(X[:5])
        
        assert len(predictions) == 5


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""
    
    def test_calculate_metrics(self, tmp_path):
        """Test metric calculation."""
        evaluator = ModelEvaluator(artifacts_path=str(tmp_path))
        
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 0 <= metrics['accuracy'] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
