"""
Model Training Module
---------------------
This module handles model training and saving.
"""

import os
import joblib
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger(__name__)


class ModelTrainer:
    """
    Handles model training for classification tasks.
    
    Supported models:
    - Random Forest
    - Logistic Regression
    - Decision Tree
    - Gradient Boosting
    """
    
    MODELS = {
        "random_forest": RandomForestClassifier,
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "gradient_boosting": GradientBoostingClassifier
    }
    
    def __init__(self, artifacts_path: str = "artifacts"):
        """
        Initialize ModelTrainer.
        
        Args:
            artifacts_path: Directory to save trained models
        """
        self.artifacts_path = artifacts_path
        os.makedirs(self.artifacts_path, exist_ok=True)
        
        self.model = None
        self.model_name = None
        self.training_info = {}
        
        logger.info("ModelTrainer initialized")
    
    def get_model(self, model_name: str, **params):
        """
        Get a model instance by name.
        
        Args:
            model_name: Name of the model
            **params: Model hyperparameters
        
        Returns:
            Model instance
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODELS.keys())}")
        
        model_class = self.MODELS[model_name]
        
        # Default parameters
        default_params = {
            "random_forest": {"n_estimators": 100, "random_state": 42},
            "logistic_regression": {"random_state": 42, "max_iter": 1000},
            "decision_tree": {"random_state": 42},
            "gradient_boosting": {"n_estimators": 100, "random_state": 42}
        }
        
        # Merge default with provided params
        final_params = {**default_params.get(model_name, {}), **params}
        
        return model_class(**final_params)
    
    def train(self, X_train, y_train, model_name: str = "random_forest", **params):
        """
        Train a model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_name: Name of model to train
            **params: Model hyperparameters
        
        Returns:
            Trained model
        """
        logger.info(f"Starting training: {model_name}")
        
        self.model_name = model_name
        self.model = self.get_model(model_name, **params)
        
        # Record training start
        start_time = datetime.now()
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Record training info
        end_time = datetime.now()
        self.training_info = {
            "model_name": model_name,
            "training_samples": len(X_train),
            "features": X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train.columns),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "parameters": self.model.get_params()
        }
        
        logger.info(f"✅ Training completed in {self.training_info['duration_seconds']:.2f} seconds")
        
        return self.model
    
    def save_model(self, filename: str = None):
        """
        Save the trained model to disk.
        
        Args:
            filename: Name for the model file
        
        Returns:
            str: Path to saved model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first!")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_{self.model_name}_{timestamp}.pkl"
        
        model_path = os.path.join(self.artifacts_path, filename)
        
        # Save model and training info
        save_data = {
            "model": self.model,
            "training_info": self.training_info
        }
        joblib.dump(save_data, model_path)
        
        logger.info(f"Model saved to: {model_path}")
        
        return model_path
    
    def load_model(self, model_path: str):
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the model file
        
        Returns:
            Loaded model
        """
        logger.info(f"Loading model from: {model_path}")
        
        save_data = joblib.load(model_path)
        self.model = save_data["model"]
        self.training_info = save_data.get("training_info", {})
        
        logger.info("Model loaded successfully")
        
        return self.model
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict
        
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first!")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X: Features to predict
        
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first!")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError(f"Model {self.model_name} doesn't support predict_proba")


if __name__ == "__main__":
    # Test the module
    trainer = ModelTrainer()
    print("ModelTrainer module loaded successfully!")
    print(f"Available models: {list(trainer.MODELS.keys())}")
