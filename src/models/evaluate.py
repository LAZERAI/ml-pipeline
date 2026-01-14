"""
Model Evaluation Module
-----------------------
This module handles model evaluation and metrics calculation.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluates model performance using various metrics.
    
    Metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - Confusion Matrix
    - ROC-AUC (for binary classification)
    """
    
    def __init__(self, artifacts_path: str = "artifacts"):
        """
        Initialize ModelEvaluator.
        
        Args:
            artifacts_path: Directory to save evaluation reports
        """
        self.artifacts_path = artifacts_path
        os.makedirs(self.artifacts_path, exist_ok=True)
        
        self.metrics = {}
        self.evaluation_report = {}
        
        logger.info("ModelEvaluator initialized")
    
    def calculate_metrics(self, y_true, y_pred, y_proba=None) -> dict:
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
        
        Returns:
            dict: All calculated metrics
        """
        logger.info("Calculating evaluation metrics")
        
        # Convert to numpy arrays if needed
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred).ravel()
        
        # Determine if binary or multiclass
        n_classes = len(np.unique(y_true))
        is_binary = n_classes == 2
        
        # Calculate metrics
        self.metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "n_classes": n_classes,
            "samples_evaluated": len(y_true)
        }
        
        # ROC-AUC for binary classification
        if is_binary and y_proba is not None:
            try:
                if len(y_proba.shape) > 1:
                    y_proba = y_proba[:, 1]
                self.metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
                self.metrics["roc_auc"] = None
        
        logger.info(f"Metrics calculated: Accuracy={self.metrics['accuracy']:.4f}, F1={self.metrics['f1_score']:.4f}")
        
        return self.metrics
    
    def generate_report(self, y_true, y_pred, model_name: str = "model") -> dict:
        """
        Generate comprehensive evaluation report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the evaluated model
        
        Returns:
            dict: Evaluation report
        """
        logger.info(f"Generating evaluation report for: {model_name}")
        
        # Get classification report as dict
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        self.evaluation_report = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics,
            "classification_report": class_report
        }
        
        return self.evaluation_report
    
    def save_report(self, filename: str = None):
        """
        Save evaluation report to file.
        
        Args:
            filename: Name for the report file
        
        Returns:
            str: Path to saved report
        """
        if not self.evaluation_report:
            raise ValueError("No report to save. Generate a report first!")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{timestamp}.json"
        
        report_path = os.path.join(self.artifacts_path, filename)
        
        with open(report_path, 'w') as f:
            json.dump(self.evaluation_report, f, indent=2)
        
        logger.info(f"Report saved to: {report_path}")
        
        return report_path
    
    def print_summary(self):
        """Print a summary of the evaluation metrics."""
        
        if not self.metrics:
            print("No metrics calculated yet.")
            return
        
        print("\n" + "="*50)
        print("           MODEL EVALUATION SUMMARY")
        print("="*50)
        print(f"  Accuracy:   {self.metrics['accuracy']:.4f} ({self.metrics['accuracy']*100:.2f}%)")
        print(f"  Precision:  {self.metrics['precision']:.4f}")
        print(f"  Recall:     {self.metrics['recall']:.4f}")
        print(f"  F1 Score:   {self.metrics['f1_score']:.4f}")
        if self.metrics.get('roc_auc'):
            print(f"  ROC-AUC:    {self.metrics['roc_auc']:.4f}")
        print(f"  Samples:    {self.metrics['samples_evaluated']}")
        print("="*50)
        print("\nConfusion Matrix:")
        cm = np.array(self.metrics['confusion_matrix'])
        print(cm)
        print("="*50 + "\n")
    
    def evaluate(self, model, X_test, y_test, model_name: str = "model") -> dict:
        """
        Complete evaluation pipeline.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
        
        Returns:
            dict: Complete evaluation report
        """
        logger.info(f"Starting evaluation for: {model_name}")
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        y_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
            except:
                pass
        
        # Calculate metrics
        self.calculate_metrics(y_test, y_pred, y_proba)
        
        # Generate report
        report = self.generate_report(y_test, y_pred, model_name)
        
        # Save report
        self.save_report()
        
        # Print summary
        self.print_summary()
        
        logger.info("✅ Evaluation completed")
        
        return report


if __name__ == "__main__":
    # Test the module
    evaluator = ModelEvaluator()
    print("ModelEvaluator module loaded successfully!")
