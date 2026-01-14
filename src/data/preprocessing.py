"""
Data Preprocessing Module
-------------------------
This module handles data transformation and feature engineering.
"""

import os
import pandas as pd
import numpy as np
import joblib
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessing:
    """
    Handles data preprocessing and feature engineering.
    
    Transformations:
    - Missing value imputation
    - Categorical encoding
    - Feature scaling
    - Train-test split
    """

    def __init__(self, processed_data_path: str = "data/processed",
                 artifacts_path: str = "artifacts"):
        self.processed_data_path = processed_data_path
        self.artifacts_path = artifacts_path
        
        os.makedirs(self.processed_data_path, exist_ok=True)
        os.makedirs(self.artifacts_path, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='mean')
        
        logger.info("DataPreprocessing initialized")

    def handle_missing_values(self, data: pd.DataFrame,
                              numerical_strategy: str = 'mean',
                              categorical_strategy: str = 'most_frequent') -> pd.DataFrame:
        """Handle missing values in the dataset."""
        logger.info("Handling missing values")
        
        data = data.copy()
        
        # Convert None to NaN for proper handling
        data = data.replace({None: np.nan})
        
        # Numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            num_imputer = SimpleImputer(strategy=numerical_strategy)
            data[numerical_cols] = num_imputer.fit_transform(data[numerical_cols])
        
        # Categorical columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy=categorical_strategy)
            data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])
        
        logger.info(f"Missing values handled. Numerical: {numerical_strategy}, Categorical: {categorical_strategy}")
        return data

    def encode_categorical(self, data: pd.DataFrame,
                           columns: List[str] = None) -> pd.DataFrame:
        """Encode categorical variables to numerical."""
        logger.info("Encoding categorical variables")
        
        data = data.copy()
        
        if columns is None:
            columns = data.select_dtypes(include=['object']).columns.tolist()
        
        for col in columns:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"Encoded column: {col}")
        
        return data

    def scale_features(self, data: pd.DataFrame,
                       columns: List[str] = None) -> pd.DataFrame:
        """Scale numerical features using StandardScaler."""
        logger.info("Scaling features")
        
        data = data.copy()
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(columns) > 0:
            data[columns] = self.scaler.fit_transform(data[columns])
            logger.info(f"Scaled {len(columns)} columns")
        
        return data

    def split_data(self, data: pd.DataFrame, target_column: str,
                   test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Split data into training and testing sets."""
        logger.info(f"Splitting data. Test size: {test_size}")
        
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        return X_train, X_test, y_train, y_test

    def save_preprocessors(self):
        """Save preprocessing objects for later use."""
        scaler_path = os.path.join(self.artifacts_path, "scaler.pkl")
        encoders_path = os.path.join(self.artifacts_path, "label_encoders.pkl")
        
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoders, encoders_path)
        
        logger.info(f"Preprocessors saved to {self.artifacts_path}")

    def preprocess_pipeline(self, data: pd.DataFrame, target_column: str,
                            test_size: float = 0.2) -> Tuple:
        """Run complete preprocessing pipeline."""
        logger.info("Starting preprocessing pipeline")
        
        # Step 1: Handle missing values
        data = self.handle_missing_values(data)
        
        # Step 2: Encode categorical (except target if it's categorical)
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)
        data = self.encode_categorical(data, categorical_cols)
        
        # Step 3: Encode target if categorical
        if data[target_column].dtype == 'object':
            le = LabelEncoder()
            data[target_column] = le.fit_transform(data[target_column])
            self.label_encoders['target'] = le
        
        # Step 4: Split data
        X_train, X_test, y_train, y_test = self.split_data(data, target_column, test_size)
        
        # Step 5: Scale features
        feature_cols = X_train.columns.tolist()
        X_train[feature_cols] = self.scaler.fit_transform(X_train[feature_cols])
        X_test[feature_cols] = self.scaler.transform(X_test[feature_cols])
        
        # Step 6: Save preprocessors
        self.save_preprocessors()
        
        # Step 7: Save processed data
        X_train.to_csv(os.path.join(self.processed_data_path, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(self.processed_data_path, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(self.processed_data_path, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(self.processed_data_path, "y_test.csv"), index=False)
        
        logger.info("[OK] Preprocessing pipeline completed")
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocessor = DataPreprocessing()
    print("DataPreprocessing module loaded successfully!")
