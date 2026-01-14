"""
Tests for Data Pipeline Modules
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.ingestion import DataIngestion
from src.data.validation import DataValidation
from src.data.preprocessing import DataPreprocessing


class TestDataIngestion:
    """Tests for DataIngestion class."""
    
    def test_init(self, tmp_path):
        """Test DataIngestion initialization."""
        ingestion = DataIngestion(raw_data_path=str(tmp_path))
        assert os.path.exists(tmp_path)
    
    def test_get_data_info(self):
        """Test data info extraction."""
        ingestion = DataIngestion()
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z']
        })
        info = ingestion.get_data_info(data)
        
        assert info['rows'] == 3
        assert info['columns'] == 2
        assert 'a' in info['column_names']


class TestDataValidation:
    """Tests for DataValidation class."""
    
    def test_validate_columns_pass(self):
        """Test column validation - pass case."""
        validator = DataValidation()
        data = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        
        result = validator.validate_columns(data, ['a', 'b'])
        assert result == True
    
    def test_validate_columns_fail(self):
        """Test column validation - fail case."""
        validator = DataValidation()
        data = pd.DataFrame({'a': [1], 'b': [2]})
        
        result = validator.validate_columns(data, ['a', 'b', 'missing'])
        assert result == False
    
    def test_validate_missing_values(self):
        """Test missing value validation."""
        validator = DataValidation()
        data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, None, 6]})
        
        # Should pass with default 30% threshold
        result = validator.validate_missing_values(data, threshold=0.5)
        assert result == True


class TestDataPreprocessing:
    """Tests for DataPreprocessing class."""
    
    def test_handle_missing_values(self, tmp_path):
        """Test missing value imputation."""
        preprocessor = DataPreprocessing(
            processed_data_path=str(tmp_path / "processed"),
            artifacts_path=str(tmp_path / "artifacts")
        )
        
        data = pd.DataFrame({
            'num': [1.0, 2.0, None, 4.0],
            'cat': ['a', None, 'b', 'a']
        })
        
        result = preprocessor.handle_missing_values(data)
        
        assert result.isnull().sum().sum() == 0
    
    def test_encode_categorical(self, tmp_path):
        """Test categorical encoding."""
        preprocessor = DataPreprocessing(
            processed_data_path=str(tmp_path / "processed"),
            artifacts_path=str(tmp_path / "artifacts")
        )
        
        data = pd.DataFrame({'cat': ['a', 'b', 'a', 'c']})
        result = preprocessor.encode_categorical(data)
        
        assert result['cat'].dtype in [np.int32, np.int64]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
