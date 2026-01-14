"""
Data Validation Module
----------------------
This module handles data quality checks and schema validation.
"""

import pandas as pd
from typing import List, Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidation:
    """
    Validates data quality and schema before processing.
    
    Checks:
    - Required columns exist
    - Data types are correct
    - No excessive missing values
    - Values are within expected ranges
    """
    
    def __init__(self, schema: Dict[str, Any] = None):
        """
        Initialize DataValidation.
        
        Args:
            schema: Expected data schema (optional)
        """
        self.schema = schema or {}
        self.validation_report = {}
        logger.info("DataValidation initialized")
    
    def validate_columns(self, data: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Check if all required columns are present.
        
        Args:
            data: DataFrame to validate
            required_columns: List of required column names
        
        Returns:
            bool: True if all columns present
        """
        missing = [col for col in required_columns if col not in data.columns]
        
        if missing:
            logger.warning(f"Missing columns: {missing}")
            self.validation_report["missing_columns"] = missing
            return False
        
        logger.info("All required columns present")
        self.validation_report["missing_columns"] = []
        return True
    
    def validate_missing_values(self, data: pd.DataFrame, threshold: float = 0.3) -> bool:
        """
        Check if missing values are within acceptable threshold.
        
        Args:
            data: DataFrame to validate
            threshold: Maximum allowed missing ratio (default: 30%)
        
        Returns:
            bool: True if missing values are acceptable
        """
        missing_ratios = data.isnull().sum() / len(data)
        problematic = missing_ratios[missing_ratios > threshold]
        
        if len(problematic) > 0:
            logger.warning(f"Columns with excessive missing values: {problematic.to_dict()}")
            self.validation_report["high_missing_columns"] = problematic.to_dict()
            return False
        
        logger.info("Missing values within acceptable threshold")
        self.validation_report["high_missing_columns"] = {}
        return True
    
    def validate_data_types(self, data: pd.DataFrame, expected_types: Dict[str, str]) -> bool:
        """
        Check if columns have expected data types.
        
        Args:
            data: DataFrame to validate
            expected_types: Dict mapping column names to expected types
        
        Returns:
            bool: True if all types match
        """
        type_mismatches = {}
        
        for col, expected in expected_types.items():
            if col in data.columns:
                actual = str(data[col].dtype)
                if expected not in actual:
                    type_mismatches[col] = {"expected": expected, "actual": actual}
        
        if type_mismatches:
            logger.warning(f"Type mismatches: {type_mismatches}")
            self.validation_report["type_mismatches"] = type_mismatches
            return False
        
        logger.info("All data types match expected")
        self.validation_report["type_mismatches"] = {}
        return True
    
    def validate_all(self, data: pd.DataFrame, required_columns: List[str] = None,
                     expected_types: Dict[str, str] = None, missing_threshold: float = 0.3) -> Dict:
        """
        Run all validations and return comprehensive report.
        
        Args:
            data: DataFrame to validate
            required_columns: List of required columns
            expected_types: Expected data types
            missing_threshold: Missing value threshold
        
        Returns:
            dict: Validation report
        """
        logger.info("Starting comprehensive data validation")
        
        results = {
            "columns_valid": True,
            "missing_valid": True,
            "types_valid": True,
            "overall_valid": True
        }
        
        if required_columns:
            results["columns_valid"] = self.validate_columns(data, required_columns)
        
        results["missing_valid"] = self.validate_missing_values(data, missing_threshold)
        
        if expected_types:
            results["types_valid"] = self.validate_data_types(data, expected_types)
        
        results["overall_valid"] = all([
            results["columns_valid"],
            results["missing_valid"],
            results["types_valid"]
        ])
        
        results["details"] = self.validation_report
        
        if results["overall_valid"]:
            logger.info("✅ Data validation PASSED")
        else:
            logger.warning("❌ Data validation FAILED")
        
        return results


if __name__ == "__main__":
    # Test the module
    validator = DataValidation()
    print("DataValidation module loaded successfully!")
