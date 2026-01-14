"""
Data Ingestion Module
---------------------
This module handles fetching and storing raw data from various sources.
"""

import os
import pandas as pd
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataIngestion:
    """
    Handles data ingestion from various sources.
    
    Supported sources:
    - Local CSV files
    - URLs (remote CSV)
    - (Future: Databases, APIs)
    """
    
    def __init__(self, raw_data_path: str = "data/raw"):
        """
        Initialize DataIngestion.
        
        Args:
            raw_data_path: Directory to store raw data
        """
        self.raw_data_path = raw_data_path
        os.makedirs(self.raw_data_path, exist_ok=True)
        logger.info(f"DataIngestion initialized. Raw data path: {self.raw_data_path}")
    
    def ingest_csv(self, source: str, filename: str = None) -> pd.DataFrame:
        """
        Ingest data from a CSV file or URL.
        
        Args:
            source: Path to CSV file or URL
            filename: Name to save the file as (optional)
        
        Returns:
            pd.DataFrame: Loaded data
        """
        logger.info(f"Starting data ingestion from: {source}")
        
        try:
            # Read the data
            data = pd.read_csv(source)
            logger.info(f"Data loaded successfully. Shape: {data.shape}")
            
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"data_{timestamp}.csv"
            
            # Save to raw data folder
            save_path = os.path.join(self.raw_data_path, filename)
            data.to_csv(save_path, index=False)
            logger.info(f"Raw data saved to: {save_path}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error during data ingestion: {str(e)}")
            raise
    
    def get_data_info(self, data: pd.DataFrame) -> dict:
        """
        Get basic information about the ingested data.
        
        Args:
            data: DataFrame to analyze
        
        Returns:
            dict: Data information
        """
        info = {
            "rows": len(data),
            "columns": len(data.columns),
            "column_names": list(data.columns),
            "missing_values": data.isnull().sum().to_dict(),
            "dtypes": data.dtypes.astype(str).to_dict()
        }
        logger.info(f"Data info: {info['rows']} rows, {info['columns']} columns")
        return info


if __name__ == "__main__":
    # Test the module
    ingestion = DataIngestion()
    print("DataIngestion module loaded successfully!")
