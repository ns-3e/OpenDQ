import logging
from typing import Dict, List, Union, Optional

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
import pyspark.sql.functions as F

# Configure logging
logger = logging.getLogger(__name__)

class DataQuality:
    """Class for performing data quality checks on DataFrames."""
    
    def __init__(self, data: Union[pd.DataFrame, SparkDataFrame]):
        self.data = data
        self._is_spark = isinstance(data, SparkDataFrame)
        logger.info(f"Initializing DataQuality with {'Spark' if self._is_spark else 'Pandas'} DataFrame")
    
    def check_missing(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """
        Compute missing value statistics for each column.
        
        Returns:
            Dict containing for each column:
                - count: number of missing values
                - percentage: percentage of missing values
        """
        try:
            if self._is_spark:
                total_rows = self.data.count()
                missing_stats = {}
                
                for col in self.data.columns:
                    missing_count = self.data.filter(F.col(col).isNull()).count()
                    missing_stats[col] = {
                        'count': missing_count,
                        'percentage': (missing_count / total_rows) * 100
                    }
            else:
                missing_counts = self.data.isnull().sum()
                total_rows = len(self.data)
                
                missing_stats = {
                    col: {
                        'count': count,
                        'percentage': (count / total_rows) * 100
                    }
                    for col, count in missing_counts.items()
                }
            
            logger.info("Successfully computed missing value statistics")
            return missing_stats
            
        except Exception as e:
            logger.error(f"Error computing missing value statistics: {str(e)}")
            raise
    
    def check_duplicates(self, subset: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Identify duplicate rows in the DataFrame.
        
        Args:
            subset: Optional list of columns to consider for duplicates
            
        Returns:
            Dict containing:
                - total_duplicates: total number of duplicate rows
                - unique_records: number of unique records
        """
        try:
            if self._is_spark:
                if subset:
                    total_records = self.data.count()
                    unique_records = self.data.select(subset).distinct().count()
                else:
                    total_records = self.data.count()
                    unique_records = self.data.distinct().count()
                
                duplicate_count = total_records - unique_records
                
            else:
                total_records = len(self.data)
                unique_records = len(self.data.drop_duplicates(subset=subset))
                duplicate_count = total_records - unique_records
            
            results = {
                'total_duplicates': duplicate_count,
                'unique_records': unique_records,
                'duplicate_percentage': (duplicate_count / total_records) * 100
            }
            
            logger.info("Successfully computed duplicate statistics")
            return results
            
        except Exception as e:
            logger.error(f"Error computing duplicate statistics: {str(e)}")
            raise
    
    def validate_schema(self, expected_schema: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Validate DataFrame schema against expected schema.
        
        Args:
            expected_schema: Dict mapping column names to expected data types
            
        Returns:
            Dict containing:
                - missing_columns: columns in expected schema but not in DataFrame
                - extra_columns: columns in DataFrame but not in expected schema
                - type_mismatches: columns with mismatched data types
        """
        try:
            if self._is_spark:
                actual_schema = {field.name: field.dataType.simpleString() 
                               for field in self.data.schema.fields}
            else:
                actual_schema = self.data.dtypes.astype(str).to_dict()
            
            actual_columns = set(actual_schema.keys())
            expected_columns = set(expected_schema.keys())
            
            missing_columns = list(expected_columns - actual_columns)
            extra_columns = list(actual_columns - expected_columns)
            
            type_mismatches = []
            for col in actual_columns.intersection(expected_columns):
                if not self._types_match(actual_schema[col], expected_schema[col]):
                    type_mismatches.append(col)
            
            results = {
                'missing_columns': missing_columns,
                'extra_columns': extra_columns,
                'type_mismatches': type_mismatches,
                'is_valid': len(missing_columns) == 0 and len(type_mismatches) == 0
            }
            
            logger.info("Successfully validated schema")
            return results
            
        except Exception as e:
            logger.error(f"Error validating schema: {str(e)}")
            raise
    
    def _types_match(self, actual_type: str, expected_type: str) -> bool:
        """Helper method to compare data types across Pandas and Spark."""
        # Normalize type names for comparison
        actual_type = actual_type.lower()
        expected_type = expected_type.lower()
        
        # Map of equivalent types
        type_mappings = {
            'int': ['int', 'int64', 'integer', 'long'],
            'float': ['float', 'float64', 'double'],
            'str': ['str', 'string', 'text'],
            'bool': ['bool', 'boolean'],
            'datetime': ['datetime', 'timestamp', 'date']
        }
        
        # Find the normalized type category
        actual_category = next((category for category, types in type_mappings.items()
                              if any(t in actual_type for t in types)), actual_type)
        expected_category = next((category for category, types in type_mappings.items()
                                if any(t in expected_type for t in types)), expected_type)
        
        return actual_category == expected_category 