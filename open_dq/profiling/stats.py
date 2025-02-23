import pandas as pd
import numpy as np
from typing import Union, Dict, List

class BasicStats:
    """Compute basic statistics for each column in the dataset."""
    
    def __init__(self, data: Union[pd.DataFrame, 'pyspark.sql.DataFrame']):
        self.data = data
        self._is_spark = not isinstance(data, pd.DataFrame)
        
    def compute_stats(self) -> Dict:
        """
        Compute basic statistics for all columns.
        
        Returns:
            Dictionary containing statistics for each column
        """
        stats = {}
        
        if self._is_spark:
            return self._compute_spark_stats()
        
        for column in self.data.columns:
            col_stats = {
                'count': len(self.data[column]),
                'missing_count': self.data[column].isna().sum(),
                'missing_percentage': (self.data[column].isna().sum() / len(self.data[column])) * 100,
                'unique_count': self.data[column].nunique(),
                'dtype': str(self.data[column].dtype)
            }
            
            if pd.api.types.is_numeric_dtype(self.data[column]):
                col_stats.update({
                    'mean': float(self.data[column].mean()),
                    'std': float(self.data[column].std()),
                    'min': float(self.data[column].min()),
                    'max': float(self.data[column].max()),
                    'median': float(self.data[column].median()),
                    'skewness': float(self.data[column].skew()),
                    'kurtosis': float(self.data[column].kurtosis())
                })
            elif pd.api.types.is_string_dtype(self.data[column]):
                non_null = self.data[column].dropna()
                col_stats.update({
                    'min_length': int(non_null.str.len().min()) if len(non_null) > 0 else 0,
                    'max_length': int(non_null.str.len().max()) if len(non_null) > 0 else 0,
                    'avg_length': float(non_null.str.len().mean()) if len(non_null) > 0 else 0,
                    'empty_count': int((non_null == '').sum())
                })
                
            stats[column] = col_stats
            
        return stats
    
    def _compute_spark_stats(self) -> Dict:
        """Compute statistics for Spark DataFrame."""
        from pyspark.sql import functions as F
        from pyspark.sql.types import IntegerType, LongType, FloatType, DoubleType, StringType
        
        stats = {}
        total_count = self.data.count()
        
        # Get summary statistics for all columns
        summary = self.data.summary(
            "count", "mean", "stddev", "min", "25%", "50%", "75%", "max"
        ).collect()
        
        # Convert summary to dictionary for easier access
        summary_dict = {row["summary"]: row.asDict() for row in summary}
        
        # Get column types
        dtypes = dict(self.data.dtypes)
        
        for column in self.data.columns:
            # Initialize column stats
            col_stats = {
                'count': total_count,
                'dtype': dtypes[column],
            }
            
            # Compute missing values
            missing_count = self.data.filter(F.col(column).isNull()).count()
            col_stats.update({
                'missing_count': missing_count,
                'missing_percentage': (missing_count / total_count) * 100 if total_count > 0 else 0
            })
            
            # Compute unique count
            unique_count = self.data.select(column).distinct().count()
            col_stats['unique_count'] = unique_count
            
            # Handle numeric columns
            if isinstance(self.data.schema[column].dataType, (IntegerType, LongType, FloatType, DoubleType)):
                col_stats.update({
                    'mean': float(summary_dict['mean'][column]) if summary_dict['mean'][column] != 'null' else None,
                    'std': float(summary_dict['stddev'][column]) if summary_dict['stddev'][column] != 'null' else None,
                    'min': float(summary_dict['min'][column]) if summary_dict['min'][column] != 'null' else None,
                    'max': float(summary_dict['max'][column]) if summary_dict['max'][column] != 'null' else None,
                    'median': float(summary_dict['50%'][column]) if summary_dict['50%'][column] != 'null' else None,
                })
                
                # Compute skewness and kurtosis
                moments = self.data.select(
                    F.skewness(F.col(column)).alias('skewness'),
                    F.kurtosis(F.col(column)).alias('kurtosis')
                ).collect()[0]
                
                col_stats.update({
                    'skewness': float(moments['skewness']) if moments['skewness'] is not None else None,
                    'kurtosis': float(moments['kurtosis']) if moments['kurtosis'] is not None else None
                })
                
            # Handle string columns
            elif isinstance(self.data.schema[column].dataType, StringType):
                # Compute string length statistics
                length_stats = self.data.select(
                    F.min(F.length(F.col(column))).alias('min_length'),
                    F.max(F.length(F.col(column))).alias('max_length'),
                    F.avg(F.length(F.col(column))).alias('avg_length'),
                    F.sum(F.when(F.col(column) == '', 1).otherwise(0)).alias('empty_count')
                ).collect()[0]
                
                col_stats.update({
                    'min_length': int(length_stats['min_length']) if length_stats['min_length'] is not None else 0,
                    'max_length': int(length_stats['max_length']) if length_stats['max_length'] is not None else 0,
                    'avg_length': float(length_stats['avg_length']) if length_stats['avg_length'] is not None else 0,
                    'empty_count': int(length_stats['empty_count'])
                })
                
            stats[column] = col_stats
            
        return stats 