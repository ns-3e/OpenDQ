import pandas as pd
import numpy as np
from typing import Union, Dict, List, Tuple
from scipy import stats

class DistributionAnalyzer:
    """Analyze the distribution of values in each column."""
    
    def __init__(self, data: Union[pd.DataFrame, 'pyspark.sql.DataFrame']):
        self.data = data
        self._is_spark = not isinstance(data, pd.DataFrame)
        
    def analyze(self) -> Dict:
        """
        Analyze the distribution of values in all columns.
        
        Returns:
            Dictionary containing distribution analysis for each column
        """
        if self._is_spark:
            return self._analyze_spark()
            
        distributions = {}
        
        for column in self.data.columns:
            col_dist = {
                'value_counts': self._get_value_counts(column),
                'quantiles': self._get_quantiles(column)
            }
            
            if pd.api.types.is_numeric_dtype(self.data[column]):
                col_dist.update({
                    'distribution_test': self._test_distribution(column),
                    'histogram': self._get_histogram(column)
                })
                
            distributions[column] = col_dist
            
        return distributions
    
    def _get_value_counts(self, column: str, top_n: int = 10) -> Dict:
        """Get the top N most frequent values and their counts."""
        value_counts = self.data[column].value_counts()
        return {
            'values': value_counts.index[:top_n].tolist(),
            'counts': value_counts.values[:top_n].tolist()
        }
    
    def _get_quantiles(self, column: str) -> Dict:
        """Calculate quantiles for the column."""
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            return {}
            
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        values = np.nanquantile(self.data[column], quantiles)
        return dict(zip([f'q{int(q*100)}' for q in quantiles], values))
    
    def _test_distribution(self, column: str) -> Dict:
        """Test if the data follows normal distribution."""
        clean_data = self.data[column].dropna()
        if len(clean_data) < 3:  # Need at least 3 points for the test
            return {}
            
        statistic, p_value = stats.normaltest(clean_data)
        return {
            'is_normal': bool(p_value > 0.05),
            'normality_p_value': float(p_value)
        }
    
    def _get_histogram(self, column: str, bins: int = 50) -> Dict:
        """Calculate histogram data for numeric columns."""
        clean_data = self.data[column].dropna()
        hist, bin_edges = np.histogram(clean_data, bins=bins)
        return {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }
    
    def _analyze_spark(self) -> Dict:
        """Analyze distributions for Spark DataFrame."""
        from pyspark.sql import functions as F
        from pyspark.sql.types import IntegerType, LongType, FloatType, DoubleType
        from pyspark.ml.feature import Bucketizer
        from pyspark.ml.stat import Summarizer
        
        distributions = {}
        
        for column in self.data.columns:
            col_dist = {}
            
            # Get value counts
            value_counts = (
                self.data.groupBy(column)
                .count()
                .orderBy(F.col('count').desc())
                .limit(10)
                .collect()
            )
            
            col_dist['value_counts'] = {
                'values': [row[column] for row in value_counts],
                'counts': [row['count'] for row in value_counts]
            }
            
            # Handle numeric columns
            if isinstance(self.data.schema[column].dataType, (IntegerType, LongType, FloatType, DoubleType)):
                # Calculate quantiles
                quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
                approx_quantiles = self.data.approxQuantile(
                    column, quantiles, 0.01  # 1% relative error
                )
                
                col_dist['quantiles'] = dict(
                    zip([f'q{int(q*100)}' for q in quantiles], approx_quantiles)
                )
                
                # Calculate histogram
                # First get min and max for bin edges
                min_max = self.data.select(
                    F.min(column).alias('min'),
                    F.max(column).alias('max')
                ).collect()[0]
                
                min_val, max_val = min_max['min'], min_max['max']
                if min_val is not None and max_val is not None:
                    bins = 50
                    bin_edges = np.linspace(min_val, max_val, bins + 1).tolist()
                    
                    # Create bucketizer
                    bucketizer = Bucketizer(
                        splits=bin_edges,
                        inputCol=column,
                        outputCol='bucket'
                    )
                    
                    # Calculate histogram
                    histogram = (
                        bucketizer.transform(self.data)
                        .groupBy('bucket')
                        .count()
                        .orderBy('bucket')
                        .collect()
                    )
                    
                    col_dist['histogram'] = {
                        'counts': [row['count'] for row in histogram],
                        'bin_edges': bin_edges
                    }
                    
                # Test for normality using skewness and kurtosis
                moments = self.data.select(
                    F.skewness(F.col(column)).alias('skewness'),
                    F.kurtosis(F.col(column)).alias('kurtosis')
                ).collect()[0]
                
                # Use skewness and kurtosis to approximate normality
                # For normal distribution: skewness ≈ 0, kurtosis ≈ 3
                skewness = moments['skewness']
                kurtosis = moments['kurtosis']
                
                if skewness is not None and kurtosis is not None:
                    # Consider approximately normal if:
                    # |skewness| < 0.5 and |kurtosis - 3| < 0.5
                    is_normal = abs(skewness) < 0.5 and abs(kurtosis - 3) < 0.5
                    col_dist['distribution_test'] = {
                        'is_normal': is_normal,
                        'skewness': float(skewness),
                        'kurtosis': float(kurtosis)
                    }
                else:
                    col_dist['distribution_test'] = {}
            
            distributions[column] = col_dist 