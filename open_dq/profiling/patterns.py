import pandas as pd
import numpy as np
from typing import Union, Dict, List
import re
from datetime import datetime

class PatternDetector:
    """Detect patterns and anomalies in the data."""
    
    def __init__(self, data: Union[pd.DataFrame, 'pyspark.sql.DataFrame']):
        self.data = data
        self._is_spark = not isinstance(data, pd.DataFrame)
        
    def detect(self) -> Dict:
        """
        Detect patterns in all columns.
        
        Returns:
            Dictionary containing pattern detection results for each column
        """
        if self._is_spark:
            return self._detect_spark()
            
        patterns = {}
        
        for column in self.data.columns:
            patterns[column] = {
                'data_patterns': self._detect_data_patterns(column),
                'outliers': self._detect_outliers(column),
                'seasonality': self._detect_seasonality(column),
                'format_patterns': self._detect_format_patterns(column)
            }
            
        return patterns
    
    def _detect_data_patterns(self, column: str) -> Dict:
        """Detect common data patterns in the column."""
        if not pd.api.types.is_string_dtype(self.data[column]):
            return {}
            
        patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?[\d\s-]{10,}$',
            'url': r'^https?://[\w\-\.]+(:\d+)?(/[\w\-\./\?%&=]*)?$',
            'date': r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$',
            'zipcode': r'^\d{5}(-\d{4})?$',
            'credit_card': r'^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$'
        }
        
        results = {}
        non_null = self.data[column].dropna()
        
        for pattern_name, regex in patterns.items():
            matches = non_null.str.match(regex, na=False)
            match_count = matches.sum()
            if match_count > 0:
                results[pattern_name] = {
                    'count': int(match_count),
                    'percentage': float(match_count / len(non_null) * 100)
                }
                
        return results
    
    def _detect_outliers(self, column: str) -> Dict:
        """Detect outliers using IQR method."""
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            return {}
            
        q1 = self.data[column].quantile(0.25)
        q3 = self.data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = self.data[
            (self.data[column] < lower_bound) | 
            (self.data[column] > upper_bound)
        ][column]
        
        return {
            'count': len(outliers),
            'percentage': float(len(outliers) / len(self.data) * 100),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'extreme_values': outliers.tolist()[:10]  # List first 10 outliers
        }
    
    def _detect_seasonality(self, column: str) -> Dict:
        """Detect potential seasonality in datetime columns."""
        try:
            if not pd.api.types.is_datetime64_any_dtype(self.data[column]):
                # Try to convert to datetime if possible
                self.data[column] = pd.to_datetime(self.data[column], errors='raise')
        except (TypeError, ValueError):
            return {}
            
        # Check for common seasonality patterns
        seasonality = {}
        series = self.data[column]
        
        # Daily patterns
        hour_counts = series.dt.hour.value_counts()
        if len(hour_counts) > 1:
            peak_hours = hour_counts.nlargest(3)
            seasonality['daily'] = {
                'peak_hours': peak_hours.index.tolist(),
                'peak_counts': peak_hours.values.tolist()
            }
            
        # Weekly patterns
        day_counts = series.dt.day_name().value_counts()
        if len(day_counts) > 1:
            peak_days = day_counts.nlargest(3)
            seasonality['weekly'] = {
                'peak_days': peak_days.index.tolist(),
                'peak_counts': peak_days.values.tolist()
            }
            
        # Monthly patterns
        month_counts = series.dt.month_name().value_counts()
        if len(month_counts) > 1:
            peak_months = month_counts.nlargest(3)
            seasonality['monthly'] = {
                'peak_months': peak_months.index.tolist(),
                'peak_counts': peak_months.values.tolist()
            }
            
        return seasonality
    
    def _detect_format_patterns(self, column: str) -> Dict:
        """Detect common format patterns in string columns."""
        if not pd.api.types.is_string_dtype(self.data[column]):
            return {}
            
        non_null = self.data[column].dropna()
        if len(non_null) == 0:
            return {}
            
        # Analyze character types
        has_numbers = non_null.str.contains(r'\d', regex=True)
        has_letters = non_null.str.contains(r'[a-zA-Z]', regex=True)
        has_special = non_null.str.contains(r'[^a-zA-Z0-9\s]', regex=True)
        
        # Analyze length patterns
        lengths = non_null.str.len()
        
        return {
            'character_composition': {
                'has_numbers': float(has_numbers.mean() * 100),
                'has_letters': float(has_letters.mean() * 100),
                'has_special': float(has_special.mean() * 100)
            },
            'length_stats': {
                'min_length': int(lengths.min()),
                'max_length': int(lengths.max()),
                'mean_length': float(lengths.mean()),
                'most_common_length': int(lengths.mode().iloc[0])
            }
        }
    
    def _detect_spark(self) -> Dict:
        """Detect patterns for Spark DataFrame."""
        from pyspark.sql import functions as F
        from pyspark.sql.types import IntegerType, LongType, FloatType, DoubleType, StringType, TimestampType
        
        patterns = {}
        total_count = self.data.count()
        
        for column in self.data.columns:
            col_patterns = {}
            
            # Detect data patterns for string columns
            if isinstance(self.data.schema[column].dataType, StringType):
                # Define regex patterns
                pattern_exprs = {
                    'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                    'phone': r'^\+?[\d\s-]{10,}$',
                    'url': r'^https?://[\w\-\.]+(:\d+)?(/[\w\-\./\?%&=]*)?$',
                    'date': r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$',
                    'zipcode': r'^\d{5}(-\d{4})?$',
                    'credit_card': r'^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$'
                }
                
                data_patterns = {}
                non_null_count = self.data.filter(F.col(column).isNotNull()).count()
                
                for pattern_name, regex in pattern_exprs.items():
                    # Count matches for each pattern
                    match_count = self.data.filter(
                        F.col(column).rlike(regex)
                    ).count()
                    
                    if match_count > 0:
                        data_patterns[pattern_name] = {
                            'count': match_count,
                            'percentage': (match_count / non_null_count * 100) if non_null_count > 0 else 0
                        }
                        
                col_patterns['data_patterns'] = data_patterns
                
                # Detect format patterns
                format_stats = self.data.select(
                    F.avg(F.length(F.col(column))).alias('avg_length'),
                    F.min(F.length(F.col(column))).alias('min_length'),
                    F.max(F.length(F.col(column))).alias('max_length'),
                    F.sum(F.when(F.col(column).rlike(r'\d'), 1).otherwise(0)).alias('num_count'),
                    F.sum(F.when(F.col(column).rlike(r'[a-zA-Z]'), 1).otherwise(0)).alias('letter_count'),
                    F.sum(F.when(F.col(column).rlike(r'[^a-zA-Z0-9\s]'), 1).otherwise(0)).alias('special_count')
                ).collect()[0]
                
                col_patterns['format_patterns'] = {
                    'character_composition': {
                        'has_numbers': (format_stats['num_count'] / non_null_count * 100) if non_null_count > 0 else 0,
                        'has_letters': (format_stats['letter_count'] / non_null_count * 100) if non_null_count > 0 else 0,
                        'has_special': (format_stats['special_count'] / non_null_count * 100) if non_null_count > 0 else 0
                    },
                    'length_stats': {
                        'min_length': int(format_stats['min_length']) if format_stats['min_length'] is not None else 0,
                        'max_length': int(format_stats['max_length']) if format_stats['max_length'] is not None else 0,
                        'avg_length': float(format_stats['avg_length']) if format_stats['avg_length'] is not None else 0
                    }
                }
            
            # Detect outliers for numeric columns
            if isinstance(self.data.schema[column].dataType, (IntegerType, LongType, FloatType, DoubleType)):
                # Calculate quartiles
                quantiles = self.data.approxQuantile(column, [0.25, 0.75], 0.01)
                q1, q3 = quantiles[0], quantiles[1]
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Find outliers
                outliers = (
                    self.data
                    .filter((F.col(column) < lower_bound) | (F.col(column) > upper_bound))
                    .select(column)
                    .limit(10)
                    .collect()
                )
                
                outlier_count = self.data.filter(
                    (F.col(column) < lower_bound) | 
                    (F.col(column) > upper_bound)
                ).count()
                
                col_patterns['outliers'] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / total_count * 100) if total_count > 0 else 0,
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'extreme_values': [float(row[column]) for row in outliers]
                }
            
            # Detect seasonality for timestamp columns
            if isinstance(self.data.schema[column].dataType, TimestampType):
                seasonality = {}
                
                # Daily patterns
                hour_counts = (
                    self.data
                    .groupBy(F.hour(column).alias('hour'))
                    .count()
                    .orderBy(F.col('count').desc())
                    .limit(3)
                    .collect()
                )
                
                if len(hour_counts) > 1:
                    seasonality['daily'] = {
                        'peak_hours': [row['hour'] for row in hour_counts],
                        'peak_counts': [row['count'] for row in hour_counts]
                    }
                
                # Weekly patterns
                day_counts = (
                    self.data
                    .groupBy(F.dayofweek(column).alias('day'))
                    .count()
                    .orderBy(F.col('count').desc())
                    .limit(3)
                    .collect()
                )
                
                if len(day_counts) > 1:
                    # Convert day numbers to names
                    day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
                    seasonality['weekly'] = {
                        'peak_days': [day_names[row['day']-1] for row in day_counts],
                        'peak_counts': [row['count'] for row in day_counts]
                    }
                
                # Monthly patterns
                month_counts = (
                    self.data
                    .groupBy(F.month(column).alias('month'))
                    .count()
                    .orderBy(F.col('count').desc())
                    .limit(3)
                    .collect()
                )
                
                if len(month_counts) > 1:
                    # Convert month numbers to names
                    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                                'July', 'August', 'September', 'October', 'November', 'December']
                    seasonality['monthly'] = {
                        'peak_months': [month_names[row['month']-1] for row in month_counts],
                        'peak_counts': [row['count'] for row in month_counts]
                    }
                    
                if seasonality:
                    col_patterns['seasonality'] = seasonality
            
            patterns[column] = col_patterns
            
        return patterns 