import logging
from typing import Union, List, Optional

import pandas as pd
import numpy as np
from pyspark.sql import DataFrame as SparkDataFrame
import pyspark.sql.functions as F
from pyspark.ml.feature import MinMaxScaler, StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)

class DataNormalization:
    """Class for normalizing numeric data in DataFrames."""
    
    def __init__(self, data: Union[pd.DataFrame, SparkDataFrame]):
        self.data = data
        self._is_spark = isinstance(data, SparkDataFrame)
        logger.info(f"Initializing DataNormalization with {'Spark' if self._is_spark else 'Pandas'} DataFrame")
    
    def min_max_normalize(
        self,
        columns: Optional[List[str]] = None,
        feature_range: tuple = (0, 1)
    ) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Normalize numeric columns using min-max scaling.
        
        Args:
            columns: List of columns to normalize. If None, normalizes all numeric columns.
            feature_range: Desired range of transformed data (min, max)
        """
        try:
            if self._is_spark:
                # If no columns specified, get all numeric columns
                if columns is None:
                    columns = [field.name for field in self.data.schema.fields
                             if str(field.dataType) in ['IntegerType', 'LongType', 'DoubleType', 'FloatType']]
                
                # Create pipeline stages
                stages = []
                
                # Assemble features into a vector
                assembler = VectorAssembler(
                    inputCols=columns,
                    outputCol="features_vector"
                )
                stages.append(assembler)
                
                # Create and configure the scaler
                scaler = MinMaxScaler(
                    inputCol="features_vector",
                    outputCol="scaled_features",
                    min=feature_range[0],
                    max=feature_range[1]
                )
                stages.append(scaler)
                
                # Create and fit the pipeline
                pipeline = Pipeline(stages=stages)
                model = pipeline.fit(self.data)
                
                # Transform the data
                scaled_data = model.transform(self.data)
                
                # Extract individual features from the vector
                for i, col in enumerate(columns):
                    scaled_data = scaled_data.withColumn(
                        f"{col}_normalized",
                        F.udf(lambda v: float(v[i]))(F.col("scaled_features"))
                    )
                
                # Drop intermediate columns
                self.data = scaled_data.drop("features_vector", "scaled_features")
                
            else:
                # If no columns specified, get all numeric columns
                if columns is None:
                    columns = self.data.select_dtypes(include=np.number).columns.tolist()
                
                # Apply min-max scaling
                for col in columns:
                    min_val = self.data[col].min()
                    max_val = self.data[col].max()
                    
                    if min_val == max_val:
                        # Handle constant column
                        self.data[f"{col}_normalized"] = feature_range[0]
                    else:
                        # Apply min-max formula
                        normalized = (self.data[col] - min_val) / (max_val - min_val)
                        self.data[f"{col}_normalized"] = (
                            normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
                        )
            
            logger.info("Successfully applied min-max normalization")
            return self.data
            
        except Exception as e:
            logger.error(f"Error applying min-max normalization: {str(e)}")
            raise
    
    def standard_score_normalize(
        self,
        columns: Optional[List[str]] = None,
        with_mean: bool = True,
        with_std: bool = True
    ) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Standardize numeric columns using z-score (standard score) normalization.
        
        Args:
            columns: List of columns to normalize. If None, normalizes all numeric columns.
            with_mean: If True, center the data before scaling
            with_std: If True, scale the data to unit variance
        """
        try:
            if self._is_spark:
                # If no columns specified, get all numeric columns
                if columns is None:
                    columns = [field.name for field in self.data.schema.fields
                             if str(field.dataType) in ['IntegerType', 'LongType', 'DoubleType', 'FloatType']]
                
                # Create pipeline stages
                stages = []
                
                # Assemble features into a vector
                assembler = VectorAssembler(
                    inputCols=columns,
                    outputCol="features_vector"
                )
                stages.append(assembler)
                
                # Create and configure the scaler
                scaler = StandardScaler(
                    inputCol="features_vector",
                    outputCol="scaled_features",
                    withMean=with_mean,
                    withStd=with_std
                )
                stages.append(scaler)
                
                # Create and fit the pipeline
                pipeline = Pipeline(stages=stages)
                model = pipeline.fit(self.data)
                
                # Transform the data
                scaled_data = model.transform(self.data)
                
                # Extract individual features from the vector
                for i, col in enumerate(columns):
                    scaled_data = scaled_data.withColumn(
                        f"{col}_normalized",
                        F.udf(lambda v: float(v[i]))(F.col("scaled_features"))
                    )
                
                # Drop intermediate columns
                self.data = scaled_data.drop("features_vector", "scaled_features")
                
            else:
                # If no columns specified, get all numeric columns
                if columns is None:
                    columns = self.data.select_dtypes(include=np.number).columns.tolist()
                
                # Apply z-score normalization
                for col in columns:
                    if with_mean:
                        mean = self.data[col].mean()
                    else:
                        mean = 0
                        
                    if with_std:
                        std = self.data[col].std()
                        if std == 0:  # Handle zero variance
                            self.data[f"{col}_normalized"] = 0
                            continue
                    else:
                        std = 1
                    
                    self.data[f"{col}_normalized"] = (self.data[col] - mean) / std
            
            logger.info("Successfully applied z-score normalization")
            return self.data
            
        except Exception as e:
            logger.error(f"Error applying z-score normalization: {str(e)}")
            raise

    def robust_scale(
        self,
        columns: Optional[List[str]] = None,
        quantile_range: tuple = (25.0, 75.0)
    ) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Scale features using statistics that are robust to outliers using the Interquartile Range.
        
        Args:
            columns: List of columns to normalize. If None, normalizes all numeric columns.
            quantile_range: Tuple (q_min, q_max) of quantiles to compute the IQR.
        """
        try:
            if self._is_spark:
                # If no columns specified, get all numeric columns
                if columns is None:
                    columns = [field.name for field in self.data.schema.fields
                             if str(field.dataType) in ['IntegerType', 'LongType', 'DoubleType', 'FloatType']]
                
                for col in columns:
                    # Calculate quartiles
                    stats_df = self.data.select(
                        F.percentile_approx(col, [quantile_range[0]/100, 0.5, quantile_range[1]/100]).alias('quartiles')
                    ).collect()[0]
                    
                    q1, median, q3 = stats_df['quartiles']
                    iqr = q3 - q1
                    
                    if iqr == 0:
                        self.data = self.data.withColumn(f"{col}_normalized", F.lit(0))
                    else:
                        self.data = self.data.withColumn(
                            f"{col}_normalized",
                            (F.col(col) - median) / iqr
                        )
            else:
                # If no columns specified, get all numeric columns
                if columns is None:
                    columns = self.data.select_dtypes(include=np.number).columns.tolist()
                
                for col in columns:
                    q1 = np.percentile(self.data[col], quantile_range[0])
                    q3 = np.percentile(self.data[col], quantile_range[1])
                    median = self.data[col].median()
                    iqr = q3 - q1
                    
                    if iqr == 0:
                        self.data[f"{col}_normalized"] = 0
                    else:
                        self.data[f"{col}_normalized"] = (self.data[col] - median) / iqr
            
            logger.info("Successfully applied robust scaling")
            return self.data
            
        except Exception as e:
            logger.error(f"Error applying robust scaling: {str(e)}")
            raise

    def log_transform(
        self,
        columns: Optional[List[str]] = None,
        base: float = np.e,
        handle_zeros: bool = True,
        offset: float = 1.0
    ) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Apply logarithmic transformation to the data.
        
        Args:
            columns: List of columns to transform. If None, transforms all numeric columns.
            base: The logarithm base to use (default is natural logarithm)
            handle_zeros: If True, adds an offset to handle zero values
            offset: The offset to add when handle_zeros is True
        """
        try:
            if self._is_spark:
                # If no columns specified, get all numeric columns
                if columns is None:
                    columns = [field.name for field in self.data.schema.fields
                             if str(field.dataType) in ['IntegerType', 'LongType', 'DoubleType', 'FloatType']]
                
                for col in columns:
                    if handle_zeros:
                        if base == np.e:
                            self.data = self.data.withColumn(
                                f"{col}_normalized",
                                F.log(F.col(col) + offset)
                            )
                        else:
                            self.data = self.data.withColumn(
                                f"{col}_normalized",
                                F.log(F.col(col) + offset) / F.log(F.lit(base))
                            )
                    else:
                        if base == np.e:
                            self.data = self.data.withColumn(
                                f"{col}_normalized",
                                F.log(F.col(col))
                            )
                        else:
                            self.data = self.data.withColumn(
                                f"{col}_normalized",
                                F.log(F.col(col)) / F.log(F.lit(base))
                            )
            else:
                # If no columns specified, get all numeric columns
                if columns is None:
                    columns = self.data.select_dtypes(include=np.number).columns.tolist()
                
                for col in columns:
                    if handle_zeros:
                        self.data[f"{col}_normalized"] = np.log(self.data[col] + offset) / np.log(base)
                    else:
                        self.data[f"{col}_normalized"] = np.log(self.data[col]) / np.log(base)
            
            logger.info("Successfully applied logarithmic transformation")
            return self.data
            
        except Exception as e:
            logger.error(f"Error applying logarithmic transformation: {str(e)}")
            raise

    def box_cox_transform(
        self,
        columns: Optional[List[str]] = None,
        lmbda: Optional[float] = None
    ) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Apply Box-Cox transformation to make the data more normally distributed.
        
        Args:
            columns: List of columns to transform. If None, transforms all numeric columns.
            lmbda: The lambda parameter for the Box-Cox transform. If None, it will be estimated.
        """
        try:
            if self._is_spark:
                logger.warning("Box-Cox transformation is not implemented for Spark DataFrames")
                return self.data
            else:
                # If no columns specified, get all numeric columns
                if columns is None:
                    columns = self.data.select_dtypes(include=np.number).columns.tolist()
                
                for col in columns:
                    # Ensure all values are positive
                    if (self.data[col] <= 0).any():
                        min_val = self.data[col].min()
                        shift = abs(min_val) + 1 if min_val <= 0 else 0
                        data_shifted = self.data[col] + shift
                    else:
                        data_shifted = self.data[col]
                        shift = 0
                    
                    # Apply Box-Cox transformation
                    if lmbda is None:
                        transformed_data, estimated_lambda = stats.boxcox(data_shifted)
                        self.data[f"{col}_normalized"] = transformed_data
                    else:
                        self.data[f"{col}_normalized"] = stats.boxcox(data_shifted, lmbda=lmbda)
            
            logger.info("Successfully applied Box-Cox transformation")
            return self.data
            
        except Exception as e:
            logger.error(f"Error applying Box-Cox transformation: {str(e)}")
            raise

    def decimal_scaling(
        self,
        columns: Optional[List[str]] = None
    ) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Normalize by dividing by the appropriate power of 10 to make the absolute values less than 1.
        
        Args:
            columns: List of columns to normalize. If None, normalizes all numeric columns.
        """
        try:
            if self._is_spark:
                # If no columns specified, get all numeric columns
                if columns is None:
                    columns = [field.name for field in self.data.schema.fields
                             if str(field.dataType) in ['IntegerType', 'LongType', 'DoubleType', 'FloatType']]
                
                for col in columns:
                    # Calculate maximum absolute value
                    max_abs = self.data.select(F.abs(F.col(col)).alias('abs_val')).agg(F.max('abs_val')).collect()[0][0]
                    
                    # Calculate number of digits
                    if max_abs > 0:
                        j = int(np.ceil(np.log10(max_abs)))
                    else:
                        j = 0
                    
                    # Apply decimal scaling
                    self.data = self.data.withColumn(
                        f"{col}_normalized",
                        F.col(col) / F.pow(F.lit(10.0), F.lit(j))
                    )
            else:
                # If no columns specified, get all numeric columns
                if columns is None:
                    columns = self.data.select_dtypes(include=np.number).columns.tolist()
                
                for col in columns:
                    # Calculate maximum absolute value
                    max_abs = np.abs(self.data[col]).max()
                    
                    # Calculate number of digits
                    if max_abs > 0:
                        j = int(np.ceil(np.log10(max_abs)))
                    else:
                        j = 0
                    
                    # Apply decimal scaling
                    self.data[f"{col}_normalized"] = self.data[col] / (10 ** j)
            
            logger.info("Successfully applied decimal scaling")
            return self.data
            
        except Exception as e:
            logger.error(f"Error applying decimal scaling: {str(e)}")
            raise 