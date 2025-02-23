import logging
from typing import Union, List, Dict
import re
from datetime import datetime

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import TimestampType

# Configure logging
logger = logging.getLogger(__name__)

class DataStandardization:
    """Class for standardizing data in DataFrames."""
    
    def __init__(self, data: Union[pd.DataFrame, SparkDataFrame]):
        self.data = data
        self._is_spark = isinstance(data, SparkDataFrame)
        logger.info(f"Initializing DataStandardization with {'Spark' if self._is_spark else 'Pandas'} DataFrame")
    
    def standardize_column_names(self) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Standardize column names by:
        - Converting to lowercase
        - Replacing spaces and special characters with underscores
        - Removing leading/trailing underscores
        """
        try:
            def clean_column_name(name: str) -> str:
                # Convert to lowercase
                name = name.lower()
                # Replace spaces and special characters with underscores
                name = re.sub(r'[^a-z0-9]', '_', name)
                # Remove consecutive underscores
                name = re.sub(r'_+', '_', name)
                # Remove leading/trailing underscores
                return name.strip('_')
            
            if self._is_spark:
                for old_col in self.data.columns:
                    new_col = clean_column_name(old_col)
                    if old_col != new_col:
                        self.data = self.data.withColumnRenamed(old_col, new_col)
            else:
                self.data.columns = [clean_column_name(col) for col in self.data.columns]
            
            logger.info("Successfully standardized column names")
            return self.data
            
        except Exception as e:
            logger.error(f"Error standardizing column names: {str(e)}")
            raise
    
    def trim_string_columns(self, columns: List[str] = None) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Trim whitespace from string columns.
        
        Args:
            columns: Optional list of columns to trim. If None, trims all string columns.
        """
        try:
            if self._is_spark:
                # If no columns specified, get all string columns
                if columns is None:
                    columns = [field.name for field in self.data.schema.fields
                             if str(field.dataType).startswith('StringType')]
                
                # Apply trim to each column
                for col in columns:
                    self.data = self.data.withColumn(col, F.trim(F.col(col)))
            else:
                # If no columns specified, get all object columns
                if columns is None:
                    columns = self.data.select_dtypes(include=['object']).columns
                
                # Apply strip to each column
                for col in columns:
                    self.data[col] = self.data[col].str.strip()
            
            logger.info("Successfully trimmed string columns")
            return self.data
            
        except Exception as e:
            logger.error(f"Error trimming string columns: {str(e)}")
            raise
    
    def standardize_date_format(
        self,
        date_columns: List[str],
        date_format: str = '%Y-%m-%d',
        input_formats: List[str] = None
    ) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Standardize date columns to a consistent format.
        
        Args:
            date_columns: List of columns containing dates
            date_format: Target date format (default: YYYY-MM-DD)
            input_formats: List of possible input date formats to try
        """
        try:
            if self._is_spark:
                for col in date_columns:
                    # Convert to timestamp first
                    self.data = self.data.withColumn(
                        col,
                        F.to_timestamp(F.col(col))
                    )
                    # Then format to desired string format
                    self.data = self.data.withColumn(
                        col,
                        F.date_format(F.col(col), date_format)
                    )
            else:
                for col in date_columns:
                    if input_formats:
                        # Try each input format
                        def parse_date(date_str):
                            if pd.isna(date_str):
                                return None
                            for fmt in input_formats:
                                try:
                                    return datetime.strptime(str(date_str), fmt)
                                except ValueError:
                                    continue
                            return None
                        
                        self.data[col] = (self.data[col]
                                        .apply(parse_date)
                                        .apply(lambda x: x.strftime(date_format) if x else None))
                    else:
                        # Try pandas automatic parsing
                        self.data[col] = pd.to_datetime(self.data[col]).dt.strftime(date_format)
            
            logger.info("Successfully standardized date columns")
            return self.data
            
        except Exception as e:
            logger.error(f"Error standardizing date columns: {str(e)}")
            raise
    
    def standardize_boolean_values(
        self,
        columns: List[str],
        true_values: List[str] = None,
        false_values: List[str] = None
    ) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Standardize boolean columns by converting various string representations to true/false.
        
        Args:
            columns: List of columns to standardize
            true_values: List of strings to interpret as True
            false_values: List of strings to interpret as False
        """
        if true_values is None:
            true_values = ['true', 'yes', '1', 't', 'y']
        if false_values is None:
            false_values = ['false', 'no', '0', 'f', 'n']
            
        try:
            if self._is_spark:
                for col in columns:
                    # Create a CASE WHEN expression for the conversion
                    when_expr = F.when(
                        F.lower(F.col(col)).isin(true_values),
                        F.lit(True)
                    ).when(
                        F.lower(F.col(col)).isin(false_values),
                        F.lit(False)
                    ).otherwise(None)
                    
                    self.data = self.data.withColumn(col, when_expr)
            else:
                for col in columns:
                    def convert_to_bool(val):
                        if pd.isna(val):
                            return None
                        val_str = str(val).lower()
                        if val_str in true_values:
                            return True
                        if val_str in false_values:
                            return False
                        return None
                    
                    self.data[col] = self.data[col].apply(convert_to_bool)
            
            logger.info("Successfully standardized boolean columns")
            return self.data
            
        except Exception as e:
            logger.error(f"Error standardizing boolean columns: {str(e)}")
            raise

    def standardize_email(self, columns: List[str]) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Standardize email addresses by:
        - Converting to lowercase
        - Removing leading/trailing whitespace
        - Removing internal spaces
        
        Args:
            columns: List of columns containing email addresses
        """
        try:
            if self._is_spark:
                for col in columns:
                    self.data = self.data.withColumn(
                        col,
                        F.lower(F.trim(F.regexp_replace(F.col(col), r'\s+', '')))
                    )
            else:
                for col in columns:
                    self.data[col] = (self.data[col]
                                    .str.lower()
                                    .str.strip()
                                    .str.replace(r'\s+', '', regex=True))
            
            logger.info("Successfully standardized email columns")
            return self.data
            
        except Exception as e:
            logger.error(f"Error standardizing email columns: {str(e)}")
            raise

    def standardize_phone(
        self,
        columns: List[str],
        country_code: str = '1',
        output_format: str = '({area}) {prefix}-{line}'
    ) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Standardize phone numbers to a consistent format.
        
        Args:
            columns: List of columns containing phone numbers
            country_code: Default country code (e.g., '1' for US/Canada)
            output_format: Format string for output. Available placeholders:
                         {area}, {prefix}, {line}, {country}
        """
        try:
            def format_phone(phone: str) -> str:
                if pd.isna(phone):
                    return None
                    
                # Remove all non-digit characters
                digits = re.sub(r'\D', '', str(phone))
                
                # Handle different cases based on length
                if len(digits) == 10:  # Standard US number
                    area, prefix, line = digits[:3], digits[3:6], digits[6:]
                    return output_format.format(area=area, prefix=prefix, line=line)
                elif len(digits) == 11 and digits.startswith(country_code):  # With country code
                    area, prefix, line = digits[1:4], digits[4:7], digits[7:]
                    return output_format.format(area=area, prefix=prefix, line=line)
                else:
                    return phone  # Return original if format unknown
            
            if self._is_spark:
                format_phone_udf = F.udf(format_phone)
                for col in columns:
                    self.data = self.data.withColumn(col, format_phone_udf(F.col(col)))
            else:
                for col in columns:
                    self.data[col] = self.data[col].apply(format_phone)
            
            logger.info("Successfully standardized phone columns")
            return self.data
            
        except Exception as e:
            logger.error(f"Error standardizing phone columns: {str(e)}")
            raise

    def standardize_address(
        self,
        columns: List[str],
        components: Dict[str, str] = None
    ) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Standardize address formatting.
        
        Args:
            columns: List of columns containing addresses
            components: Dictionary mapping component names to their standardization rules
                       e.g., {'street': 'title', 'state': 'upper'}
        """
        try:
            if components is None:
                components = {
                    'street': 'title',  # Title case for street names
                    'city': 'title',    # Title case for city names
                    'state': 'upper',   # Uppercase for state abbreviations
                    'zip': 'strip'      # Strip whitespace for ZIP codes
                }
            
            def standardize_component(value: str, rule: str) -> str:
                if pd.isna(value):
                    return None
                    
                value = str(value).strip()
                if rule == 'upper':
                    return value.upper()
                elif rule == 'title':
                    # Handle special cases in title casing
                    special_cases = {'Po': 'PO', 'Ne': 'NE', 'Nw': 'NW', 'Se': 'SE', 'Sw': 'SW'}
                    value = value.title()
                    for old, new in special_cases.items():
                        value = value.replace(old, new)
                    return value
                elif rule == 'strip':
                    return value.strip()
                return value
            
            if self._is_spark:
                for col in columns:
                    for component, rule in components.items():
                        if f"{col}_{component}" in self.data.columns:
                            standardize_udf = F.udf(lambda x: standardize_component(x, rule))
                            self.data = self.data.withColumn(
                                f"{col}_{component}",
                                standardize_udf(F.col(f"{col}_{component}"))
                            )
            else:
                for col in columns:
                    for component, rule in components.items():
                        if f"{col}_{component}" in self.data.columns:
                            self.data[f"{col}_{component}"] = (
                                self.data[f"{col}_{component}"]
                                .apply(lambda x: standardize_component(x, rule))
                            )
            
            logger.info("Successfully standardized address columns")
            return self.data
            
        except Exception as e:
            logger.error(f"Error standardizing address columns: {str(e)}")
            raise 