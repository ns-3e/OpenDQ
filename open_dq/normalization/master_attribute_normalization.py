import logging
from typing import Union, List, Dict, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from pyspark.sql import DataFrame as SparkDataFrame
import pyspark.sql.functions as F
from pyspark.sql.window import Window

# Configure logging
logger = logging.getLogger(__name__)

class MasterAttributeNormalizer:
    """
    Class for extracting and normalizing master data from transactional data sources.
    Supports both Pandas and Spark DataFrames for processing large-scale data.
    """
    
    def __init__(self, data: Union[pd.DataFrame, SparkDataFrame]):
        """
        Initialize the MasterAttributeNormalizer with transaction data.
        
        Args:
            data: Input DataFrame containing transactional data
        """
        self.data = data
        self._is_spark = isinstance(data, SparkDataFrame)
        logger.info(f"Initializing MasterAttributeNormalizer with {'Spark' if self._is_spark else 'Pandas'} DataFrame")

    def extract_entities(
        self,
        entity_columns: List[str],
        timestamp_column: Optional[str] = None,
        deduplication_strategy: str = 'most_recent',
        match_criteria: Optional[Dict[str, List[str]]] = None
    ) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Extract master data entities from transactional data.
        
        Args:
            entity_columns: List of columns that form the entity (e.g., ['customer_id', 'customer_name'])
            timestamp_column: Column containing transaction timestamps (required for most_recent strategy)
            deduplication_strategy: Strategy for handling duplicates ('most_recent', 'most_frequent', 'first', 'last')
            match_criteria: Dictionary mapping columns to their fuzzy matching criteria
                          e.g., {'customer_name': ['levenshtein', 'soundex']}
        
        Returns:
            DataFrame containing unique entities with normalized attributes
        """
        try:
            if self._is_spark:
                return self._extract_entities_spark(
                    entity_columns,
                    timestamp_column,
                    deduplication_strategy,
                    match_criteria
                )
            else:
                return self._extract_entities_pandas(
                    entity_columns,
                    timestamp_column,
                    deduplication_strategy,
                    match_criteria
                )
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            raise

    def _extract_entities_spark(
        self,
        entity_columns: List[str],
        timestamp_column: Optional[str],
        deduplication_strategy: str,
        match_criteria: Optional[Dict[str, List[str]]]
    ) -> SparkDataFrame:
        """Spark implementation of entity extraction"""
        
        # Apply fuzzy matching if criteria specified
        if match_criteria:
            for column, criteria in match_criteria.items():
                if 'levenshtein' in criteria:
                    # Add Levenshtein distance for similar strings
                    window = Window.partitionBy(entity_columns[0])
                    self.data = self.data.withColumn(
                        f"{column}_normalized",
                        F.first(F.col(column)).over(window)
                    )
                if 'soundex' in criteria:
                    # Add Soundex phonetic matching
                    self.data = self.data.withColumn(
                        f"{column}_soundex",
                        F.soundex(F.col(column))
                    )

        # Apply deduplication strategy
        if deduplication_strategy == 'most_recent':
            if not timestamp_column:
                raise ValueError("timestamp_column is required for most_recent strategy")
            
            window = Window.partitionBy(entity_columns).orderBy(F.desc(timestamp_column))
            result = self.data.withColumn(
                "row_number",
                F.row_number().over(window)
            ).filter(F.col("row_number") == 1).drop("row_number")
            
        elif deduplication_strategy == 'most_frequent':
            window = Window.partitionBy(entity_columns)
            result = self.data.withColumn(
                "frequency",
                F.count("*").over(window)
            ).orderBy(F.desc("frequency"))
            
            # Keep the most frequent occurrence
            result = result.dropDuplicates(entity_columns)
            
        elif deduplication_strategy in ['first', 'last']:
            order_direction = F.asc if deduplication_strategy == 'first' else F.desc
            window = Window.partitionBy(entity_columns).orderBy(order_direction("*"))
            result = self.data.withColumn(
                "row_number",
                F.row_number().over(window)
            ).filter(F.col("row_number") == 1).drop("row_number")
        
        else:
            raise ValueError(f"Unsupported deduplication strategy: {deduplication_strategy}")

        return result.select(entity_columns)

    def _extract_entities_pandas(
        self,
        entity_columns: List[str],
        timestamp_column: Optional[str],
        deduplication_strategy: str,
        match_criteria: Optional[Dict[str, List[str]]]
    ) -> pd.DataFrame:
        """Pandas implementation of entity extraction"""
        
        result = self.data.copy()

        # Apply fuzzy matching if criteria specified
        if match_criteria:
            for column, criteria in match_criteria.items():
                if 'levenshtein' in criteria:
                    from rapidfuzz import process
                    
                    def normalize_similar(group):
                        names = group[column].unique()
                        if len(names) > 1:
                            # Find most similar name as canonical version
                            canonical = process.extractOne(
                                group[column].mode()[0],
                                names
                            )[0]
                            return canonical
                        return names[0]
                    
                    result[f"{column}_normalized"] = result.groupby(
                        entity_columns[0]
                    )[column].transform(normalize_similar)
                
                if 'soundex' in criteria:
                    from jellyfish import soundex
                    result[f"{column}_soundex"] = result[column].apply(soundex)

        # Apply deduplication strategy
        if deduplication_strategy == 'most_recent':
            if not timestamp_column:
                raise ValueError("timestamp_column is required for most_recent strategy")
            
            result = result.sort_values(timestamp_column, ascending=False)
            result = result.drop_duplicates(subset=entity_columns, keep='first')
            
        elif deduplication_strategy == 'most_frequent':
            # Count frequencies
            frequencies = result.groupby(entity_columns).size().reset_index(name='frequency')
            result = pd.merge(result, frequencies, on=entity_columns)
            result = result.sort_values('frequency', ascending=False)
            result = result.drop_duplicates(subset=entity_columns, keep='first')
            
        elif deduplication_strategy == 'first':
            result = result.drop_duplicates(subset=entity_columns, keep='first')
            
        elif deduplication_strategy == 'last':
            result = result.drop_duplicates(subset=entity_columns, keep='last')
            
        else:
            raise ValueError(f"Unsupported deduplication strategy: {deduplication_strategy}")

        return result[entity_columns]

    def standardize_attributes(
        self,
        rules: Dict[str, Dict[str, Union[str, List[str], Dict[str, str]]]]
    ) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Standardize entity attributes based on defined rules.
        
        Args:
            rules: Dictionary of standardization rules for each column
                  Example:
                  {
                      'address': {
                          'case': 'upper',
                          'remove': ['PO BOX', 'SUITE'],
                          'replacements': {'ST': 'STREET', 'AVE': 'AVENUE'}
                      },
                      'phone': {
                          'format': 'XXX-XXX-XXXX',
                          'remove': ['-', '.', ' ']
                      }
                  }
        
        Returns:
            DataFrame with standardized attributes
        """
        try:
            if self._is_spark:
                return self._standardize_attributes_spark(rules)
            else:
                return self._standardize_attributes_pandas(rules)
        except Exception as e:
            logger.error(f"Error standardizing attributes: {str(e)}")
            raise

    def _standardize_attributes_spark(
        self,
        rules: Dict[str, Dict[str, Union[str, List[str], Dict[str, str]]]]
    ) -> SparkDataFrame:
        """Spark implementation of attribute standardization"""
        
        result = self.data
        
        for column, rule in rules.items():
            if column not in result.columns:
                logger.warning(f"Column {column} not found in DataFrame")
                continue
            
            expr = F.col(column)
            
            # Apply case transformation
            if 'case' in rule:
                if rule['case'].lower() == 'upper':
                    expr = F.upper(expr)
                elif rule['case'].lower() == 'lower':
                    expr = F.lower(expr)
                elif rule['case'].lower() == 'title':
                    expr = F.initcap(expr)
            
            # Remove specified patterns
            if 'remove' in rule:
                for pattern in rule['remove']:
                    expr = F.regexp_replace(expr, pattern, '')
            
            # Apply replacements
            if 'replacements' in rule:
                for old, new in rule['replacements'].items():
                    expr = F.regexp_replace(expr, old, new)
            
            # Apply format
            if 'format' in rule:
                # Implement custom formatting logic based on the format string
                pass  # Complex formatting would need custom implementation
            
            result = result.withColumn(f"{column}_standardized", expr)
        
        return result

    def _standardize_attributes_pandas(
        self,
        rules: Dict[str, Dict[str, Union[str, List[str], Dict[str, str]]]]
    ) -> pd.DataFrame:
        """Pandas implementation of attribute standardization"""
        
        result = self.data.copy()
        
        for column, rule in rules.items():
            if column not in result.columns:
                logger.warning(f"Column {column} not found in DataFrame")
                continue
            
            # Create standardized column
            result[f"{column}_standardized"] = result[column].astype(str)
            
            # Apply case transformation
            if 'case' in rule:
                if rule['case'].lower() == 'upper':
                    result[f"{column}_standardized"] = result[f"{column}_standardized"].str.upper()
                elif rule['case'].lower() == 'lower':
                    result[f"{column}_standardized"] = result[f"{column}_standardized"].str.lower()
                elif rule['case'].lower() == 'title':
                    result[f"{column}_standardized"] = result[f"{column}_standardized"].str.title()
            
            # Remove specified patterns
            if 'remove' in rule:
                for pattern in rule['remove']:
                    result[f"{column}_standardized"] = result[f"{column}_standardized"].str.replace(
                        pattern, '', regex=True
                    )
            
            # Apply replacements
            if 'replacements' in rule:
                for old, new in rule['replacements'].items():
                    result[f"{column}_standardized"] = result[f"{column}_standardized"].str.replace(
                        old, new, regex=True
                    )
            
            # Apply format
            if 'format' in rule:
                format_str = rule['format']
                # Implement custom formatting based on format string
                # Example: Format phone numbers
                if format_str == 'XXX-XXX-XXXX':
                    result[f"{column}_standardized"] = result[f"{column}_standardized"].str.replace(
                        r'[\D]', '', regex=True
                    ).apply(lambda x: f"{x[:3]}-{x[3:6]}-{x[6:]}" if len(x) >= 10 else x)
        
        return result

    def validate_entities(
        self,
        validation_rules: Dict[str, Dict[str, Union[str, List[str], Dict[str, str]]]]
    ) -> pd.DataFrame:
        """
        Validate extracted entities against defined rules and return validation results.
        
        Args:
            validation_rules: Dictionary of validation rules for each column
                            Example:
                            {
                                'email': {
                                    'type': 'email',
                                    'required': True
                                },
                                'phone': {
                                    'type': 'phone',
                                    'format': 'XXX-XXX-XXXX',
                                    'required': False
                                }
                            }
        
        Returns:
            DataFrame with validation results
        """
        try:
            # Convert Spark DataFrame to Pandas for validation if necessary
            data = self.data.toPandas() if self._is_spark else self.data.copy()
            
            validation_results = pd.DataFrame(index=data.index)
            
            for column, rules in validation_rules.items():
                if column not in data.columns:
                    logger.warning(f"Column {column} not found in DataFrame")
                    continue
                
                # Check required fields
                if rules.get('required', False):
                    validation_results[f"{column}_missing"] = data[column].isna()
                
                # Validate data types
                if 'type' in rules:
                    if rules['type'] == 'email':
                        validation_results[f"{column}_invalid"] = ~data[column].str.match(
                            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                        )
                    elif rules['type'] == 'phone':
                        validation_results[f"{column}_invalid"] = ~data[column].str.match(
                            r'^\d{3}-\d{3}-\d{4}$'
                        )
                    # Add more type validations as needed
                
                # Validate format
                if 'format' in rules:
                    # Implement format validation based on the format string
                    pass
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating entities: {str(e)}")
            raise 