# OpenDQ: Comprehensive Documentation

<p align="center">
  <img src="assets/OpenDQ_logo.png" alt="OpenDQ Logo" width="300">
</p>

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Core Features](#core-features)
4. [Data Connectors](#data-connectors)
5. [Data Quality Assessment](#data-quality-assessment)
6. [Data Standardization](#data-standardization)
7. [Data Normalization](#data-normalization)
8. [Data Profiling](#data-profiling)
9. [Master Data Management](#master-data-management)
10. [Best Practices](#best-practices)
11. [API Reference](#api-reference)

## Introduction

OpenDQ is a high-performance Python library designed for comprehensive data quality management, standardization, and normalization. It supports both small-scale and large-scale data processing through various connectors and provides extensive functionality for data quality assessment, standardization, and normalization.

### Key Features
- Multiple database connectors (ODBC, JDBC, Apache Spark)
- Comprehensive data quality assessment
- Data standardization and normalization
- Advanced data profiling
- Master data management
- Support for both Pandas and Spark DataFrames

## Installation

```bash
pip install open-dq
```

### Requirements
- Python 3.8+
- pandas
- pyspark
- pyodbc
- JayDeBeApi
- scikit-learn
- pytest

## Core Features

### Quick Start Example

```python
from open_dq.connection import ODBCConnector
from open_dq.quality import DataQuality
from open_dq.standardization import DataStandardization
from open_dq.normalization import DataNormalization

# Connect to database
connector = ODBCConnector(
    connection_string="Driver={SQL Server};Server=server_name;Database=db_name;Trusted_Connection=yes;"
)
df = connector.query("SELECT * FROM your_table")

# Check data quality
quality = DataQuality(df)
missing_stats = quality.check_missing()
duplicate_stats = quality.check_duplicates()

# Standardize data
standardizer = DataStandardization(df)
df_standardized = standardizer.standardize_column_names()
df_standardized = standardizer.trim_string_columns()

# Normalize data
normalizer = DataNormalization(df_standardized)
df_normalized = normalizer.min_max_normalize(['numeric_column1', 'numeric_column2'])
```

## Data Connectors

OpenDQ provides three types of connectors for different data sources:

### ODBC Connector

```python
from open_dq.connectors import ODBCConnector

connector = ODBCConnector(connection_string="your_connection_string")
```

**Features:**
- Standard ODBC connectivity
- Automatic connection management
- Error handling and logging
- Pandas DataFrame output

**Methods:**
- `connect()`: Establish database connection
- `query(query: str)`: Execute SQL query and return results as DataFrame
- `close()`: Close the connection

### JDBC Connector

```python
from open_dq.connectors import JDBCConnector

connector = JDBCConnector(
    driver_class="com.mysql.jdbc.Driver",
    url="jdbc:mysql://localhost:3306/db",
    username="user",
    password="pass",
    jar_path="/path/to/jdbc/driver.jar"  # Optional
)
```

**Features:**
- Java database connectivity
- Support for all JDBC-compliant databases
- Optional JAR path specification
- Automatic resource management

### Spark Connector

```python
from open_dq.connectors import SparkConnector

connector = SparkConnector(
    app_name="MyApp",
    master="local[*]",
    spark_config={
        "spark.executor.memory": "2g",
        "spark.driver.memory": "1g"
    }
)
```

**Features:**
- Apache Spark integration
- Configurable Spark session
- Support for large-scale data processing
- Direct Spark DataFrame output

## Data Quality Assessment

The `DataQuality` class provides comprehensive data quality checks:

```python
from open_dq.quality import DataQuality

quality = DataQuality(df)
```

### Missing Value Analysis

```python
missing_stats = quality.check_missing()
```

Returns statistics for each column:
- Count of missing values
- Percentage of missing values

### Duplicate Detection

```python
duplicate_stats = quality.check_duplicates()
```

Features:
- Identifies exact duplicates
- Supports subset duplicate checking
- Provides duplicate statistics

### Schema Validation

```python
schema = {
    'column1': 'int',
    'column2': 'string',
    'column3': 'float'
}
validation_result = quality.validate_schema(schema)
```

Validates:
- Data types
- Column presence
- Type compatibility

## Data Standardization

The `DataStandardization` class handles data cleaning and standardization:

```python
from open_dq.standardization import DataStandardization

standardizer = DataStandardization(df)
```

### Column Name Standardization

```python
df = standardizer.standardize_column_names()
```

Features:
- Converts to lowercase
- Replaces special characters with underscores
- Removes leading/trailing underscores

### String Value Cleanup

```python
df = standardizer.trim_string_columns()
```

Features:
- Removes leading/trailing whitespace
- Optional column specification
- Handles both Pandas and Spark DataFrames

### Date Format Standardization

```python
df = standardizer.standardize_date_format(
    date_columns=['date1', 'date2'],
    date_format='%Y-%m-%d',
    input_formats=['%d/%m/%Y', '%m-%d-%Y']
)
```

Features:
- Consistent date formatting
- Multiple input format support
- Timezone handling

## Data Normalization

The `DataNormalization` class provides various normalization methods:

```python
from open_dq.normalization import DataNormalization

normalizer = DataNormalization(df)
```

### Min-Max Normalization

```python
df = normalizer.min_max_normalize(
    columns=['col1', 'col2'],
    feature_range=(0, 1)
)
```

Features:
- Scales data to specified range
- Handles numeric columns
- Preserves zero values

### Z-Score Normalization

```python
df = normalizer.standard_score_normalize(
    columns=['col1', 'col2'],
    with_mean=True,
    with_std=True
)
```

Features:
- Centers data around mean
- Scales to unit variance
- Optional mean centering and scaling

### Robust Scaling

```python
df = normalizer.robust_scale(
    columns=['col1', 'col2'],
    quantile_range=(25.0, 75.0)
)
```

Features:
- Outlier-resistant scaling
- IQR-based normalization
- Configurable quantile range

### Decimal Scaling

```python
df = normalizer.decimal_scaling(columns=['col1', 'col2'])
```

Features:
- Scales by powers of 10
- Preserves number relationships
- Automatic scale detection

## Data Profiling

The `DataProfiler` class provides comprehensive data analysis:

```python
from open_dq.profiling import DataProfiler

profiler = DataProfiler(
    data=df,
    sample_size=10000,  # Optional
    random_seed=42      # Optional
)
```

### Basic Statistics

```python
stats = profiler.basic_stats.compute()
```

Provides:
- Descriptive statistics
- Data type information
- Value counts
- String length statistics

### Distribution Analysis

```python
distributions = profiler.distribution.analyze()
```

Features:
- Value distribution analysis
- Histogram generation
- Frequency analysis
- Quantile statistics

### Correlation Analysis

```python
correlations = profiler.correlation.analyze(
    method='pearson',
    threshold=0.7
)
```

Features:
- Multiple correlation methods
- Strong correlation detection
- Categorical association analysis

### Pattern Detection

```python
patterns = profiler.patterns.detect()
```

Features:
- Outlier detection
- Pattern recognition
- Anomaly identification
- Value range analysis

## Master Data Management

The `MasterAttributeNormalizer` class handles master data standardization:

```python
from open_dq.normalization import MasterAttributeNormalizer

normalizer = MasterAttributeNormalizer(df)
```

### Attribute Standardization

```python
rules = {
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

df = normalizer.standardize_attributes(rules)
```

Features:
- Rule-based standardization
- Multiple transformation types
- Format enforcement
- Custom replacement rules

### Entity Validation

```python
validation_rules = {
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

validation_results = normalizer.validate_entities(validation_rules)
```

Features:
- Type validation
- Format validation
- Required field checking
- Custom validation rules

## Best Practices

1. **Connection Management**
   - Always close connections after use
   - Use context managers when possible
   - Handle connection errors appropriately

2. **Performance Optimization**
   - Use Spark for large datasets
   - Apply sampling for profiling
   - Optimize memory usage

3. **Data Quality**
   - Check for missing values first
   - Validate data types early
   - Document quality issues

4. **Standardization**
   - Define consistent naming conventions
   - Document transformation rules
   - Maintain transformation history

5. **Error Handling**
   - Implement proper error handling
   - Log errors and warnings
   - Provide meaningful error messages

## API Reference

For detailed API documentation, please refer to the individual class and method docstrings in the source code.

### Common Parameters

- `data`: Union[pd.DataFrame, SparkDataFrame]
- `columns`: Optional[List[str]]
- `threshold`: float
- `feature_range`: tuple
- `date_format`: str
- `rules`: Dict[str, Dict]

### Return Types

Most methods return either:
- Modified DataFrame (same type as input)
- Dictionary of statistics/results
- Validation report

### Error Handling

All classes use Python's standard exception handling with specific error messages and logging. 