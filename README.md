# OpenDQ: Data Quality Library for Python

![OpenDQ Logo](docs/images/OpenDQ_Logo.png)

OpenDQ is a high-performance Python library for comprehensive data quality management, standardization, and normalization. It supports both small-scale and large-scale data processing through various connectors and provides extensive functionality for data quality assessment, standardization, and normalization.

## 🌟 Key Features

- **Multiple Data Source Connectors**
  - ODBC support for relational databases
  - JDBC connectivity
  - Apache Spark integration for cloud databases/warehouses

- **Data Quality Assessment**
  - Missing value detection
  - Duplicate record identification
  - Schema validation
  - Pattern detection and anomaly identification

- **Data Standardization**
  - Column name standardization
  - String value cleanup
  - Date format standardization
  - Custom rule-based transformations

- **Data Normalization**
  - Min-Max scaling
  - Z-Score normalization
  - Robust scaling (IQR-based)
  - Decimal scaling

- **Advanced Data Profiling**
  - Basic statistics
  - Distribution analysis
  - Correlation analysis
  - Pattern detection

- **Master Data Management**
  - Attribute standardization
  - Entity validation
  - Rule-based transformations

## 📦 Installation

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

## 🚀 Quick Start

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

## 📚 Documentation

### Data Connectors

OpenDQ provides three types of connectors:

#### ODBC Connector
```python
from open_dq.connectors import ODBCConnector

connector = ODBCConnector(connection_string="your_connection_string")
df = connector.query("SELECT * FROM table")
```

#### JDBC Connector
```python
from open_dq.connectors import JDBCConnector

connector = JDBCConnector(
    driver_class="com.mysql.jdbc.Driver",
    url="jdbc:mysql://localhost:3306/db",
    username="user",
    password="pass"
)
df = connector.query("SELECT * FROM table")
```

#### Spark Connector
```python
from open_dq.connectors import SparkConnector

connector = SparkConnector(
    app_name="MyApp",
    spark_config={
        "spark.executor.memory": "2g",
        "spark.driver.memory": "1g"
    }
)
df = connector.read_table("my_table")
```

### Data Quality Assessment

```python
from open_dq.quality import DataQuality

quality = DataQuality(df)

# Check missing values
missing_report = quality.check_missing()

# Check duplicates
duplicate_report = quality.check_duplicates()

# Validate schema
schema = {
    'column1': 'int',
    'column2': 'string'
}
validation_result = quality.validate_schema(schema)
```

### Data Standardization

```python
from open_dq.standardization import DataStandardization

standardizer = DataStandardization(df)

# Standardize column names
df = standardizer.standardize_column_names()

# Clean string columns
df = standardizer.trim_string_columns()

# Standardize date columns
df = standardizer.standardize_date_format(
    date_columns=['date1', 'date2'],
    date_format='%Y-%m-%d'
)
```

### Data Normalization

```python
from open_dq.normalization import DataNormalization

normalizer = DataNormalization(df)

# Min-Max normalization
df = normalizer.min_max_normalize(['column1', 'column2'])

# Z-Score normalization
df = normalizer.standard_score_normalize(['column1', 'column2'])

# Robust scaling
df = normalizer.robust_scale(
    columns=['col1', 'col2'],
    quantile_range=(25.0, 75.0)
)

# Decimal scaling
df = normalizer.decimal_scaling(['column1', 'column2'])
```

### Data Profiling

```python
from open_dq.profiling import DataProfiler

profiler = DataProfiler(
    data=df,
    sample_size=10000  # Optional
)

# Get basic statistics
stats = profiler.basic_stats.compute()

# Analyze distributions
distributions = profiler.distribution.analyze()

# Analyze correlations
correlations = profiler.correlation.analyze(method='pearson', threshold=0.7)

# Detect patterns
patterns = profiler.patterns.detect()
```

### Master Data Management

```python
from open_dq.normalization import MasterAttributeNormalizer

normalizer = MasterAttributeNormalizer(df)

# Standardize attributes
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

# Validate entities
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

## 🌟 Best Practices

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

## 🔧 API Reference

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

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ✨ Testing

To run tests:

```bash
pytest tests/
``` 