"""
Example script demonstrating basic usage of OpenDQ library.
"""

import os
from dotenv import load_dotenv
import pandas as pd

from open_dq.connectors import ODBCConnector, JDBCConnector, SparkConnector
from open_dq.quality import DataQuality
from open_dq.standardization import DataStandardization
from open_dq.normalization import DataNormalization

# Load environment variables
load_dotenv()

def demonstrate_odbc_connection():
    """Demonstrate ODBC connection and data quality checks."""
    # Create an ODBC connection
    connection_string = os.getenv("ODBC_CONNECTION_STRING")
    connector = ODBCConnector(connection_string=connection_string)
    
    # Execute a query
    df = connector.query("SELECT * FROM sample_table")
    
    # Perform data quality checks
    quality = DataQuality(df)
    
    # Check for missing values
    missing_stats = quality.check_missing()
    print("\nMissing Value Statistics:")
    print(missing_stats)
    
    # Check for duplicates
    duplicate_stats = quality.check_duplicates()
    print("\nDuplicate Statistics:")
    print(duplicate_stats)
    
    # Validate schema
    expected_schema = {
        'id': 'int',
        'name': 'string',
        'value': 'float'
    }
    schema_validation = quality.validate_schema(expected_schema)
    print("\nSchema Validation Results:")
    print(schema_validation)
    
    return df

def demonstrate_data_standardization(df):
    """Demonstrate data standardization operations."""
    standardizer = DataStandardization(df)
    
    # Standardize column names
    df = standardizer.standardize_column_names()
    
    # Trim string columns
    df = standardizer.trim_string_columns()
    
    # Standardize date columns
    df = standardizer.standardize_date_format(
        date_columns=['created_at', 'updated_at'],
        date_format='%Y-%m-%d'
    )
    
    # Standardize boolean columns
    df = standardizer.standardize_boolean_values(
        columns=['is_active', 'is_deleted']
    )
    
    return df

def demonstrate_data_normalization(df):
    """Demonstrate data normalization operations."""
    normalizer = DataNormalization(df)
    
    # Apply min-max normalization
    df = normalizer.min_max_normalize(
        columns=['value', 'score'],
        feature_range=(0, 1)
    )
    
    # Apply z-score normalization
    df = normalizer.standard_score_normalize(
        columns=['rating', 'weight']
    )
    
    return df

def demonstrate_spark_connection():
    """Demonstrate Spark connection for big data processing."""
    # Create a Spark connection
    spark = SparkConnector(
        app_name="OpenDQ_Demo",
        spark_config={
            "spark.executor.memory": "2g",
            "spark.driver.memory": "1g"
        }
    )
    
    # Read a table using Spark
    spark_df = spark.read_table("large_table")
    
    # Perform operations on Spark DataFrame
    quality = DataQuality(spark_df)
    missing_stats = quality.check_missing()
    
    print("\nMissing Value Statistics (Spark):")
    print(missing_stats)
    
    return spark_df

def main():
    """Main function demonstrating the complete workflow."""
    try:
        # Demonstrate ODBC connection and data quality
        print("Demonstrating ODBC connection and data quality checks...")
        df = demonstrate_odbc_connection()
        
        # Demonstrate data standardization
        print("\nDemonstrating data standardization...")
        df = demonstrate_data_standardization(df)
        
        # Demonstrate data normalization
        print("\nDemonstrating data normalization...")
        df = demonstrate_data_normalization(df)
        
        # Demonstrate Spark connection
        print("\nDemonstrating Spark connection...")
        spark_df = demonstrate_spark_connection()
        
        print("\nAll demonstrations completed successfully!")
        
    except Exception as e:
        print(f"Error during demonstration: {str(e)}")
        raise

if __name__ == "__main__":
    main() 