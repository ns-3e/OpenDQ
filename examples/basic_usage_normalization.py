"""
Example usage of the MasterAttributeNormalizer class for extracting and normalizing master data
from transactional sources.

This example demonstrates:
1. Working with both Pandas and PySpark DataFrames
2. Extracting master records with exact matching
3. Handling conflicting values in non-matching fields
4. Standardizing attributes with various rules
5. Validating the extracted entities
"""

import pandas as pd
from pyspark.sql import SparkSession
from open_dq.normalization.master_attribute_normalization import MasterAttributeNormalizer

def create_sample_transaction_data_pandas():
    """Create a sample transaction DataFrame using pandas"""
    return pd.DataFrame({
        'account_id': [1001, 1001, 1001, 1002, 1002, 1003],
        'ssn': ['123-45-6789', '123-45-6789', '123-45-6789', '987-65-4321', '987-65-4321', '111-22-3333'],
        'customer_name': ['John Doe', 'Johnny Doe', 'J. Doe', 'Mary Smith', 'M. Smith', 'Bob Jones'],
        'email': ['john@email.com', 'john.doe@email.com', 'johnd@email.com', 'mary@email.com', 'mary.smith@email.com', 'bob@email.com'],
        'phone': ['123-456-7890', '1234567890', '(123) 456-7890', '555-555-5555', '5555555555', '999-999-9999'],
        'address': ['123 Main St', '123 Main Street', '123 MAIN ST.', '456 Oak Ave', '456 Oak Avenue', '789 Pine Rd'],
        'transaction_date': ['2024-01-01', '2024-01-15', '2024-02-01', '2024-01-10', '2024-02-10', '2024-01-20']
    })

def create_sample_transaction_data_spark(spark):
    """Create a sample transaction DataFrame using PySpark"""
    # Convert pandas DataFrame to Spark DataFrame for demonstration
    pandas_df = create_sample_transaction_data_pandas()
    return spark.createDataFrame(pandas_df)

def demonstrate_pandas_normalization():
    """Demonstrate normalization with pandas DataFrame"""
    print("\n=== Demonstrating Pandas DataFrame Normalization ===")
    
    # Create sample transaction data
    transaction_data = create_sample_transaction_data_pandas()
    print("\nSample Transaction Data:")
    print(transaction_data)
    
    # Initialize normalizer
    normalizer = MasterAttributeNormalizer(transaction_data)
    
    # Extract master records with exact matching
    master_records = normalizer.extract_entities(
        exact_match_columns=['account_id', 'ssn'],
        timestamp_column='transaction_date',
        selection_strategy='most_recent'
    )
    
    print("\nExtracted Master Records with Conflicts:")
    print(master_records)
    
    # Define standardization rules
    standardization_rules = {
        'customer_name_1': {'case': 'title'},
        'customer_name_2': {'case': 'title'},
        'customer_name_3': {'case': 'title'},
        'address_1': {
            'case': 'upper',
            'remove': ['PO BOX', 'SUITE'],
            'replacements': {'ST': 'STREET', 'AVE': 'AVENUE'}
        },
        'phone_1': {
            'format': 'XXX-XXX-XXXX',
            'remove': ['-', '.', ' ', '(', ')']
        }
    }
    
    # Standardize attributes
    standardized_data = normalizer.standardize_attributes(standardization_rules)
    print("\nStandardized Master Records:")
    print(standardized_data)
    
    # Validate entities
    validation_rules = {
        'email_1': {
            'type': 'email',
            'required': True
        },
        'phone_1': {
            'type': 'phone',
            'format': 'XXX-XXX-XXXX',
            'required': True
        }
    }
    
    validation_results = normalizer.validate_entities(validation_rules)
    print("\nValidation Results:")
    print(validation_results)

def demonstrate_spark_normalization():
    """Demonstrate normalization with PySpark DataFrame"""
    print("\n=== Demonstrating PySpark DataFrame Normalization ===")
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName("MasterAttributeNormalization") \
        .getOrCreate()
    
    # Create sample transaction data
    transaction_data = create_sample_transaction_data_spark(spark)
    print("\nSample Transaction Data:")
    transaction_data.show()
    
    # Initialize normalizer
    normalizer = MasterAttributeNormalizer(transaction_data)
    
    # Extract master records with exact matching
    master_records = normalizer.extract_entities(
        exact_match_columns=['account_id', 'ssn'],
        timestamp_column='transaction_date',
        selection_strategy='most_recent'
    )
    
    print("\nExtracted Master Records with Conflicts:")
    master_records.show()
    
    # Define standardization rules
    standardization_rules = {
        'customer_name_1': {'case': 'title'},
        'customer_name_2': {'case': 'title'},
        'customer_name_3': {'case': 'title'},
        'address_1': {
            'case': 'upper',
            'remove': ['PO BOX', 'SUITE'],
            'replacements': {'ST': 'STREET', 'AVE': 'AVENUE'}
        },
        'phone_1': {
            'format': 'XXX-XXX-XXXX',
            'remove': ['-', '.', ' ', '(', ')']
        }
    }
    
    # Standardize attributes
    standardized_data = normalizer.standardize_attributes(standardization_rules)
    print("\nStandardized Master Records:")
    standardized_data.show()
    
    # Validate entities
    validation_rules = {
        'email_1': {
            'type': 'email',
            'required': True
        },
        'phone_1': {
            'type': 'phone',
            'format': 'XXX-XXX-XXXX',
            'required': True
        }
    }
    
    validation_results = normalizer.validate_entities(validation_rules)
    print("\nValidation Results:")
    validation_results.show()
    
    # Stop Spark session
    spark.stop()

def main():
    """Main function to run the examples"""
    # Demonstrate with pandas DataFrame
    demonstrate_pandas_normalization()
    
    # Demonstrate with PySpark DataFrame
    demonstrate_spark_normalization()

if __name__ == "__main__":
    main() 