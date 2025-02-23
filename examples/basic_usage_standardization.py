"""
Example usage of OpenDQ's data standardization functionality.

This example demonstrates how to use the DataStandardization class to standardize:
- Column names
- String values (trimming)
- Date formats
- Boolean values
- Email addresses
- Phone numbers
- Addresses

Examples are shown for both Pandas and PySpark DataFrames.
"""

import pandas as pd
from pyspark.sql import SparkSession
from open_dq.standardization import DataStandardization

def pandas_example():
    """Example using Pandas DataFrame."""
    # Create sample data
    data = {
        'Customer Email ': ['  John.Doe@EXAMPLE.com ', 'jane_smith@email.com  ', 'bob.jones @test.net'],
        'Phone#': ['(555) 123-4567', '1-555-987-6543', '5551112222'],
        'Address_street': ['123 Main st', '456 oak DRIVE', '789 elm st SW'],
        'Address_city': ['new YORK', 'Los angeles', 'CHICAGO'],
        'Address_state': ['ny', 'ca', 'IL'],
        'Address_zip': ['  10001 ', '90210', ' 60601 '],
        'Active Status': ['Yes', 'N', 'TRUE'],
        'Last Updated': ['2023-01-01', '01/15/2023', '2023-02-28']
    }
    df = pd.DataFrame(data)
    
    # Initialize standardizer
    standardizer = DataStandardization(df)
    
    # 1. Standardize column names
    df = standardizer.standardize_column_names()
    print("\nStandardized column names:")
    print(df.columns.tolist())
    
    # 2. Standardize email
    df = standardizer.standardize_email(['customer_email'])
    print("\nStandardized emails:")
    print(df['customer_email'].tolist())
    
    # 3. Standardize phone numbers
    df = standardizer.standardize_phone(['phone'])
    print("\nStandardized phone numbers:")
    print(df['phone'].tolist())
    
    # 4. Standardize address components
    df = standardizer.standardize_address(['address'])
    print("\nStandardized addresses:")
    print("Street:", df['address_street'].tolist())
    print("City:", df['address_city'].tolist())
    print("State:", df['address_state'].tolist())
    print("ZIP:", df['address_zip'].tolist())
    
    # 5. Standardize boolean values
    df = standardizer.standardize_boolean_values(['active_status'])
    print("\nStandardized boolean values:")
    print(df['active_status'].tolist())
    
    # 6. Standardize dates
    df = standardizer.standardize_date_format(
        ['last_updated'],
        date_format='%Y-%m-%d',
        input_formats=['%Y-%m-%d', '%m/%d/%Y']
    )
    print("\nStandardized dates:")
    print(df['last_updated'].tolist())

def spark_example():
    """Example using PySpark DataFrame."""
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("OpenDQ-Standardization-Example") \
        .getOrCreate()
    
    # Create sample data
    data = [
        {
            'Customer Email ': '  John.Doe@EXAMPLE.com ',
            'Phone#': '(555) 123-4567',
            'Address_street': '123 Main st',
            'Address_city': 'new YORK',
            'Address_state': 'ny',
            'Address_zip': '  10001 ',
            'Active Status': 'Yes',
            'Last Updated': '2023-01-01'
        },
        {
            'Customer Email ': 'jane_smith@email.com  ',
            'Phone#': '1-555-987-6543',
            'Address_street': '456 oak DRIVE',
            'Address_city': 'Los angeles',
            'Address_state': 'ca',
            'Address_zip': '90210',
            'Active Status': 'N',
            'Last Updated': '01/15/2023'
        }
    ]
    df = spark.createDataFrame(data)
    
    # Initialize standardizer
    standardizer = DataStandardization(df)
    
    # Apply standardizations
    df = (standardizer
          .standardize_column_names()
          .standardize_email(['customer_email'])
          .standardize_phone(['phone'])
          .standardize_address(['address'])
          .standardize_boolean_values(['active_status'])
          .standardize_date_format(
              ['last_updated'],
              date_format='yyyy-MM-dd'
          ))
    
    # Show results
    print("\nStandardized Spark DataFrame:")
    df.show(truncate=False)
    
    # Clean up
    spark.stop()

def main():
    """Run both Pandas and PySpark examples."""
    print("Running Pandas example...")
    pandas_example()
    
    print("\nRunning PySpark example...")
    spark_example()

if __name__ == "__main__":
    main() 