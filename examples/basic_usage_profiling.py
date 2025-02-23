"""
This example demonstrates the basic usage of OpenDQ's profiling functionality.
It shows how to:
1. Profile pandas and Spark DataFrames
2. Generate different types of reports
3. Use various profiling options
4. Visualize the results
"""

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from open_dq.profiling import DataProfiler

def create_sample_pandas_data():
    """Create a sample pandas DataFrame with various data types and patterns."""
    np.random.seed(42)
    n_rows = 1000
    
    # Generate structured data with patterns
    data = {
        # Numeric columns with different distributions
        'normal_dist': np.random.normal(100, 15, n_rows),
        'uniform_dist': np.random.uniform(0, 100, n_rows),
        'exponential': np.random.exponential(10, n_rows),
        
        # Categorical data
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
        
        # Datetime data with seasonality
        'timestamp': pd.date_range('2023-01-01', periods=n_rows, freq='H'),
        
        # String data with patterns
        'email': [f'user{i}@example.com' for i in range(n_rows)],
        'phone': [f'+1-555-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}' for _ in range(n_rows)],
        
        # Correlated columns
        'base': np.random.normal(50, 10, n_rows)
    }
    
    # Add correlated column
    data['correlated'] = data['base'] * 2 + np.random.normal(0, 5, n_rows)
    
    # Add some missing values
    df = pd.DataFrame(data)
    df.loc[np.random.choice(n_rows, 50), 'normal_dist'] = np.nan
    df.loc[np.random.choice(n_rows, 30), 'category'] = np.nan
    df.loc[np.random.choice(n_rows, 20), 'email'] = np.nan
    
    return df

def create_sample_spark_data(spark):
    """Create a sample Spark DataFrame using the pandas DataFrame."""
    pandas_df = create_sample_pandas_data()
    return spark.createDataFrame(pandas_df)

def profile_pandas_data():
    """Demonstrate profiling with pandas DataFrame."""
    print("\n=== Profiling Pandas DataFrame ===")
    
    # Create sample data
    df = create_sample_pandas_data()
    print(f"Created pandas DataFrame with shape: {df.shape}")
    
    # Initialize profiler with sampling
    profiler = DataProfiler(df, sample_size=500)
    
    # Generate complete profile
    print("\nGenerating complete profile...")
    profile = profiler.generate_profile()
    
    # Save profile to JSON
    profiler.to_json('pandas_profile.json')
    print("Saved profile to pandas_profile.json")
    
    # Generate HTML report with visualizations
    profiler.to_html('pandas_profile.html')
    print("Generated HTML report at pandas_profile.html")
    
    # Generate focused profile
    print("\nGenerating focused profile (only stats and patterns)...")
    focused_profile = profiler.generate_profile(
        include=['basic_stats', 'patterns']
    )
    
    # Print some interesting findings
    print("\nInteresting findings:")
    print(f"- Number of columns: {profile['metadata']['total_columns']}")
    print(f"- Missing values in normal_dist: {profile['basic_stats']['normal_dist']['missing_count']}")
    
    if 'correlation' in profile:
        strong_correlations = profile['correlation']['strong_correlations']
        if strong_correlations:
            print("\nStrong correlations found:")
            for corr in strong_correlations:
                print(f"- {corr['column1']} vs {corr['column2']}: {corr['correlation']:.2f}")
    
    if 'patterns' in profile:
        print("\nDetected patterns:")
        for column, patterns in profile['patterns'].items():
            if 'data_patterns' in patterns and patterns['data_patterns']:
                print(f"- {column}: {list(patterns['data_patterns'].keys())}")

def profile_spark_data():
    """Demonstrate profiling with Spark DataFrame."""
    print("\n=== Profiling Spark DataFrame ===")
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName("OpenDQ-Profiling-Example") \
        .getOrCreate()
    
    # Create sample Spark DataFrame
    df = create_sample_spark_data(spark)
    print(f"Created Spark DataFrame with {df.count()} rows and {len(df.columns)} columns")
    
    # Initialize profiler
    profiler = DataProfiler(df)
    
    # Generate and save profile
    print("\nGenerating profile...")
    profiler.to_json('spark_profile.json')
    profiler.to_html('spark_profile.html')
    print("Generated profile reports")
    
    # Clean up
    spark.stop()

def main():
    """Run the profiling examples."""
    print("OpenDQ Profiling Examples")
    print("=" * 50)
    
    # Profile pandas DataFrame
    profile_pandas_data()
    
    # Profile Spark DataFrame
    profile_spark_data()
    
    print("\nExample completed! Check the generated JSON and HTML reports.")

if __name__ == "__main__":
    main() 