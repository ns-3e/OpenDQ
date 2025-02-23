import logging
from typing import Optional, Dict, Any
import pandas as pd
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame

from .base import BaseConnector

# Configure logging
logger = logging.getLogger(__name__)

class SparkConnector(BaseConnector):
    """Apache Spark connector for big data processing."""
    
    def __init__(
        self,
        app_name: str,
        master: str = "local[*]",
        spark_config: Optional[Dict[str, Any]] = None
    ):
        self.app_name = app_name
        self.master = master
        self.spark_config = spark_config or {}
        self.spark = None
        logger.info("Initializing Spark connector")
    
    def connect(self) -> None:
        try:
            builder = SparkSession.builder.appName(self.app_name).master(self.master)
            
            # Apply any additional Spark configurations
            for key, value in self.spark_config.items():
                builder = builder.config(key, value)
            
            self.spark = builder.getOrCreate()
            logger.info("Successfully created Spark session")
        except Exception as e:
            logger.error(f"Failed to create Spark session: {str(e)}")
            raise
    
    def query(self, query: str) -> pd.DataFrame:
        if not self.spark:
            self.connect()
        try:
            spark_df = self.spark.sql(query)
            return spark_df.toPandas()
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
    
    def read_table(self, table_name: str) -> SparkDataFrame:
        """Read a table using Spark."""
        if not self.spark:
            self.connect()
        try:
            return self.spark.table(table_name)
        except Exception as e:
            logger.error(f"Failed to read table {table_name}: {str(e)}")
            raise
    
    def close(self) -> None:
        if self.spark:
            self.spark.stop()
            logger.info("Spark session stopped") 