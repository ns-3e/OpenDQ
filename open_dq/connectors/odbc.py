import logging
import pandas as pd
import pyodbc

from .base import BaseConnector

# Configure logging
logger = logging.getLogger(__name__)

class ODBCConnector(BaseConnector):
    """ODBC connector for relational databases."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection = None
        logger.info("Initializing ODBC connector")
    
    def connect(self) -> None:
        try:
            self.connection = pyodbc.connect(self.connection_string)
            logger.info("Successfully established ODBC connection")
        except pyodbc.Error as e:
            logger.error(f"Failed to establish ODBC connection: {str(e)}")
            raise
    
    def query(self, query: str) -> pd.DataFrame:
        if not self.connection:
            self.connect()
        try:
            return pd.read_sql(query, self.connection)
        except (pyodbc.Error, pd.io.sql.DatabaseError) as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
    
    def close(self) -> None:
        if self.connection:
            self.connection.close()
            logger.info("ODBC connection closed") 