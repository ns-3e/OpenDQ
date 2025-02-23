import logging
from typing import Optional
import pandas as pd
import jaydebeapi

from .base import BaseConnector

# Configure logging
logger = logging.getLogger(__name__)

class JDBCConnector(BaseConnector):
    """JDBC connector for Java-based database connections."""
    
    def __init__(
        self,
        driver_class: str,
        url: str,
        username: str,
        password: str,
        jar_path: Optional[str] = None
    ):
        self.driver_class = driver_class
        self.url = url
        self.username = username
        self.password = password
        self.jar_path = jar_path
        self.connection = None
        logger.info("Initializing JDBC connector")
    
    def connect(self) -> None:
        try:
            self.connection = jaydebeapi.connect(
                self.driver_class,
                self.url,
                [self.username, self.password],
                self.jar_path
            )
            logger.info("Successfully established JDBC connection")
        except Exception as e:
            logger.error(f"Failed to establish JDBC connection: {str(e)}")
            raise
    
    def query(self, query: str) -> pd.DataFrame:
        if not self.connection:
            self.connect()
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            return pd.DataFrame(data, columns=columns)
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
        finally:
            if cursor:
                cursor.close()
    
    def close(self) -> None:
        if self.connection:
            self.connection.close()
            logger.info("JDBC connection closed") 