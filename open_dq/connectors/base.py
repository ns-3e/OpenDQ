from abc import ABC, abstractmethod
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseConnector(ABC):
    """Base class for all database connectors."""
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the data source."""
        pass
    
    @abstractmethod
    def query(self, query: str) -> pd.DataFrame:
        """Execute a query and return results as a pandas DataFrame."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the connection."""
        pass 