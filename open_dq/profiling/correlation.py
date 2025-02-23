import pandas as pd
import numpy as np
from typing import Union, Dict, List, Optional
from scipy import stats

class CorrelationAnalyzer:
    """Analyze correlations and relationships between columns."""
    
    def __init__(self, data: Union[pd.DataFrame, 'pyspark.sql.DataFrame']):
        self.data = data
        self._is_spark = not isinstance(data, pd.DataFrame)
        
    def analyze(self, method: str = 'pearson', threshold: float = 0.7) -> Dict:
        """
        Analyze correlations between columns.
        
        Args:
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            threshold: Correlation coefficient threshold for strong correlations
            
        Returns:
            Dictionary containing correlation analysis results
        """
        if self._is_spark:
            return self._analyze_spark(threshold)
            
        results = {
            'correlation_matrix': self._compute_correlation_matrix(method),
            'strong_correlations': self._find_strong_correlations(threshold, method),
            'categorical_associations': self._analyze_categorical_associations()
        }
        
        return results
    
    def _compute_correlation_matrix(self, method: str) -> Dict:
        """Compute correlation matrix for numeric columns."""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {}
            
        corr_matrix = self.data[numeric_cols].corr(method=method)
        return {
            'columns': numeric_cols.tolist(),
            'values': corr_matrix.values.tolist()
        }
    
    def _find_strong_correlations(self, threshold: float, method: str) -> List[Dict]:
        """Find pairs of columns with correlation above threshold."""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return []
            
        corr_matrix = self.data[numeric_cols].corr(method=method)
        strong_correlations = []
        
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                correlation = corr_matrix.iloc[i, j]
                if abs(correlation) >= threshold:
                    strong_correlations.append({
                        'column1': numeric_cols[i],
                        'column2': numeric_cols[j],
                        'correlation': float(correlation)
                    })
                    
        return strong_correlations
    
    def _analyze_categorical_associations(self) -> List[Dict]:
        """Analyze associations between categorical columns using chi-square test."""
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        associations = []
        
        for i in range(len(categorical_cols)):
            for j in range(i + 1, len(categorical_cols)):
                col1, col2 = categorical_cols[i], categorical_cols[j]
                contingency = pd.crosstab(self.data[col1], self.data[col2])
                
                try:
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                    associations.append({
                        'column1': col1,
                        'column2': col2,
                        'chi2_statistic': float(chi2),
                        'p_value': float(p_value),
                        'degrees_of_freedom': int(dof)
                    })
                except ValueError:
                    # Skip if chi-square test cannot be computed
                    continue
                    
        return associations
    
    def _analyze_spark(self, threshold: float) -> Dict:
        """Analyze correlations for Spark DataFrame."""
        from pyspark.sql import functions as F
        from pyspark.sql.types import IntegerType, LongType, FloatType, DoubleType, StringType
        from pyspark.ml.stat import Correlation
        from pyspark.ml.feature import VectorAssembler, StringIndexer
        
        # Get numeric and categorical columns
        numeric_cols = [
            col for col, dtype in self.data.dtypes
            if isinstance(self.data.schema[col].dataType, (IntegerType, LongType, FloatType, DoubleType))
        ]
        categorical_cols = [
            col for col, dtype in self.data.dtypes
            if isinstance(self.data.schema[col].dataType, StringType)
        ]
        
        results = {}
        
        # Compute correlations for numeric columns
        if len(numeric_cols) >= 2:
            # Assemble features into vector
            assembler = VectorAssembler(
                inputCols=numeric_cols,
                outputCol='features',
                handleInvalid='skip'
            )
            
            # Compute correlation matrix
            vector_df = assembler.transform(self.data)
            corr_matrix = Correlation.corr(vector_df, 'features').collect()[0][0]
            
            # Convert to numpy array for easier processing
            corr_matrix = np.array(corr_matrix.toArray())
            
            results['correlation_matrix'] = {
                'columns': numeric_cols,
                'values': corr_matrix.tolist()
            }
            
            # Find strong correlations
            strong_correlations = []
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    correlation = corr_matrix[i, j]
                    if abs(correlation) >= threshold:
                        strong_correlations.append({
                            'column1': numeric_cols[i],
                            'column2': numeric_cols[j],
                            'correlation': float(correlation)
                        })
                        
            results['strong_correlations'] = strong_correlations
        else:
            results['correlation_matrix'] = {}
            results['strong_correlations'] = []
            
        # Analyze categorical associations
        categorical_associations = []
        
        if len(categorical_cols) >= 2:
            for i in range(len(categorical_cols)):
                for j in range(i + 1, len(categorical_cols)):
                    col1, col2 = categorical_cols[i], categorical_cols[j]
                    
                    # Compute contingency table
                    contingency = (
                        self.data
                        .groupBy(col1, col2)
                        .count()
                        .orderBy(col1, col2)
                        .collect()
                    )
                    
                    # Convert to matrix form
                    unique_vals1 = sorted(set(row[col1] for row in contingency))
                    unique_vals2 = sorted(set(row[col2] for row in contingency))
                    
                    matrix = np.zeros((len(unique_vals1), len(unique_vals2)))
                    for row in contingency:
                        i = unique_vals1.index(row[col1])
                        j = unique_vals2.index(row[col2])
                        matrix[i, j] = row['count']
                    
                    try:
                        # Compute chi-square test
                        chi2, p_value, dof, expected = stats.chi2_contingency(matrix)
                        categorical_associations.append({
                            'column1': col1,
                            'column2': col2,
                            'chi2_statistic': float(chi2),
                            'p_value': float(p_value),
                            'degrees_of_freedom': int(dof)
                        })
                    except ValueError:
                        # Skip if chi-square test cannot be computed
                        continue
                        
        results['categorical_associations'] = categorical_associations
        
        return results 