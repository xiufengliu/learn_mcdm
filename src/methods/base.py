"""
Base class for MCDM methods
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class MCDMMethod(ABC):
    """
    Abstract base class for all MCDM methods
    """
    
    def __init__(self, decision_matrix, weights, criterion_types, alternatives=None, criteria=None):
        """
        Initialize MCDM method
        
        Args:
            decision_matrix (pd.DataFrame or np.array): Decision matrix
            weights (list): Criteria weights
            criterion_types (list): List of 'benefit' or 'cost' for each criterion
            alternatives (list, optional): Alternative names
            criteria (list, optional): Criteria names
        """
        self.decision_matrix = self._prepare_matrix(decision_matrix, alternatives, criteria)
        self.weights = np.array(weights)
        self.criterion_types = criterion_types
        self.alternatives = self.decision_matrix.index.tolist()
        self.criteria = self.decision_matrix.columns.tolist()
        
        # Validate inputs
        self._validate_inputs()
        
        # Results storage
        self.results = {}
        self.intermediate_steps = {}
    
    def _prepare_matrix(self, matrix, alternatives=None, criteria=None):
        """Convert matrix to DataFrame with proper index and columns"""
        if isinstance(matrix, pd.DataFrame):
            return matrix
        else:
            if alternatives is None:
                alternatives = [f"A{i+1}" for i in range(matrix.shape[0])]
            if criteria is None:
                criteria = [f"C{i+1}" for i in range(matrix.shape[1])]
            return pd.DataFrame(matrix, index=alternatives, columns=criteria)
    
    def _validate_inputs(self):
        """Validate input parameters"""
        n_alternatives, n_criteria = self.decision_matrix.shape
        
        if len(self.weights) != n_criteria:
            raise ValueError(f"Number of weights ({len(self.weights)}) must match number of criteria ({n_criteria})")
        
        if len(self.criterion_types) != n_criteria:
            raise ValueError(f"Number of criterion types ({len(self.criterion_types)}) must match number of criteria ({n_criteria})")
        
        if not all(ct in ['benefit', 'cost'] for ct in self.criterion_types):
            raise ValueError("Criterion types must be 'benefit' or 'cost'")
        
        if not np.allclose(np.sum(self.weights), 1.0, rtol=1e-5):
            # Normalize weights
            self.weights = self.weights / np.sum(self.weights)
    
    def normalize_matrix(self, method='linear'):
        """
        Normalize the decision matrix
        
        Args:
            method (str): Normalization method ('linear', 'vector')
            
        Returns:
            pd.DataFrame: Normalized matrix
        """
        matrix = self.decision_matrix.copy()
        
        if method == 'linear':
            # Linear normalization (min-max)
            for i, (col, ctype) in enumerate(zip(matrix.columns, self.criterion_types)):
                col_values = matrix[col]
                range_val = col_values.max() - col_values.min()
                if range_val == 0:
                    # Avoid division by zero when all values are equal
                    matrix[col] = 0.0
                else:
                    if ctype == 'benefit':
                        # For benefit criteria: (x - min) / (max - min)
                        matrix[col] = (col_values - col_values.min()) / range_val
                    else:
                        # For cost criteria: (max - x) / (max - min)
                        matrix[col] = (col_values.max() - col_values) / range_val
        
        elif method == 'vector':
            # Vector normalization
            for i, (col, ctype) in enumerate(zip(matrix.columns, self.criterion_types)):
                col_values = matrix[col].astype(float)
                if ctype == 'benefit':
                    norm = np.sqrt(np.sum(col_values ** 2))
                    if norm != 0:
                        matrix[col] = col_values / norm
                    else:
                        matrix[col] = 0.0
                else:
                    # For cost criteria, convert values so reciprocals are finite
                    if np.any(col_values <= 0):
                        # Shift values based on smallest positive entry or
                        # by the absolute minimum when all are non-positive
                        positives = col_values[col_values > 0]
                        if len(positives) > 0:
                            offset = positives.min() / 1000.0
                        else:
                            offset = abs(col_values.min()) + 1e-6
                        col_values = col_values + offset

                    inv_values = 1 / col_values
                    norm = np.sqrt(np.sum(inv_values ** 2))
                    if norm != 0:
                        matrix[col] = inv_values / norm
                    else:
                        matrix[col] = 0.0
        
        return matrix
    
    @abstractmethod
    def calculate(self):
        """
        Calculate the MCDM method results
        Must be implemented by subclasses
        
        Returns:
            dict: Results dictionary with scores and rankings
        """
        pass
    
    def get_ranking(self, scores):
        """
        Get ranking from scores (1 = best)
        
        Args:
            scores (list or np.array): Scores for alternatives
            
        Returns:
            list: Rankings (1-based)
        """
        # Higher scores get better (lower) rankings
        rankings = np.argsort(np.argsort(scores)[::-1]) + 1
        return rankings.tolist()
    
    def get_results_dataframe(self):
        """
        Get results as a formatted DataFrame
        
        Returns:
            pd.DataFrame: Results with alternatives, scores, and rankings
        """
        if not self.results:
            self.calculate()
        
        df = pd.DataFrame({
            'Alternative': self.alternatives,
            'Score': self.results['scores'],
            'Rank': self.results['rankings']
        })
        
        # Sort by rank
        df = df.sort_values('Rank').reset_index(drop=True)
        
        return df
    
    def get_method_info(self):
        """
        Get information about the method
        
        Returns:
            dict: Method information
        """
        return {
            'name': self.__class__.__name__,
            'description': self.__doc__ or "No description available",
            'alternatives': self.alternatives,
            'criteria': self.criteria,
            'weights': self.weights.tolist(),
            'criterion_types': self.criterion_types
        }
