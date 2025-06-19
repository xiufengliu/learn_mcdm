"""
Base class for MCDM methods
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from ..utils.normalization import normalize_matrix as normalize_matrix_util

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
        """
        Validate input parameters with comprehensive error checking

        Raises:
            ValueError: If inputs are invalid
            TypeError: If inputs have wrong types
        """
        if self.decision_matrix is None:
            raise ValueError("Decision matrix cannot be None")

        if self.decision_matrix.empty:
            raise ValueError("Decision matrix cannot be empty")

        n_alternatives, n_criteria = self.decision_matrix.shape

        # Validate matrix dimensions
        if n_alternatives < 2:
            raise ValueError(f"Decision matrix must have at least 2 alternatives, got {n_alternatives}")

        if n_criteria < 1:
            raise ValueError(f"Decision matrix must have at least 1 criterion, got {n_criteria}")

        # Validate weights
        if self.weights is None:
            raise ValueError("Weights cannot be None")

        if len(self.weights) != n_criteria:
            raise ValueError(f"Number of weights ({len(self.weights)}) must match number of criteria ({n_criteria})")

        # Check for non-numeric weights
        try:
            self.weights = np.array(self.weights, dtype=float)
        except (ValueError, TypeError) as e:
            raise TypeError(f"Weights must be numeric: {e}")

        # Check for negative weights
        if np.any(self.weights < 0):
            raise ValueError("All weights must be non-negative")

        # Check for zero sum
        if np.sum(self.weights) == 0:
            raise ValueError("Sum of weights cannot be zero")

        # Validate criterion types
        if self.criterion_types is None:
            raise ValueError("Criterion types cannot be None")

        if len(self.criterion_types) != n_criteria:
            raise ValueError(f"Number of criterion types ({len(self.criterion_types)}) must match number of criteria ({n_criteria})")

        if not all(ct in ['benefit', 'cost'] for ct in self.criterion_types):
            invalid_types = [ct for ct in self.criterion_types if ct not in ['benefit', 'cost']]
            raise ValueError(f"Invalid criterion types: {invalid_types}. Must be 'benefit' or 'cost'")

        # Validate decision matrix values
        try:
            self.decision_matrix.astype(float)
        except (ValueError, TypeError) as e:
            raise TypeError(f"Decision matrix must contain only numeric values: {e}")

        # Check for missing values
        if self.decision_matrix.isnull().any().any():
            raise ValueError("Decision matrix cannot contain missing values")

        # Normalize weights if needed
        if not np.allclose(np.sum(self.weights), 1.0, rtol=1e-5):
            import warnings
            original_sum = np.sum(self.weights)
            self.weights = self.weights / original_sum
            warnings.warn(f"Weights were normalized to sum to 1.0 (original sum: {original_sum:.4f})",
                         UserWarning)
    
    def normalize_matrix(self, method='linear'):
        """
        Normalize the decision matrix using improved normalization utilities

        Args:
            method (str): Normalization method ('linear', 'vector', 'max', 'sum')

        Returns:
            pd.DataFrame: Normalized matrix

        Raises:
            ValueError: If normalization method is not supported
        """
        return normalize_matrix_util(self.decision_matrix, self.criterion_types, method)
    
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
