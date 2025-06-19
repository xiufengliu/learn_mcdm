"""
Normalization utilities for MCDM methods
"""

import numpy as np
import pandas as pd
import warnings


def ensure_positive_values(values, column_name="column", method="offset"):
    """
    Ensure all values are positive for calculations that require positive values
    
    Args:
        values (pd.Series or np.array): Values to process
        column_name (str): Name of the column for error reporting
        method (str): Method to handle non-positive values ('offset', 'error')
        
    Returns:
        np.array: Positive values
        
    Raises:
        ValueError: If values cannot be made positive meaningfully
    """
    values = np.array(values, dtype=float)
    
    if np.all(values > 0):
        return values
    
    min_val = np.min(values)
    
    if min_val <= 0:
        if method == "error":
            raise ValueError(f"Column '{column_name}' contains non-positive values. "
                           "This method requires all positive values.")
        
        if np.all(values <= 0):
            raise ValueError(f"Column '{column_name}' contains only non-positive values. "
                           "Cannot meaningfully convert to positive values.")
        
        # Shift all values to make them positive
        # Use the absolute value of the minimum plus 1 to ensure all values are >= 1
        offset = abs(min_val) + 1.0
        values = values + offset
        
        # Log a warning about the transformation
        warnings.warn(f"Column '{column_name}' contained non-positive values. "
                     f"Added offset of {offset:.3f} to ensure positive values. "
                     "This may affect the relative importance of alternatives.",
                     UserWarning)
    
    return values


def linear_normalization(matrix, criterion_types):
    """
    Perform linear (min-max) normalization on a decision matrix
    
    Args:
        matrix (pd.DataFrame): Decision matrix
        criterion_types (list): List of 'benefit' or 'cost' for each criterion
        
    Returns:
        pd.DataFrame: Normalized matrix
    """
    normalized_matrix = matrix.copy()
    
    for i, (col, ctype) in enumerate(zip(matrix.columns, criterion_types)):
        col_values = matrix[col].astype(float)
        range_val = col_values.max() - col_values.min()
        
        if range_val == 0:
            # All values are equal - set to 0.5 for neutral normalization
            normalized_matrix[col] = 0.5
        else:
            if ctype == 'benefit':
                # For benefit criteria: (x - min) / (max - min)
                normalized_matrix[col] = (col_values - col_values.min()) / range_val
            else:
                # For cost criteria: (max - x) / (max - min)
                normalized_matrix[col] = (col_values.max() - col_values) / range_val
    
    return normalized_matrix


def vector_normalization(matrix, criterion_types):
    """
    Perform vector normalization on a decision matrix
    
    Args:
        matrix (pd.DataFrame): Decision matrix
        criterion_types (list): List of 'benefit' or 'cost' for each criterion
        
    Returns:
        pd.DataFrame: Normalized matrix
    """
    normalized_matrix = matrix.copy()
    
    for i, (col, ctype) in enumerate(zip(matrix.columns, criterion_types)):
        col_values = matrix[col].astype(float)
        
        if ctype == 'benefit':
            # Standard vector normalization for benefit criteria
            norm = np.sqrt(np.sum(col_values ** 2))
            if norm != 0:
                normalized_matrix[col] = col_values / norm
            else:
                normalized_matrix[col] = 0.0
        else:
            # For cost criteria, use reciprocals then normalize
            # Ensure positive values first
            positive_values = ensure_positive_values(col_values, col, method="offset")
            
            # Take reciprocals
            inv_values = 1.0 / positive_values
            
            # Normalize the reciprocals
            norm = np.sqrt(np.sum(inv_values ** 2))
            if norm != 0:
                normalized_matrix[col] = inv_values / norm
            else:
                normalized_matrix[col] = 0.0
    
    return normalized_matrix


def max_normalization(matrix, criterion_types):
    """
    Perform max normalization on a decision matrix
    
    Args:
        matrix (pd.DataFrame): Decision matrix
        criterion_types (list): List of 'benefit' or 'cost' for each criterion
        
    Returns:
        pd.DataFrame: Normalized matrix
    """
    normalized_matrix = matrix.copy()
    
    for i, (col, ctype) in enumerate(zip(matrix.columns, criterion_types)):
        col_values = matrix[col].astype(float)
        max_val = col_values.max()
        
        if max_val == 0:
            # All values are zero or negative
            normalized_matrix[col] = 0.0
        else:
            if ctype == 'benefit':
                # For benefit criteria: x / max
                normalized_matrix[col] = col_values / max_val
            else:
                # For cost criteria: min / x (after ensuring positive values)
                positive_values = ensure_positive_values(col_values, col, method="offset")
                min_val = positive_values.min()
                normalized_matrix[col] = min_val / positive_values
    
    return normalized_matrix


def sum_normalization(matrix, criterion_types):
    """
    Perform sum normalization on a decision matrix
    
    Args:
        matrix (pd.DataFrame): Decision matrix
        criterion_types (list): List of 'benefit' or 'cost' for each criterion
        
    Returns:
        pd.DataFrame: Normalized matrix
    """
    normalized_matrix = matrix.copy()
    
    for i, (col, ctype) in enumerate(zip(matrix.columns, criterion_types)):
        col_values = matrix[col].astype(float)
        
        if ctype == 'benefit':
            # For benefit criteria: x / sum
            sum_val = col_values.sum()
            if sum_val != 0:
                normalized_matrix[col] = col_values / sum_val
            else:
                normalized_matrix[col] = 0.0
        else:
            # For cost criteria: (1/x) / sum(1/x)
            positive_values = ensure_positive_values(col_values, col, method="offset")
            inv_values = 1.0 / positive_values
            sum_inv = inv_values.sum()
            if sum_inv != 0:
                normalized_matrix[col] = inv_values / sum_inv
            else:
                normalized_matrix[col] = 0.0
    
    return normalized_matrix


# Dictionary mapping normalization method names to functions
NORMALIZATION_METHODS = {
    'linear': linear_normalization,
    'vector': vector_normalization,
    'max': max_normalization,
    'sum': sum_normalization
}


def normalize_matrix(matrix, criterion_types, method='linear'):
    """
    Normalize a decision matrix using the specified method
    
    Args:
        matrix (pd.DataFrame): Decision matrix
        criterion_types (list): List of 'benefit' or 'cost' for each criterion
        method (str): Normalization method ('linear', 'vector', 'max', 'sum')
        
    Returns:
        pd.DataFrame: Normalized matrix
        
    Raises:
        ValueError: If method is not supported
    """
    if method not in NORMALIZATION_METHODS:
        raise ValueError(f"Unsupported normalization method: {method}. "
                        f"Supported methods: {list(NORMALIZATION_METHODS.keys())}")
    
    return NORMALIZATION_METHODS[method](matrix, criterion_types)
