"""
Input validation utilities for the MCDM Learning Tool
"""

import pandas as pd
import numpy as np
import streamlit as st

def validate_decision_matrix(matrix, method_name=None):
    """
    Validate the decision matrix for common issues

    Args:
        matrix (pd.DataFrame): Decision matrix to validate
        method_name (str, optional): Name of the MCDM method for specific validation

    Returns:
        tuple: (is_valid, error_messages, warnings)
    """
    errors = []
    warnings = []

    # Check for empty matrix
    if matrix is None or matrix.empty:
        errors.append("Decision matrix is empty or not provided")
        return False, errors, warnings

    # Check matrix dimensions
    if matrix.shape[0] < 2:
        errors.append("Decision matrix must have at least 2 alternatives (rows)")

    if matrix.shape[1] < 1:
        errors.append("Decision matrix must have at least 1 criterion (column)")

    # Check for missing values
    if matrix.isnull().any().any():
        null_locations = []
        for col in matrix.columns:
            null_rows = matrix[matrix[col].isnull()].index.tolist()
            if null_rows:
                null_locations.append(f"Column '{col}': rows {null_rows}")
        errors.append(f"Decision matrix contains missing values in: {'; '.join(null_locations)}")

    # Check for non-numeric values
    try:
        numeric_matrix = matrix.astype(float)
    except (ValueError, TypeError) as e:
        non_numeric_locations = []
        for col in matrix.columns:
            for idx in matrix.index:
                try:
                    float(matrix.loc[idx, col])
                except (ValueError, TypeError):
                    non_numeric_locations.append(f"({idx}, {col})")
        errors.append(f"Decision matrix contains non-numeric values at: {'; '.join(non_numeric_locations[:5])}"
                     + ("..." if len(non_numeric_locations) > 5 else ""))
        return False, errors, warnings

    # Method-specific validations
    if method_name:
        method_errors, method_warnings = _validate_matrix_for_method(numeric_matrix, method_name)
        errors.extend(method_errors)
        warnings.extend(method_warnings)
    else:
        # General warnings
        if (numeric_matrix < 0).any().any():
            negative_locations = []
            for col in numeric_matrix.columns:
                negative_rows = numeric_matrix[numeric_matrix[col] < 0].index.tolist()
                if negative_rows:
                    negative_locations.append(f"Column '{col}': rows {negative_rows[:3]}")
            warnings.append(f"Matrix contains negative values in: {'; '.join(negative_locations[:3])}"
                          + ("..." if len(negative_locations) > 3 else ""))

        if (numeric_matrix == 0).any().any():
            zero_locations = []
            for col in numeric_matrix.columns:
                zero_rows = numeric_matrix[numeric_matrix[col] == 0].index.tolist()
                if zero_rows:
                    zero_locations.append(f"Column '{col}': rows {zero_rows[:3]}")
            warnings.append(f"Matrix contains zero values in: {'; '.join(zero_locations[:3])}"
                          + ("..." if len(zero_locations) > 3 else ""))

    # Check for constant columns (all values the same)
    for col in numeric_matrix.columns:
        if numeric_matrix[col].nunique() == 1:
            warnings.append(f"Column '{col}' has constant values - this may affect ranking results")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def _validate_matrix_for_method(matrix, method_name):
    """
    Perform method-specific validation

    Args:
        matrix (pd.DataFrame): Numeric decision matrix
        method_name (str): Name of the MCDM method

    Returns:
        tuple: (errors, warnings)
    """
    errors = []
    warnings = []

    if method_name.upper() == 'WPM':
        # WPM requires positive values for meaningful calculations
        if (matrix <= 0).any().any():
            warnings.append("WPM works best with positive values. Non-positive values will be adjusted.")

    elif method_name.upper() == 'AHP':
        # AHP doesn't use decision matrix in the traditional sense
        warnings.append("AHP uses pairwise comparison matrices instead of a decision matrix")

    elif method_name.upper() == 'TOPSIS':
        # TOPSIS can handle negative values but warn about zero values
        if (matrix == 0).any().any():
            warnings.append("TOPSIS may produce unexpected results with zero values")

    return errors, warnings

def validate_weights(weights, n_criteria=None):
    """
    Validate criteria weights

    Args:
        weights (list): List of weight values
        n_criteria (int, optional): Expected number of criteria

    Returns:
        tuple: (is_valid, error_messages, warnings, normalized_weights)
    """
    errors = []
    warnings = []

    # Check if weights is empty
    if not weights:
        errors.append("No weights provided")
        return False, errors, warnings, None

    # Check length if n_criteria is provided
    if n_criteria is not None and len(weights) != n_criteria:
        errors.append(f"Number of weights ({len(weights)}) doesn't match number of criteria ({n_criteria})")

    # Convert to numpy array for easier manipulation
    try:
        weights_array = np.array(weights, dtype=float)
    except (ValueError, TypeError) as e:
        non_numeric_indices = []
        for i, w in enumerate(weights):
            try:
                float(w)
            except (ValueError, TypeError):
                non_numeric_indices.append(i)
        errors.append(f"Weights contain non-numeric values at positions: {non_numeric_indices}")
        return False, errors, warnings, None

    # Check for negative weights
    negative_indices = np.where(weights_array < 0)[0]
    if len(negative_indices) > 0:
        errors.append(f"Weights cannot be negative. Found negative values at positions: {negative_indices.tolist()}")

    # Check for all zero weights
    if np.sum(weights_array) == 0:
        errors.append("Sum of weights cannot be zero")
        return False, errors, warnings, None

    # Check for very small weights that might cause numerical issues
    very_small_indices = np.where((weights_array > 0) & (weights_array < 1e-10))[0]
    if len(very_small_indices) > 0:
        warnings.append(f"Very small weights found at positions {very_small_indices.tolist()}. "
                       "This might cause numerical precision issues.")

    # Check for extremely unbalanced weights
    if np.max(weights_array) / np.min(weights_array[weights_array > 0]) > 1000:
        warnings.append("Weights are highly unbalanced (max/min ratio > 1000). "
                       "Consider if this reflects your true preferences.")

    # Normalize weights to sum to 1
    normalized_weights = weights_array / np.sum(weights_array)

    # Check if normalization was needed
    original_sum = np.sum(weights_array)
    if not np.isclose(original_sum, 1.0, rtol=1e-5):
        warnings.append(f"Weights normalized to sum to 1.0 (original sum: {original_sum:.4f})")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings, normalized_weights.tolist()

def validate_criterion_types(criterion_types, n_criteria):
    """
    Validate criterion types (benefit/cost)
    
    Args:
        criterion_types (list): List of criterion types
        n_criteria (int): Expected number of criteria
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    # Check length
    if len(criterion_types) != n_criteria:
        errors.append(f"Number of criterion types ({len(criterion_types)}) doesn't match number of criteria ({n_criteria})")
    
    # Check valid values
    valid_types = ['benefit', 'cost']
    for i, ctype in enumerate(criterion_types):
        if ctype not in valid_types:
            errors.append(f"Invalid criterion type '{ctype}' at position {i+1}. Must be 'benefit' or 'cost'")
    
    return len(errors) == 0, errors

def validate_alternatives_and_criteria(alternatives, criteria):
    """
    Validate alternatives and criteria lists
    
    Args:
        alternatives (list): List of alternative names
        criteria (list): List of criteria names
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    # Check minimum requirements
    if len(alternatives) < 2:
        errors.append("At least 2 alternatives are required")
    
    if len(criteria) < 1:
        errors.append("At least 1 criterion is required")
    
    # Check for duplicates
    if len(set(alternatives)) != len(alternatives):
        errors.append("Alternative names must be unique")
    
    if len(set(criteria)) != len(criteria):
        errors.append("Criteria names must be unique")
    
    # Check for empty names
    if any(not alt.strip() for alt in alternatives):
        errors.append("Alternative names cannot be empty")
    
    if any(not crit.strip() for crit in criteria):
        errors.append("Criteria names cannot be empty")
    
    return len(errors) == 0, errors

def display_validation_messages(errors=None, warnings=None, infos=None):
    """
    Display validation messages in Streamlit

    Args:
        errors (list, optional): List of error messages
        warnings (list, optional): List of warning messages
        infos (list, optional): List of info messages
    """
    if errors:
        for error in errors:
            st.error(f"❌ {error}")

    if warnings:
        for warning in warnings:
            st.warning(f"⚠️ {warning}")

    if infos:
        for info in infos:
            st.info(f"ℹ️ {info}")


def validate_mcdm_inputs(decision_matrix, weights, criterion_types, alternatives=None, criteria=None, method_name=None):
    """
    Comprehensive validation of all MCDM inputs

    Args:
        decision_matrix (pd.DataFrame): Decision matrix
        weights (list): Criteria weights
        criterion_types (list): Criterion types ('benefit' or 'cost')
        alternatives (list, optional): Alternative names
        criteria (list, optional): Criteria names
        method_name (str, optional): MCDM method name

    Returns:
        dict: Validation results with 'is_valid', 'errors', 'warnings', and processed data
    """
    all_errors = []
    all_warnings = []

    # Validate decision matrix
    matrix_valid, matrix_errors, matrix_warnings = validate_decision_matrix(decision_matrix, method_name)
    all_errors.extend(matrix_errors)
    all_warnings.extend(matrix_warnings)

    # Validate weights
    n_criteria = len(decision_matrix.columns) if decision_matrix is not None and not decision_matrix.empty else None
    weights_valid, weight_errors, weight_warnings, normalized_weights = validate_weights(weights, n_criteria)
    all_errors.extend(weight_errors)
    all_warnings.extend(weight_warnings)

    # Validate criterion types
    if n_criteria:
        types_valid, type_errors = validate_criterion_types(criterion_types, n_criteria)
        all_errors.extend(type_errors)

    # Validate alternatives and criteria names
    if alternatives and criteria:
        names_valid, name_errors = validate_alternatives_and_criteria(alternatives, criteria)
        all_errors.extend(name_errors)

    is_valid = len(all_errors) == 0

    return {
        'is_valid': is_valid,
        'errors': all_errors,
        'warnings': all_warnings,
        'normalized_weights': normalized_weights if weights_valid else None
    }
