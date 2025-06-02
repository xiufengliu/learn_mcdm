"""
Input validation utilities for the MCDM Learning Tool
"""

import pandas as pd
import numpy as np
import streamlit as st

def validate_decision_matrix(matrix):
    """
    Validate the decision matrix for common issues
    
    Args:
        matrix (pd.DataFrame): Decision matrix to validate
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    # Check for empty matrix
    if matrix.empty:
        errors.append("Decision matrix is empty")
        return False, errors
    
    # Check for missing values
    if matrix.isnull().any().any():
        errors.append("Decision matrix contains missing values")
    
    # Check for non-numeric values
    try:
        matrix.astype(float)
    except (ValueError, TypeError):
        errors.append("Decision matrix contains non-numeric values")
    
    # Check for negative values (warning, not error)
    if (matrix < 0).any().any():
        errors.append("Warning: Decision matrix contains negative values")
    
    # Check for zero values in criteria that will be used for division
    if (matrix == 0).any().any():
        errors.append("Warning: Decision matrix contains zero values")
    
    return len(errors) == 0, errors

def validate_weights(weights):
    """
    Validate criteria weights
    
    Args:
        weights (list): List of weight values
        
    Returns:
        tuple: (is_valid, error_messages, normalized_weights)
    """
    errors = []
    
    # Check if weights is empty
    if not weights:
        errors.append("No weights provided")
        return False, errors, None
    
    # Convert to numpy array for easier manipulation
    try:
        weights_array = np.array(weights, dtype=float)
    except (ValueError, TypeError):
        errors.append("Weights contain non-numeric values")
        return False, errors, None
    
    # Check for negative weights
    if (weights_array < 0).any():
        errors.append("Weights cannot be negative")
    
    # Check for all zero weights
    if np.sum(weights_array) == 0:
        errors.append("Sum of weights cannot be zero")
        return False, errors, None
    
    # Normalize weights to sum to 1
    normalized_weights = weights_array / np.sum(weights_array)
    
    # Check if normalization was needed
    if not np.isclose(np.sum(weights_array), 1.0, rtol=1e-5):
        errors.append(f"Info: Weights normalized to sum to 1.0 (original sum: {np.sum(weights_array):.4f})")
    
    return len([e for e in errors if not e.startswith("Info:")]) == 0, errors, normalized_weights.tolist()

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

def display_validation_messages(errors, message_type="error"):
    """
    Display validation messages in Streamlit
    
    Args:
        errors (list): List of error messages
        message_type (str): Type of message ('error', 'warning', 'info')
    """
    for error in errors:
        if error.startswith("Warning:"):
            st.warning(error)
        elif error.startswith("Info:"):
            st.info(error)
        else:
            if message_type == "error":
                st.error(error)
            elif message_type == "warning":
                st.warning(error)
            else:
                st.info(error)
