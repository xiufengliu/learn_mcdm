"""
Session state management for the MCDM Learning Tool
"""

import streamlit as st
import pandas as pd
import numpy as np
from config.settings import EXAMPLE_PROBLEMS, UI_CONFIG

def initialize_session_state():
    """Initialize all session state variables"""
    
    # Method selection
    if 'selected_method' not in st.session_state:
        st.session_state.selected_method = 'SAW'
    
    # Problem definition
    if 'alternatives' not in st.session_state:
        st.session_state.alternatives = [f"Alternative {i+1}" for i in range(UI_CONFIG["default_alternatives"])]
    
    if 'criteria' not in st.session_state:
        st.session_state.criteria = [f"Criterion {i+1}" for i in range(UI_CONFIG["default_criteria"])]
    
    if 'criterion_types' not in st.session_state:
        st.session_state.criterion_types = ['benefit'] * UI_CONFIG["default_criteria"]
    
    # Decision matrix
    if 'decision_matrix' not in st.session_state:
        st.session_state.decision_matrix = pd.DataFrame(
            np.ones((UI_CONFIG["default_alternatives"], UI_CONFIG["default_criteria"])),
            index=st.session_state.alternatives,
            columns=st.session_state.criteria
        )
    
    # Weights
    if 'weights' not in st.session_state:
        n_criteria = len(st.session_state.criteria)
        st.session_state.weights = [1.0/n_criteria] * n_criteria
    
    # Results
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # UI state
    if 'show_intermediate_steps' not in st.session_state:
        st.session_state.show_intermediate_steps = False

    if 'example_loaded' not in st.session_state:
        st.session_state.example_loaded = None

    # Custom problems UI state
    if 'show_custom_problems' not in st.session_state:
        st.session_state.show_custom_problems = False

    # AHP-specific session state
    initialize_ahp_session_state()

def load_example_problem(example_name):
    """Load an example problem into session state"""
    # Get all problems (built-in + custom)
    from src.utils.custom_problems import get_all_problems
    all_problems = get_all_problems()

    if example_name in all_problems:
        example = all_problems[example_name]
        
        st.session_state.alternatives = example["alternatives"]
        st.session_state.criteria = example["criteria"]
        st.session_state.criterion_types = example["criterion_types"]

        # Handle weights (may be None for AHP examples)
        if example["weights"] is not None:
            st.session_state.weights = example["weights"]
        else:
            n_criteria = len(example["criteria"])
            st.session_state.weights = [1.0/n_criteria] * n_criteria

        # Create decision matrix DataFrame (may be None for AHP examples)
        if example["decision_matrix"] is not None:
            st.session_state.decision_matrix = pd.DataFrame(
                example["decision_matrix"],
                index=example["alternatives"],
                columns=example["criteria"]
            )
        else:
            # Create default matrix for AHP examples
            st.session_state.decision_matrix = pd.DataFrame(
                np.ones((len(example["alternatives"]), len(example["criteria"]))),
                index=example["alternatives"],
                columns=example["criteria"]
            )
        
        st.session_state.example_loaded = example_name
        st.session_state.results = None  # Clear previous results

        # Reset AHP matrices for new problem structure
        reset_ahp_session_state()

def reset_problem():
    """Reset the problem to default state"""
    st.session_state.alternatives = [f"Alternative {i+1}" for i in range(UI_CONFIG["default_alternatives"])]
    st.session_state.criteria = [f"Criterion {i+1}" for i in range(UI_CONFIG["default_criteria"])]
    st.session_state.criterion_types = ['benefit'] * UI_CONFIG["default_criteria"]
    
    n_criteria = len(st.session_state.criteria)
    st.session_state.weights = [1.0/n_criteria] * n_criteria
    
    st.session_state.decision_matrix = pd.DataFrame(
        np.ones((UI_CONFIG["default_alternatives"], UI_CONFIG["default_criteria"])),
        index=st.session_state.alternatives,
        columns=st.session_state.criteria
    )
    
    st.session_state.results = None
    st.session_state.example_loaded = None

    # Reset AHP matrices
    reset_ahp_session_state()

def update_decision_matrix_structure():
    """Update decision matrix when alternatives or criteria change"""
    current_matrix = st.session_state.decision_matrix
    
    # Create new matrix with current alternatives and criteria
    new_matrix = pd.DataFrame(
        index=st.session_state.alternatives,
        columns=st.session_state.criteria
    )
    
    # Copy existing values where possible
    for alt in st.session_state.alternatives:
        for crit in st.session_state.criteria:
            if alt in current_matrix.index and crit in current_matrix.columns:
                new_matrix.loc[alt, crit] = current_matrix.loc[alt, crit]
            else:
                new_matrix.loc[alt, crit] = 1.0  # Default value
    
    st.session_state.decision_matrix = new_matrix
    
    # Update weights if criteria changed
    n_criteria = len(st.session_state.criteria)
    if len(st.session_state.weights) != n_criteria:
        st.session_state.weights = [1.0/n_criteria] * n_criteria
    
    # Update criterion types if criteria changed
    if len(st.session_state.criterion_types) != n_criteria:
        st.session_state.criterion_types = ['benefit'] * n_criteria

    # Update AHP matrices if structure changed
    update_ahp_matrices_structure()

def initialize_ahp_session_state():
    """Initialize AHP-specific session state variables"""
    if 'ahp_criteria_matrix' not in st.session_state:
        n_criteria = len(st.session_state.criteria)
        st.session_state.ahp_criteria_matrix = np.ones((n_criteria, n_criteria))

    if 'ahp_alternative_matrices' not in st.session_state:
        st.session_state.ahp_alternative_matrices = {}

def reset_ahp_session_state():
    """Reset AHP-specific session state variables"""
    n_criteria = len(st.session_state.criteria)
    n_alternatives = len(st.session_state.alternatives)

    # Reset criteria comparison matrix
    st.session_state.ahp_criteria_matrix = np.ones((n_criteria, n_criteria))

    # Reset alternative comparison matrices
    st.session_state.ahp_alternative_matrices = {}

def update_ahp_matrices_structure():
    """Update AHP matrices when problem structure changes"""
    n_criteria = len(st.session_state.criteria)
    n_alternatives = len(st.session_state.alternatives)

    # Update criteria matrix if size changed
    if ('ahp_criteria_matrix' not in st.session_state or
        st.session_state.ahp_criteria_matrix.shape != (n_criteria, n_criteria)):

        # Try to preserve existing values if possible
        old_matrix = st.session_state.get('ahp_criteria_matrix', np.ones((n_criteria, n_criteria)))
        new_matrix = np.ones((n_criteria, n_criteria))

        # Copy existing values where possible
        min_size = min(old_matrix.shape[0], n_criteria)
        if min_size > 0:
            new_matrix[:min_size, :min_size] = old_matrix[:min_size, :min_size]

        st.session_state.ahp_criteria_matrix = new_matrix

    # Update alternative matrices
    if 'ahp_alternative_matrices' not in st.session_state:
        st.session_state.ahp_alternative_matrices = {}

    # Remove matrices for criteria that no longer exist
    current_criteria = set(st.session_state.criteria)
    matrices_to_remove = []
    for criterion in st.session_state.ahp_alternative_matrices:
        if criterion not in current_criteria:
            matrices_to_remove.append(criterion)

    for criterion in matrices_to_remove:
        del st.session_state.ahp_alternative_matrices[criterion]

    # Update existing matrices for new alternative count
    for criterion in st.session_state.ahp_alternative_matrices:
        old_matrix = st.session_state.ahp_alternative_matrices[criterion]
        if old_matrix.shape != (n_alternatives, n_alternatives):
            new_matrix = np.ones((n_alternatives, n_alternatives))

            # Copy existing values where possible
            min_size = min(old_matrix.shape[0], n_alternatives)
            if min_size > 0:
                new_matrix[:min_size, :min_size] = old_matrix[:min_size, :min_size]

            st.session_state.ahp_alternative_matrices[criterion] = new_matrix
