"""
Sidebar UI components for method selection and problem setup
"""

import streamlit as st
import pandas as pd
import numpy as np
from config.settings import MCDM_METHODS, EXAMPLE_PROBLEMS, UI_CONFIG
from src.utils.session_state import load_example_problem, reset_problem, update_decision_matrix_structure
from src.utils.validation import validate_weights, display_validation_messages
from src.utils.custom_problems import get_all_problems

def render_sidebar():
    """Render the complete sidebar with all components"""
    
    st.sidebar.title("🎯 MCDM Configuration")
    
    # Method selection
    render_method_selection()
    
    st.sidebar.markdown("---")
    
    # Example problems
    render_example_problems()
    
    st.sidebar.markdown("---")
    
    # Problem definition
    render_problem_definition()
    
    st.sidebar.markdown("---")
    
    # Criteria weights
    render_criteria_weights()
    
    st.sidebar.markdown("---")
    
    # Options
    render_options()

def render_method_selection():
    """Render method selection dropdown"""
    
    st.sidebar.subheader("📊 Select MCDM Method")
    
    # Get enabled methods
    enabled_methods = {k: v for k, v in MCDM_METHODS.items() if v["enabled"]}
    
    # Create method options with descriptions
    method_options = []
    method_descriptions = {}
    
    for key, info in enabled_methods.items():
        display_name = f"{info['name']} ({key})"
        method_options.append(display_name)
        method_descriptions[display_name] = f"**{info['complexity']}** - {info['description']}"
    
    # Method selection
    selected_display = st.sidebar.selectbox(
        "Choose a method:",
        method_options,
        index=0,
        help="Select the MCDM method you want to learn and apply"
    )
    
    # Extract method key
    selected_method = selected_display.split('(')[-1].rstrip(')')
    st.session_state.selected_method = selected_method
    
    # Show method description
    if selected_display in method_descriptions:
        st.sidebar.info(method_descriptions[selected_display])

def render_example_problems():
    """Render example problems section"""

    st.sidebar.subheader("📚 Example Problems")

    # Get all problems (built-in + custom)
    all_problems = get_all_problems()

    # Separate built-in and custom problems
    builtin_problems = {k: v for k, v in all_problems.items() if not v.get('custom', False)}
    custom_problems = {k: v for k, v in all_problems.items() if v.get('custom', False)}

    # Create options list with separators
    problem_options = ["None"]

    if builtin_problems:
        problem_options.append("--- Built-in Examples ---")
        problem_options.extend(list(builtin_problems.keys()))

    if custom_problems:
        problem_options.append("--- Custom Problems ---")
        problem_options.extend(list(custom_problems.keys()))

    col1, col2 = st.sidebar.columns(2)

    with col1:
        example_choice = st.selectbox(
            "Load example:",
            problem_options,
            index=0,
            help="Load a pre-defined or custom problem"
        )

    with col2:
        st.write("")
        st.write("")
        if st.button("Load", help="Load the selected example"):
            if example_choice != "None" and not example_choice.startswith("---"):
                load_example_problem(example_choice)
                st.rerun()

    # Reset and manage buttons
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("🔄 Reset", help="Reset to default empty problem"):
            reset_problem()
            st.rerun()

    with col2:
        if st.button("📝 Manage", help="Manage custom problems"):
            st.session_state.show_custom_problems = True
            st.rerun()

    # Show loaded example info
    if st.session_state.example_loaded:
        problem_type = "Custom" if st.session_state.example_loaded in custom_problems else "Built-in"
        st.sidebar.success(f"✅ Loaded: {st.session_state.example_loaded} ({problem_type})")

def render_problem_definition():
    """Render problem definition section"""
    
    st.sidebar.subheader("🎯 Problem Definition")
    
    # Alternatives
    st.sidebar.write("**Alternatives:**")
    n_alternatives = st.sidebar.number_input(
        "Number of alternatives:",
        min_value=2,
        max_value=UI_CONFIG["max_alternatives"],
        value=len(st.session_state.alternatives),
        help="Number of decision alternatives to compare"
    )
    
    # Update alternatives list
    current_alts = len(st.session_state.alternatives)
    if n_alternatives != current_alts:
        if n_alternatives > current_alts:
            # Add new alternatives
            for i in range(current_alts, n_alternatives):
                st.session_state.alternatives.append(f"Alternative {i+1}")
        else:
            # Remove alternatives
            st.session_state.alternatives = st.session_state.alternatives[:n_alternatives]
        update_decision_matrix_structure()
    
    # Edit alternative names
    for i in range(n_alternatives):
        new_name = st.sidebar.text_input(
            f"Alt {i+1}:",
            value=st.session_state.alternatives[i],
            key=f"alt_{i}",
            help=f"Name for alternative {i+1}"
        )
        if new_name != st.session_state.alternatives[i]:
            st.session_state.alternatives[i] = new_name
            update_decision_matrix_structure()
    
    # Criteria
    st.sidebar.write("**Criteria:**")
    n_criteria = st.sidebar.number_input(
        "Number of criteria:",
        min_value=1,
        max_value=UI_CONFIG["max_criteria"],
        value=len(st.session_state.criteria),
        help="Number of decision criteria"
    )
    
    # Update criteria list
    current_crit = len(st.session_state.criteria)
    if n_criteria != current_crit:
        if n_criteria > current_crit:
            # Add new criteria
            for i in range(current_crit, n_criteria):
                st.session_state.criteria.append(f"Criterion {i+1}")
                st.session_state.criterion_types.append('benefit')
        else:
            # Remove criteria
            st.session_state.criteria = st.session_state.criteria[:n_criteria]
            st.session_state.criterion_types = st.session_state.criterion_types[:n_criteria]
        update_decision_matrix_structure()
    
    # Edit criteria names and types
    for i in range(n_criteria):
        col1, col2 = st.sidebar.columns([2, 1])
        
        with col1:
            new_name = st.text_input(
                f"Crit {i+1}:",
                value=st.session_state.criteria[i],
                key=f"crit_{i}",
                help=f"Name for criterion {i+1}"
            )
            if new_name != st.session_state.criteria[i]:
                st.session_state.criteria[i] = new_name
                update_decision_matrix_structure()
        
        with col2:
            new_type = st.selectbox(
                "Type:",
                ["benefit", "cost"],
                index=0 if st.session_state.criterion_types[i] == 'benefit' else 1,
                key=f"crit_type_{i}",
                help="Benefit: higher is better, Cost: lower is better"
            )
            st.session_state.criterion_types[i] = new_type

def render_criteria_weights():
    """Render criteria weights section"""
    
    st.sidebar.subheader("⚖️ Criteria Weights")
    
    n_criteria = len(st.session_state.criteria)
    
    # Weight input method
    weight_method = st.sidebar.radio(
        "Weight input method:",
        ["Equal weights", "Manual input"],
        help="Choose how to set criteria weights"
    )
    
    if weight_method == "Equal weights":
        # Set equal weights
        equal_weight = 1.0 / n_criteria
        st.session_state.weights = [equal_weight] * n_criteria
        st.sidebar.info(f"All weights set to {equal_weight:.4f}")
    
    else:
        # Manual weight input
        st.sidebar.write("Enter weights for each criterion:")
        
        new_weights = []
        for i, criterion in enumerate(st.session_state.criteria):
            weight = st.sidebar.number_input(
                f"{criterion}:",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.weights[i] if i < len(st.session_state.weights) else 1.0/n_criteria,
                step=0.01,
                format="%.3f",
                key=f"weight_{i}",
                help=f"Weight for {criterion} (0.0 to 1.0)"
            )
            new_weights.append(weight)
        
        # Validate and normalize weights
        is_valid, errors, normalized_weights = validate_weights(new_weights)
        
        if normalized_weights:
            st.session_state.weights = normalized_weights
        
        # Display validation messages
        if errors:
            for error in errors:
                if error.startswith("Info:"):
                    st.sidebar.info(error)
                else:
                    st.sidebar.warning(error)
        
        # Show weight sum
        weight_sum = sum(new_weights)
        if not np.isclose(weight_sum, 1.0, rtol=1e-5):
            st.sidebar.warning(f"Weights sum to {weight_sum:.4f}, will be normalized to 1.0")

def render_options():
    """Render additional options"""
    
    st.sidebar.subheader("⚙️ Options")
    
    # Show intermediate steps
    st.session_state.show_intermediate_steps = st.sidebar.checkbox(
        "Show calculation steps",
        value=st.session_state.show_intermediate_steps,
        help="Display step-by-step calculations for educational purposes"
    )
    
    # Download options (placeholder for future implementation)
    st.sidebar.write("**Export Options:**")
    st.sidebar.button("📊 Download Results", disabled=True, help="Coming soon!")
    st.sidebar.button("📋 Download Data", disabled=True, help="Coming soon!")
