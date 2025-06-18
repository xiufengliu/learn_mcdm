"""
UI components for managing custom problems
"""

import streamlit as st
import pandas as pd
import numpy as np
from src.utils.custom_problems import (
    save_custom_problem, load_custom_problems, delete_custom_problem,
    export_custom_problem, import_custom_problem, validate_problem_data
)
from src.utils.session_state import load_example_problem

def render_custom_problems_interface():
    """Render the custom problems management interface"""
    
    st.header("📝 Custom Problems Manager")
    
    # Create tabs for different functions
    tab1, tab2, tab3, tab4 = st.tabs(["➕ Create New", "📂 Manage Existing", "📤 Export/Import", "📋 Current Problem"])
    
    with tab1:
        render_create_problem_tab()
    
    with tab2:
        render_manage_problems_tab()
    
    with tab3:
        render_export_import_tab()
    
    with tab4:
        render_save_current_tab()

def render_create_problem_tab():
    """Render the create new problem tab"""
    
    st.subheader("Create New Custom Problem")
    
    with st.form("create_custom_problem"):
        # Basic information
        col1, col2 = st.columns(2)
        
        with col1:
            problem_name = st.text_input(
                "Problem Name*",
                placeholder="e.g., My Decision Problem",
                help="Give your problem a descriptive name"
            )
        
        with col2:
            problem_description = st.text_area(
                "Description*",
                placeholder="Describe what this decision problem is about...",
                help="Provide context for this decision problem"
            )
        
        # Alternatives
        st.subheader("Alternatives")
        n_alternatives = st.number_input(
            "Number of alternatives",
            min_value=2,
            max_value=10,
            value=3,
            help="How many options are you comparing?"
        )
        
        alternatives = []
        alt_cols = st.columns(min(n_alternatives, 3))
        for i in range(n_alternatives):
            col_idx = i % 3
            with alt_cols[col_idx]:
                alt_name = st.text_input(
                    f"Alternative {i+1}*",
                    value=f"Alternative {i+1}",
                    key=f"new_alt_{i}"
                )
                alternatives.append(alt_name)
        
        # Criteria
        st.subheader("Criteria")
        n_criteria = st.number_input(
            "Number of criteria",
            min_value=1,
            max_value=8,
            value=3,
            help="How many factors will you evaluate?"
        )
        
        criteria = []
        criterion_types = []
        
        for i in range(n_criteria):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                crit_name = st.text_input(
                    f"Criterion {i+1}*",
                    value=f"Criterion {i+1}",
                    key=f"new_crit_{i}"
                )
                criteria.append(crit_name)
            
            with col2:
                crit_type = st.selectbox(
                    "Type",
                    ["benefit", "cost"],
                    key=f"new_crit_type_{i}",
                    help="Benefit: higher is better, Cost: lower is better"
                )
                criterion_types.append(crit_type)
        
        # Decision Matrix
        st.subheader("Decision Matrix")
        st.write("Enter the performance values for each alternative on each criterion:")
        
        # Create a matrix input
        matrix_data = []
        for i in range(n_alternatives):
            row = []
            cols = st.columns(n_criteria)
            for j in range(n_criteria):
                with cols[j]:
                    value = st.number_input(
                        f"{alternatives[i]} - {criteria[j]}",
                        value=1.0,
                        key=f"matrix_{i}_{j}",
                        format="%.2f"
                    )
                    row.append(value)
            matrix_data.append(row)
        
        # Weights
        st.subheader("Criteria Weights")
        weight_method = st.radio(
            "Weight assignment method",
            ["Equal weights", "Custom weights"],
            help="Choose how to assign importance to criteria"
        )
        
        weights = []
        if weight_method == "Equal weights":
            equal_weight = 1.0 / n_criteria
            weights = [equal_weight] * n_criteria
            st.info(f"All criteria will have equal weight: {equal_weight:.3f}")
        else:
            st.write("Assign weights to each criterion (they will be normalized to sum to 1):")
            weight_cols = st.columns(min(n_criteria, 3))
            for i in range(n_criteria):
                col_idx = i % 3
                with weight_cols[col_idx]:
                    weight = st.number_input(
                        f"{criteria[i]}",
                        min_value=0.0,
                        max_value=1.0,
                        value=1.0/n_criteria,
                        step=0.01,
                        key=f"custom_weight_{i}",
                        format="%.3f"
                    )
                    weights.append(weight)
            
            # Normalize weights
            weight_sum = sum(weights)
            if weight_sum > 0:
                weights = [w / weight_sum for w in weights]
                st.info(f"Weights will be normalized. Sum: {weight_sum:.3f}")
        
        # Submit button
        submitted = st.form_submit_button("💾 Save Custom Problem", type="primary")
        
        if submitted:
            # Validate inputs
            if not problem_name.strip():
                st.error("Problem name is required")
            elif not problem_description.strip():
                st.error("Problem description is required")
            else:
                # Create decision matrix DataFrame
                decision_matrix = pd.DataFrame(
                    matrix_data,
                    index=alternatives,
                    columns=criteria
                )
                
                # Validate problem data
                errors = validate_problem_data(
                    alternatives, criteria, criterion_types, 
                    decision_matrix, weights
                )
                
                if errors:
                    st.error("Please fix the following issues:")
                    for error in errors:
                        st.write(f"• {error}")
                else:
                    # Save the problem
                    success = save_custom_problem(
                        problem_name, problem_description, alternatives,
                        criteria, criterion_types, decision_matrix, weights
                    )
                    
                    if success:
                        st.success(f"✅ Custom problem '{problem_name}' saved successfully!")
                        st.balloons()
                    else:
                        st.error("Failed to save custom problem")

def render_manage_problems_tab():
    """Render the manage existing problems tab"""
    
    st.subheader("Manage Existing Custom Problems")
    
    # Load custom problems
    custom_problems = load_custom_problems()
    
    if not custom_problems:
        st.info("No custom problems found. Create one in the 'Create New' tab!")
        return
    
    # Display custom problems
    for name, problem in custom_problems.items():
        with st.expander(f"📋 {name}", expanded=False):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Description:** {problem.get('description', 'No description')}")
                st.write(f"**Alternatives:** {len(problem.get('alternatives', []))}")
                st.write(f"**Criteria:** {len(problem.get('criteria', []))}")
                if 'created_date' in problem:
                    st.write(f"**Created:** {problem['created_date'][:10]}")
            
            with col2:
                if st.button(f"📂 Load", key=f"load_{name}"):
                    # Load this problem into session state
                    try:
                        # Convert back to the format expected by load_example_problem
                        problem_data = {
                            "alternatives": problem["alternatives"],
                            "criteria": problem["criteria"],
                            "criterion_types": problem["criterion_types"],
                            "weights": problem["weights"],
                            "decision_matrix": problem["decision_matrix"]
                        }
                        
                        # Load into session state
                        st.session_state.alternatives = problem_data["alternatives"]
                        st.session_state.criteria = problem_data["criteria"]
                        st.session_state.criterion_types = problem_data["criterion_types"]
                        st.session_state.weights = problem_data["weights"]
                        
                        if problem_data["decision_matrix"]:
                            st.session_state.decision_matrix = pd.DataFrame(
                                problem_data["decision_matrix"],
                                index=problem_data["alternatives"],
                                columns=problem_data["criteria"]
                            )
                        
                        st.session_state.example_loaded = name
                        st.session_state.results = None
                        
                        # Reset AHP matrices
                        from src.utils.session_state import reset_ahp_session_state
                        reset_ahp_session_state()
                        
                        st.success(f"✅ Loaded '{name}' successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error loading problem: {str(e)}")
            
            with col3:
                if st.button(f"🗑️ Delete", key=f"delete_{name}"):
                    if delete_custom_problem(name):
                        st.success(f"✅ Deleted '{name}' successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to delete problem")

def render_export_import_tab():
    """Render the export/import tab"""
    
    st.subheader("Export & Import Problems")
    
    # Export section
    st.write("### 📤 Export Problems")
    custom_problems = load_custom_problems()
    
    if custom_problems:
        selected_problem = st.selectbox(
            "Select problem to export",
            list(custom_problems.keys())
        )
        
        if st.button("📤 Export Problem"):
            json_data = export_custom_problem(selected_problem)
            if json_data:
                st.download_button(
                    label="💾 Download JSON File",
                    data=json_data,
                    file_name=f"{selected_problem.replace(' ', '_')}.json",
                    mime="application/json"
                )
    else:
        st.info("No custom problems available for export")
    
    st.divider()
    
    # Import section
    st.write("### 📥 Import Problems")
    
    import_method = st.radio(
        "Import method",
        ["Upload JSON file", "Paste JSON data"]
    )
    
    if import_method == "Upload JSON file":
        uploaded_file = st.file_uploader(
            "Choose a JSON file",
            type="json",
            help="Upload a previously exported problem file"
        )
        
        if uploaded_file is not None:
            try:
                json_data = uploaded_file.read().decode('utf-8')
                if st.button("📥 Import Problem"):
                    if import_custom_problem(json_data):
                        st.success("✅ Problem imported successfully!")
                        st.rerun()
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    else:
        json_input = st.text_area(
            "Paste JSON data",
            height=200,
            placeholder="Paste the exported JSON data here..."
        )
        
        if st.button("📥 Import Problem") and json_input.strip():
            if import_custom_problem(json_input):
                st.success("✅ Problem imported successfully!")
                st.rerun()

def render_save_current_tab():
    """Render the save current problem tab"""
    
    st.subheader("Save Current Problem as Custom Problem")
    
    # Check if there's a current problem
    if not hasattr(st.session_state, 'alternatives') or not st.session_state.alternatives:
        st.info("No current problem to save. Please define a problem first.")
        return
    
    # Show current problem summary
    st.write("### Current Problem Summary")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Alternatives:** {len(st.session_state.alternatives)}")
        for alt in st.session_state.alternatives:
            st.write(f"• {alt}")
    
    with col2:
        st.write(f"**Criteria:** {len(st.session_state.criteria)}")
        for i, crit in enumerate(st.session_state.criteria):
            crit_type = st.session_state.criterion_types[i] if i < len(st.session_state.criterion_types) else "benefit"
            st.write(f"• {crit} ({crit_type})")
    
    # Decision matrix preview
    if hasattr(st.session_state, 'decision_matrix') and st.session_state.decision_matrix is not None:
        st.write("### Decision Matrix")
        st.dataframe(st.session_state.decision_matrix, use_container_width=True)
    
    # Save form
    with st.form("save_current_problem"):
        problem_name = st.text_input(
            "Problem Name*",
            value=st.session_state.get('example_loaded', 'My Custom Problem'),
            help="Give this problem a unique name"
        )
        
        problem_description = st.text_area(
            "Description*",
            value="Custom problem created from current session",
            help="Describe this decision problem"
        )
        
        submitted = st.form_submit_button("💾 Save as Custom Problem", type="primary")
        
        if submitted:
            if not problem_name.strip():
                st.error("Problem name is required")
            elif not problem_description.strip():
                st.error("Problem description is required")
            else:
                # Save current problem
                success = save_custom_problem(
                    problem_name,
                    problem_description,
                    st.session_state.alternatives,
                    st.session_state.criteria,
                    st.session_state.criterion_types,
                    st.session_state.decision_matrix,
                    st.session_state.weights
                )
                
                if success:
                    st.success(f"✅ Current problem saved as '{problem_name}'!")
                    st.balloons()
                else:
                    st.error("Failed to save current problem")
