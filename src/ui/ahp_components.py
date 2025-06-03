"""
AHP-specific UI components for pairwise comparisons
"""

import streamlit as st
import pandas as pd
import numpy as np
from src.methods.ahp import AHP

def render_ahp_interface():
    """Render the AHP-specific interface for pairwise comparisons"""

    st.header("🔄 AHP Pairwise Comparisons")

    # Ensure AHP session state is properly initialized
    from src.utils.session_state import initialize_ahp_session_state, update_ahp_matrices_structure
    initialize_ahp_session_state()
    update_ahp_matrices_structure()

    # Create tabs for different comparison types
    tab1, tab2, tab3 = st.tabs(["🎯 Criteria Comparisons", "📊 Alternative Comparisons", "📋 Results"])

    with tab1:
        render_criteria_comparisons()

    with tab2:
        render_alternative_comparisons()

    with tab3:
        render_ahp_results()

def render_criteria_comparisons():
    """Render criteria pairwise comparison interface"""
    
    st.subheader("Compare Criteria Importance")
    st.write("Compare the relative importance of each pair of criteria using Saaty's 1-9 scale:")
    
    # Show Saaty scale reference
    with st.expander("📖 Saaty's Comparison Scale", expanded=False):
        scale_df = pd.DataFrame({
            'Intensity': [1, 3, 5, 7, 9, 2, 4, 6, 8],
            'Definition': [
                'Equal importance',
                'Moderate importance',
                'Strong importance', 
                'Very strong importance',
                'Extreme importance',
                'Intermediate value',
                'Intermediate value',
                'Intermediate value',
                'Intermediate value'
            ],
            'Explanation': [
                'Two criteria contribute equally',
                'Experience slightly favors one over another',
                'Experience strongly favors one over another',
                'One criterion is very strongly favored',
                'The evidence favoring one is of highest order',
                'When compromise is needed',
                'When compromise is needed',
                'When compromise is needed',
                'When compromise is needed'
            ]
        })
        st.dataframe(scale_df, use_container_width=True)
    
    criteria = st.session_state.criteria
    n_criteria = len(criteria)

    if n_criteria < 2:
        st.warning("At least 2 criteria are needed for pairwise comparisons.")
        return

    # Ensure matrix has correct dimensions
    if (st.session_state.ahp_criteria_matrix.shape != (n_criteria, n_criteria)):
        st.session_state.ahp_criteria_matrix = np.ones((n_criteria, n_criteria))
    
    # Create comparison inputs
    st.write("**Make pairwise comparisons:**")
    
    comparisons = []
    for i in range(n_criteria):
        for j in range(i + 1, n_criteria):
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                st.write(f"**{criteria[i]}**")
            
            with col2:
                # Use slider for comparison
                try:
                    current_value = st.session_state.ahp_criteria_matrix[i, j]
                    # Ensure the value is valid (with tolerance for floating point comparison)
                    valid_values = [1/9, 1/7, 1/5, 1/3, 1, 3, 5, 7, 9]
                    if not any(abs(current_value - v) < 1e-10 for v in valid_values):
                        current_value = 1
                except (IndexError, KeyError):
                    current_value = 1

                comparison_value = st.select_slider(
                    f"vs",
                    options=[1/9, 1/7, 1/5, 1/3, 1, 3, 5, 7, 9],
                    value=current_value,
                    format_func=lambda x: f"{x:.2f}" if x < 1 else f"{int(x)}",
                    key=f"criteria_comp_{i}_{j}",
                    help=f"How important is {criteria[i]} compared to {criteria[j]}?"
                )
                
                # Update matrix
                st.session_state.ahp_criteria_matrix[i, j] = comparison_value
                st.session_state.ahp_criteria_matrix[j, i] = 1.0 / comparison_value
                
                comparisons.append(comparison_value)
            
            with col3:
                st.write(f"**{criteria[j]}**")
    
    # Show current comparison matrix
    st.subheader("Current Comparison Matrix")
    criteria_matrix_df = pd.DataFrame(
        st.session_state.ahp_criteria_matrix,
        index=criteria,
        columns=criteria
    )
    st.dataframe(criteria_matrix_df.round(3), use_container_width=True)
    
    # Calculate and show consistency
    if st.button("🔍 Check Consistency", key="check_criteria_consistency"):
        ahp = AHP(alternatives=st.session_state.alternatives, criteria=criteria)
        ahp.set_criteria_comparison_matrix(st.session_state.ahp_criteria_matrix)
        
        weights, cr = ahp.calculate_priority_weights(st.session_state.ahp_criteria_matrix)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Consistency Ratio", f"{cr:.4f}")
            if cr < 0.1:
                st.success("✅ Acceptable consistency")
            elif cr < 0.2:
                st.warning("⚠️ Marginal consistency - consider reviewing")
            else:
                st.error("❌ Poor consistency - revision recommended")
        
        with col2:
            weights_df = pd.DataFrame({
                'Criterion': criteria,
                'Weight': weights
            })
            st.write("**Derived Weights:**")
            st.dataframe(weights_df, use_container_width=True)

def render_alternative_comparisons():
    """Render alternative pairwise comparison interface"""
    
    st.subheader("Compare Alternatives for Each Criterion")
    
    criteria = st.session_state.criteria
    alternatives = st.session_state.alternatives
    n_alternatives = len(alternatives)
    
    if n_alternatives < 2:
        st.warning("At least 2 alternatives are needed for pairwise comparisons.")
        return
    
    # Select criterion to compare
    selected_criterion = st.selectbox(
        "Select criterion to compare alternatives:",
        criteria,
        key="ahp_selected_criterion"
    )
    
    # Initialize matrix for this criterion if not exists or wrong size
    if (selected_criterion not in st.session_state.ahp_alternative_matrices or
        st.session_state.ahp_alternative_matrices[selected_criterion].shape != (n_alternatives, n_alternatives)):
        st.session_state.ahp_alternative_matrices[selected_criterion] = np.ones((n_alternatives, n_alternatives))
    
    st.write(f"**Compare alternatives with respect to: {selected_criterion}**")
    
    # Create comparison inputs for alternatives
    for i in range(n_alternatives):
        for j in range(i + 1, n_alternatives):
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                st.write(f"**{alternatives[i]}**")
            
            with col2:
                try:
                    current_value = st.session_state.ahp_alternative_matrices[selected_criterion][i, j]
                    # Ensure the value is valid (with tolerance for floating point comparison)
                    valid_values = [1/9, 1/7, 1/5, 1/3, 1, 3, 5, 7, 9]
                    if not any(abs(current_value - v) < 1e-10 for v in valid_values):
                        current_value = 1
                except (IndexError, KeyError):
                    current_value = 1

                comparison_value = st.select_slider(
                    f"vs",
                    options=[1/9, 1/7, 1/5, 1/3, 1, 3, 5, 7, 9],
                    value=current_value,
                    format_func=lambda x: f"{x:.2f}" if x < 1 else f"{int(x)}",
                    key=f"alt_comp_{selected_criterion}_{i}_{j}",
                    help=f"How much better is {alternatives[i]} compared to {alternatives[j]} for {selected_criterion}?"
                )
                
                # Update matrix
                st.session_state.ahp_alternative_matrices[selected_criterion][i, j] = comparison_value
                st.session_state.ahp_alternative_matrices[selected_criterion][j, i] = 1.0 / comparison_value
            
            with col3:
                st.write(f"**{alternatives[j]}**")
    
    # Show current comparison matrix for selected criterion
    st.subheader(f"Comparison Matrix for {selected_criterion}")
    alt_matrix_df = pd.DataFrame(
        st.session_state.ahp_alternative_matrices[selected_criterion],
        index=alternatives,
        columns=alternatives
    )
    st.dataframe(alt_matrix_df.round(3), use_container_width=True)
    
    # Check consistency for this criterion
    if st.button(f"🔍 Check Consistency for {selected_criterion}", key=f"check_alt_consistency_{selected_criterion}"):
        ahp = AHP(alternatives=alternatives, criteria=criteria)
        weights, cr = ahp.calculate_priority_weights(st.session_state.ahp_alternative_matrices[selected_criterion])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Consistency Ratio", f"{cr:.4f}")
            if cr < 0.1:
                st.success("✅ Acceptable consistency")
            elif cr < 0.2:
                st.warning("⚠️ Marginal consistency")
            else:
                st.error("❌ Poor consistency")
        
        with col2:
            weights_df = pd.DataFrame({
                'Alternative': alternatives,
                'Weight': weights
            })
            st.write("**Derived Weights:**")
            st.dataframe(weights_df, use_container_width=True)
    
    # Show progress
    completed_criteria = len(st.session_state.ahp_alternative_matrices)
    total_criteria = len(criteria)
    st.progress(completed_criteria / total_criteria)
    st.write(f"Completed: {completed_criteria}/{total_criteria} criteria")

def render_ahp_results():
    """Render AHP calculation results"""
    
    st.subheader("AHP Analysis Results")
    
    # Check if all comparisons are complete
    criteria = st.session_state.criteria
    alternatives = st.session_state.alternatives
    
    missing_comparisons = []
    for criterion in criteria:
        if criterion not in st.session_state.ahp_alternative_matrices:
            missing_comparisons.append(criterion)
    
    if missing_comparisons:
        st.warning(f"Please complete pairwise comparisons for: {', '.join(missing_comparisons)}")
        return
    
    # Calculate AHP results
    if st.button("🚀 Calculate AHP Results", type="primary"):
        try:
            # Create AHP instance
            ahp = AHP(alternatives=alternatives, criteria=criteria)
            
            # Set criteria comparison matrix
            ahp.set_criteria_comparison_matrix(st.session_state.ahp_criteria_matrix)
            
            # Set alternative comparison matrices
            for criterion in criteria:
                if criterion in st.session_state.ahp_alternative_matrices:
                    ahp.set_alternative_comparison_matrix(
                        criterion, 
                        st.session_state.ahp_alternative_matrices[criterion]
                    )
            
            # Calculate results
            results = ahp.calculate()
            
            # Store in session state
            st.session_state.results = results
            st.session_state.method_instance = ahp
            
            st.success("✅ AHP results calculated successfully!")
            
        except Exception as e:
            st.error(f"Error calculating AHP results: {str(e)}")
    
    # Show results if available
    if st.session_state.results and st.session_state.results.get('method') == 'AHP':
        display_ahp_results()

def display_ahp_results():
    """Display detailed AHP results"""
    
    method_instance = st.session_state.method_instance
    
    # Results summary
    st.subheader("📊 Final Rankings")
    results_df = method_instance.get_results_dataframe()
    
    # Highlight the best alternative
    def highlight_best(row):
        if row['Rank'] == 1:
            return ['background-color: #90EE90'] * len(row)
        return [''] * len(row)
    
    styled_df = results_df.style.apply(highlight_best, axis=1)
    st.dataframe(styled_df, use_container_width=True)
    
    # Best alternative callout
    best_alternative = results_df.iloc[0]['Alternative']
    best_score = results_df.iloc[0]['Score']
    st.success(f"🏆 **Best Alternative:** {best_alternative} (Score: {best_score:.4f})")
    
    # Consistency analysis
    st.subheader("🔍 Consistency Analysis")
    consistency_results = method_instance.is_consistent()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if consistency_results['overall_consistent']:
            st.success("✅ All comparisons are consistent")
        else:
            st.warning("⚠️ Some comparisons may need review")
    
    with col2:
        consistency_df = pd.DataFrame([
            {'Matrix': name, 'Consistency Ratio': f"{ratio:.4f}"}
            for name, ratio in consistency_results['all_ratios'].items()
        ])
        st.dataframe(consistency_df, use_container_width=True)
