"""
Debug script to test custom weights issue
"""

import streamlit as st

st.title("Debug Custom Weights Issue")

# Simulate the form structure
with st.form("debug_form"):
    n_criteria = 3
    criteria = ["Criterion 1", "Criterion 2", "Criterion 3"]
    
    st.write(f"Number of criteria: {n_criteria}")
    st.write(f"Criteria: {criteria}")
    
    # Weight method selection
    weight_method = st.radio(
        "Weight assignment method",
        ["Equal weights", "Custom weights"],
        help="Choose how to assign importance to criteria"
    )
    
    st.write(f"Selected method: {weight_method}")
    
    # Weights section - EXACTLY like in the original code
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
    
    # Debug output
    st.write("### Debug Information")
    st.write(f"Weight method: {weight_method}")
    st.write(f"Raw weights: {weights}")
    st.write(f"Number of weights: {len(weights)}")
    
    submitted = st.form_submit_button("Test Submit")
    
    if submitted:
        st.write("### Form Submitted!")
        st.write(f"Final weights: {weights}")
        st.write(f"Weight sum: {sum(weights)}")
        st.write(f"Weight method was: {weight_method}")
