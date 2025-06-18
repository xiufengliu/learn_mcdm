"""
Test script to reproduce the custom weights issue
"""

import streamlit as st
import pandas as pd

st.title("Test Custom Weights Issue")

with st.form("test_form"):
    # Number of criteria
    n_criteria = st.number_input("Number of criteria", min_value=1, max_value=5, value=3)
    
    # Build criteria list
    criteria = []
    for i in range(n_criteria):
        crit_name = st.text_input(f"Criterion {i+1}", value=f"Criterion {i+1}", key=f"crit_{i}")
        criteria.append(crit_name)
    
    # Weight method selection
    weight_method = st.radio("Weight method", ["Equal weights", "Custom weights"])
    
    # Weights section
    weights = []
    if weight_method == "Equal weights":
        equal_weight = 1.0 / n_criteria
        weights = [equal_weight] * n_criteria
        st.info(f"Equal weights: {weights}")
    else:
        st.write("Custom weights:")
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
                    key=f"weight_{i}",
                    format="%.3f"
                )
                weights.append(weight)
        
        # Normalize weights
        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w / weight_sum for w in weights]
            st.info(f"Normalized weights: {weights}")
    
    submitted = st.form_submit_button("Test Submit")
    
    if submitted:
        st.write("Form submitted!")
        st.write(f"Criteria: {criteria}")
        st.write(f"Number of criteria: {len(criteria)}")
        st.write(f"Weights: {weights}")
        st.write(f"Number of weights: {len(weights)}")
        st.write(f"Weight method: {weight_method}")
        
        # Check if lengths match
        if len(weights) != len(criteria):
            st.error(f"Mismatch! Criteria: {len(criteria)}, Weights: {len(weights)}")
        else:
            st.success("Lengths match!")
