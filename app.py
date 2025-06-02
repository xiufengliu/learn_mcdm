"""
MCDM Learning Tool - Main Streamlit Application
Interactive tool for learning Multi-Criteria Decision Making methods
"""

import streamlit as st
import pandas as pd
import numpy as np
from src.ui.sidebar import render_sidebar
from src.ui.main_content import render_main_content
from src.utils.session_state import initialize_session_state
from config.settings import APP_CONFIG

def main():
    """Main application function"""
    
    # Page configuration
    st.set_page_config(
        page_title=APP_CONFIG["page_title"],
        page_icon=APP_CONFIG["page_icon"],
        layout=APP_CONFIG["layout"],
        initial_sidebar_state=APP_CONFIG["initial_sidebar_state"]
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .method-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .result-highlight {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🎯 MCDM Learning Tool</h1>', unsafe_allow_html=True)
    st.markdown("**Learn Multi-Criteria Decision Making through interactive examples**")
    
    # Sidebar for method selection and inputs
    render_sidebar()
    
    # Main content area
    render_main_content()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tip**: Start with the Simple Additive Weighting (SAW) method if you're new to MCDM!"
    )

if __name__ == "__main__":
    main()
