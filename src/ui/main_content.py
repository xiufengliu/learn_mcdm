"""
Main content area UI components
"""

import streamlit as st
import pandas as pd
import numpy as np
from src.methods.saw import SAW
from src.methods.wpm import WPM
from src.methods.topsis import TOPSIS
from src.methods.ahp import AHP
from src.utils.validation import (
    validate_decision_matrix, 
    validate_alternatives_and_criteria,
    display_validation_messages
)
try:
    from src.visualization.charts import (
        create_results_chart,
        create_comparison_chart,
        create_topsis_analysis_chart,
        create_criteria_impact_chart
    )
except ImportError:
    # Fallback if visualization module is not available
    def create_results_chart(*args, **kwargs):
        return None
    def create_comparison_chart(*args, **kwargs):
        return None
    def create_topsis_analysis_chart(*args, **kwargs):
        return None
    def create_criteria_impact_chart(*args, **kwargs):
        return None
from config.settings import MCDM_METHODS

def render_main_content():
    """Render the main content area"""

    # Check if custom problems interface should be shown
    if st.session_state.get('show_custom_problems', False):
        from src.ui.custom_problems_ui import render_custom_problems_interface
        render_custom_problems_interface()

        # Add a button to go back
        if st.button("← Back to Main Interface"):
            st.session_state.show_custom_problems = False
            st.rerun()
        return

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Input", "🧮 Results", "📈 Visualizations", "📚 Learn"])

    with tab1:
        render_data_input_tab()

    with tab2:
        render_results_tab()

    with tab3:
        render_visualizations_tab()

    with tab4:
        render_learning_tab()

def render_data_input_tab():
    """Render data input section"""

    # Check if AHP is selected
    if st.session_state.selected_method == 'AHP':
        # Import and render AHP interface
        from src.ui.ahp_components import render_ahp_interface
        render_ahp_interface()
        return

    st.header("📊 Decision Matrix Input")

    # Validation
    is_valid, errors = validate_alternatives_and_criteria(
        st.session_state.alternatives,
        st.session_state.criteria
    )

    if not is_valid:
        st.error("Please fix the following issues in the sidebar:")
        display_validation_messages(errors)
        return

    # Decision matrix input
    st.subheader("Enter Performance Values")
    st.write("Enter the performance value of each alternative for each criterion:")

    # Create editable dataframe
    edited_df = st.data_editor(
        st.session_state.decision_matrix,
        use_container_width=True,
        num_rows="fixed",
        key="decision_matrix_editor"
    )

    # Update session state
    st.session_state.decision_matrix = edited_df

    # Validate decision matrix
    matrix_valid, matrix_errors = validate_decision_matrix(edited_df)
    if matrix_errors:
        display_validation_messages(matrix_errors, "warning")

    # Show current problem summary
    st.subheader("Problem Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Alternatives", len(st.session_state.alternatives))

    with col2:
        st.metric("Criteria", len(st.session_state.criteria))

    with col3:
        st.metric("Method", st.session_state.selected_method)

    # Show criteria information
    criteria_info = pd.DataFrame({
        'Criterion': st.session_state.criteria,
        'Type': st.session_state.criterion_types,
        'Weight': [f"{w:.3f}" for w in st.session_state.weights]
    })

    st.subheader("Criteria Information")
    st.dataframe(criteria_info, use_container_width=True)

def render_results_tab():
    """Render results section"""

    st.header("🧮 MCDM Results")

    # For AHP, results are handled in the AHP interface
    if st.session_state.selected_method == 'AHP':
        if st.session_state.results and st.session_state.results.get('method') == 'AHP':
            display_results()
        else:
            st.info("👈 Please complete the pairwise comparisons in the Data Input tab and calculate AHP results.")
        return

    # Validate inputs before calculation for other methods
    matrix_valid, matrix_errors = validate_decision_matrix(st.session_state.decision_matrix)

    if not matrix_valid:
        st.error("Cannot calculate results due to data issues:")
        display_validation_messages(matrix_errors)
        return

    # Calculate button
    if st.button("🚀 Calculate Results", type="primary"):
        calculate_results()

    # Show results if available
    if st.session_state.results:
        display_results()

def calculate_results():
    """Calculate MCDM results based on selected method"""
    
    try:
        # Get method class
        method_class = get_method_class(st.session_state.selected_method)
        
        if method_class is None:
            st.error(f"Method {st.session_state.selected_method} not implemented yet")
            return
        
        # Create method instance
        method = method_class(
            decision_matrix=st.session_state.decision_matrix,
            weights=st.session_state.weights,
            criterion_types=st.session_state.criterion_types
        )
        
        # Calculate results
        results = method.calculate()
        
        # Store results and method instance in session state
        st.session_state.results = results
        st.session_state.method_instance = method
        
        st.success("✅ Results calculated successfully!")
        
    except Exception as e:
        st.error(f"Error calculating results: {str(e)}")

def get_method_class(method_name):
    """Get the method class based on method name"""
    method_classes = {
        'SAW': SAW,
        'WPM': WPM,
        'TOPSIS': TOPSIS,
        'AHP': AHP,
        # Add more methods as they are implemented
    }
    return method_classes.get(method_name)

def display_results():
    """Display calculation results"""
    
    method_instance = st.session_state.get('method_instance')
    if not method_instance:
        return
    
    # Results summary
    st.subheader("📊 Results Summary")
    
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
    
    # Show intermediate steps if requested
    if st.session_state.show_intermediate_steps:
        show_calculation_steps()

def show_calculation_steps():
    """Show detailed calculation steps"""

    method_instance = st.session_state.get('method_instance')
    if not method_instance:
        return

    st.subheader("🔍 Step-by-Step Calculation")

    explanation = method_instance.get_step_by_step_explanation()

    for step in explanation['steps']:
        with st.expander(f"Step {step['step_number']}: {step['title']}", expanded=False):
            st.write(step['description'])

            if 'formula' in step:
                render_step_formula(step['formula'], step.get('step_number'), st.session_state.selected_method)

            if isinstance(step['matrix'], pd.DataFrame):
                st.dataframe(step['matrix'], use_container_width=True)
            else:
                st.write(step['matrix'])

def render_step_formula(formula_text, step_number, method_name):
    """Render step formulas with LaTeX when possible"""

    # Convert formula text to LaTeX based on content and method
    if method_name == 'TOPSIS':
        render_topsis_step_formula(formula_text, step_number)
    elif method_name == 'SAW':
        render_saw_step_formula(formula_text, step_number)
    elif method_name == 'WPM':
        render_wpm_step_formula(formula_text, step_number)
    elif method_name == 'AHP':
        render_ahp_step_formula(formula_text, step_number)
    else:
        # Fallback to code format
        st.code(formula_text, language='text')

def render_topsis_step_formula(formula_text, step_number):
    """Render TOPSIS step formulas with LaTeX"""

    if step_number == 1:
        st.write("**Formula:**")
        st.latex(r"X = \begin{bmatrix} x_{11} & x_{12} & \cdots & x_{1n} \\ x_{21} & x_{22} & \cdots & x_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ x_{m1} & x_{m2} & \cdots & x_{mn} \end{bmatrix}")
    elif step_number == 2:
        st.write("**Vector Normalization Formula:**")
        st.latex(r"r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{i=1}^{m} x_{ij}^2}}")
    elif step_number == 3:
        st.write("**Weighted Normalized Formula:**")
        st.latex(r"v_{ij} = w_j \times r_{ij}")
    elif step_number == 4:
        st.write("**Ideal Solutions:**")
        st.latex(r"\text{For benefit criteria: } PIS_j = \max_i(v_{ij}), \quad NIS_j = \min_i(v_{ij})")
        st.latex(r"\text{For cost criteria: } PIS_j = \min_i(v_{ij}), \quad NIS_j = \max_i(v_{ij})")
    elif step_number == 5:
        st.write("**Separation Measures:**")
        st.latex(r"S_i^+ = \sqrt{\sum_{j=1}^{n} (v_{ij} - PIS_j)^2}")
        st.latex(r"S_i^- = \sqrt{\sum_{j=1}^{n} (v_{ij} - NIS_j)^2}")
    elif step_number == 6:
        st.write("**Relative Closeness:**")
        st.latex(r"C_i^* = \frac{S_i^-}{S_i^+ + S_i^-}")
    else:
        st.code(formula_text, language='text')

def render_saw_step_formula(formula_text, step_number):
    """Render SAW step formulas with LaTeX"""

    if step_number == 1:
        st.write("**Decision Matrix:**")
        st.latex(r"X = \begin{bmatrix} x_{11} & x_{12} & \cdots & x_{1n} \\ x_{21} & x_{22} & \cdots & x_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ x_{m1} & x_{m2} & \cdots & x_{mn} \end{bmatrix}")
    elif step_number == 2:
        st.write("**Normalization Formulas (Min-Max):**")
        st.latex(r"\text{Benefit criteria: } r_{ij} = \frac{x_{ij} - \min_j(x_{ij})}{\max_j(x_{ij}) - \min_j(x_{ij})}")
        st.latex(r"\text{Cost criteria: } r_{ij} = \frac{\max_j(x_{ij}) - x_{ij}}{\max_j(x_{ij}) - \min_j(x_{ij})}")
    elif step_number == 3:
        st.write("**Weighted Values:**")
        st.latex(r"v_{ij} = w_j \times r_{ij}")
    elif step_number == 4:
        st.write("**Final Scores:**")
        st.latex(r"S_i = \sum_{j=1}^{n} v_{ij}")
    else:
        st.code(formula_text, language='text')

def render_wpm_step_formula(formula_text, step_number):
    """Render WPM step formulas with LaTeX"""

    if step_number == 1:
        st.write("**Decision Matrix:**")
        st.latex(r"X = \begin{bmatrix} x_{11} & x_{12} & \cdots & x_{1n} \\ x_{21} & x_{22} & \cdots & x_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ x_{m1} & x_{m2} & \cdots & x_{mn} \end{bmatrix}")
    elif step_number == 2:
        st.write("**Cost Criteria Processing:**")
        st.latex(r"\text{For cost criteria: } p_{ij} = \frac{1}{x_{ij}}")
        st.latex(r"\text{For benefit criteria: } p_{ij} = x_{ij}")
    elif step_number == 3:
        st.write("**Weighted Values (Raised to Powers):**")
        st.latex(r"v_{ij} = (p_{ij})^{w_j}")
    elif step_number == 4:
        st.write("**Final Product Scores:**")
        st.latex(r"S_i = \prod_{j=1}^{n} v_{ij}")
    else:
        st.code(formula_text, language='text')

def render_ahp_step_formula(formula_text, step_number):
    """Render AHP step formulas with LaTeX"""

    if step_number == 1:
        st.write("**Criteria Pairwise Comparison:**")
        st.latex(r"A_{ij} = \text{importance of criterion } i \text{ relative to criterion } j")
        st.latex(r"A = \begin{bmatrix} 1 & a_{12} & \cdots & a_{1n} \\ 1/a_{12} & 1 & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ 1/a_{1n} & 1/a_{2n} & \cdots & 1 \end{bmatrix}")
    elif step_number == 2:
        st.write("**Priority Vector Calculation:**")
        st.latex(r"A \mathbf{w} = \lambda_{max} \mathbf{w}")
        st.write(r"Where $\mathbf{w}$ is the principal eigenvector (criteria weights)")
    elif step_number == 3:
        st.write("**Alternative Pairwise Comparisons:**")
        st.latex(r"A^{(j)}_{ik} = \text{preference of alternative } i \text{ over } k \text{ for criterion } j")
    elif step_number == 4:
        st.write("**Final Synthesis:**")
        st.latex(r"S_i = \sum_{j=1}^{n} w_j \times w_{ij}")
        st.write("Where:")
        st.write(r"- $w_j$ = weight of criterion $j$")
        st.write(r"- $w_{ij}$ = weight of alternative $i$ for criterion $j$")
    else:
        st.code(formula_text, language='text')

def render_visualizations_tab():
    """Render visualizations section"""
    
    st.header("📈 Visualizations")
    
    if not st.session_state.results:
        st.info("👆 Calculate results first to see visualizations")
        return
    
    method_instance = st.session_state.get('method_instance')
    if not method_instance:
        return
    
    # Create tabs for different visualization types
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 Results", "🎯 Method Analysis", "📈 Criteria Analysis"])

    with viz_tab1:
        # Results chart
        st.subheader("Alternative Scores")
        results_df = method_instance.get_results_dataframe()

        fig = create_results_chart(results_df, st.session_state.selected_method)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(results_df.set_index('Alternative')['Score'])

        # Comparison chart (radar chart for top alternatives)
        if len(st.session_state.alternatives) <= 5:  # Only for small number of alternatives
            st.subheader("Multi-Criteria Comparison")

            comparison_fig = create_comparison_chart(
                st.session_state.decision_matrix,
                st.session_state.criteria,
                st.session_state.alternatives[:3]  # Top 3 alternatives
            )
            if comparison_fig:
                st.plotly_chart(comparison_fig, use_container_width=True)
            else:
                st.info("Advanced visualizations require plotly. Showing basic chart instead.")
                st.line_chart(st.session_state.decision_matrix.T)

    with viz_tab2:
        # Method-specific analysis
        if st.session_state.selected_method == 'TOPSIS':
            st.subheader("TOPSIS Distance Analysis")

            # Check if method has been executed and has intermediate steps
            if (method_instance and
                hasattr(method_instance, 'intermediate_steps') and
                method_instance.intermediate_steps and
                's_plus' in method_instance.intermediate_steps):

                topsis_fig = create_topsis_analysis_chart(method_instance)
                if topsis_fig:
                    st.plotly_chart(topsis_fig, use_container_width=True)

                    st.info("""
                    **Understanding TOPSIS Analysis:**
                    - **Distance to PIS (S+)**: Lower is better (closer to ideal)
                    - **Distance to NIS (S-)**: Higher is better (farther from anti-ideal)
                    - **Relative Closeness (C*)**: Higher is better (final score)
                    """)
                else:
                    st.info("Unable to generate TOPSIS visualization. Please recalculate results.")
            else:
                st.info("TOPSIS analysis will be available after calculating results.")

        else:
            st.subheader("Method Performance Analysis")

            # Show normalized vs original data comparison
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Original Data**")
                st.dataframe(st.session_state.decision_matrix, use_container_width=True)

            with col2:
                if hasattr(method_instance, 'intermediate_steps'):
                    if 'normalized_matrix' in method_instance.intermediate_steps:
                        st.write("**Normalized Data**")
                        st.dataframe(method_instance.intermediate_steps['normalized_matrix'], use_container_width=True)

    with viz_tab3:
        # Criteria analysis
        st.subheader("Criteria Impact Analysis")

        criteria_fig = create_criteria_impact_chart(
            st.session_state.decision_matrix,
            st.session_state.weights,
            st.session_state.criterion_types
        )
        if criteria_fig:
            st.plotly_chart(criteria_fig, use_container_width=True)

            st.info("""
            **Understanding Criteria Impact:**
            - **Weight**: The importance assigned to each criterion
            - **Impact**: Weight × Variability - shows which criteria have the most influence on the decision
            - **Color coding**: Blue = Benefit criteria, Red = Cost criteria
            """)
        else:
            # Fallback visualization
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Criteria Weights**")
                weights_df = pd.DataFrame({
                    'Criterion': st.session_state.criteria,
                    'Weight': st.session_state.weights,
                    'Type': st.session_state.criterion_types
                })
                st.dataframe(weights_df, use_container_width=True)

            with col2:
                st.write("**Weight Distribution**")
                st.bar_chart(weights_df.set_index('Criterion')['Weight'])

def render_learning_tab():
    """Render learning/educational content"""

    st.header("📚 Learn About MCDM")

    # Create sub-tabs for different learning content
    learn_tab1, learn_tab2 = st.tabs(["📖 Method Guide", "📚 Learning Resources"])

    with learn_tab1:
        render_method_guide()

    with learn_tab2:
        from src.ui.learning_resources import render_learning_resources
        render_learning_resources()

def render_method_guide():
    """Render the method guide content"""

    selected_method = st.session_state.selected_method

    # Get method description
    method_class = get_method_class(selected_method)
    if method_class and hasattr(method_class, 'get_method_description'):
        method_info = method_class.get_method_description()

        # Method overview
        st.subheader(f"📖 {method_info['name']}")
        st.write(method_info['description'])

        # When to use
        st.subheader("🎯 When to Use")
        for point in method_info['when_to_use']:
            st.write(f"• {point}")

        # Advantages and disadvantages
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Advantages")
            for advantage in method_info['advantages']:
                st.write(f"• {advantage}")

        with col2:
            st.subheader("⚠️ Disadvantages")
            for disadvantage in method_info['disadvantages']:
                st.write(f"• {disadvantage}")

        # Mathematical foundation
        if 'mathematical_foundation' in method_info:
            st.subheader("🧮 Mathematical Foundation")
            render_mathematical_foundation(selected_method, method_info)

    else:
        st.info(f"Educational content for {selected_method} is being prepared...")

    # General MCDM information
    st.subheader("🌟 About Multi-Criteria Decision Making")
    st.write("""
    Multi-Criteria Decision Making (MCDM) is a branch of operations research that deals with
    finding optimal results in complex scenarios including various indicators, conflicting
    objectives and criteria.

    **Key Concepts:**
    - **Alternatives**: The different options or choices available
    - **Criteria**: The factors or attributes used to evaluate alternatives
    - **Weights**: The relative importance of each criterion
    - **Decision Matrix**: A table showing the performance of each alternative on each criterion
    """)

    # Method comparison table
    st.subheader("📊 Method Comparison")

    comparison_data = {
        'Method': ['SAW', 'WPM', 'TOPSIS', 'AHP'],
        'Complexity': ['Beginner', 'Beginner', 'Intermediate', 'Advanced'],
        'Input Type': ['Decision Matrix', 'Decision Matrix', 'Decision Matrix', 'Pairwise Comparisons'],
        'Normalization': ['Linear', 'None/Reciprocal', 'Vector', 'Eigenvalue'],
        'Aggregation': ['Weighted Sum', 'Weighted Product', 'Distance-based', 'Hierarchical'],
        'Consistency Check': ['No', 'No', 'No', 'Yes'],
        'Best For': [
            'Simple problems',
            'Avoiding rank reversal',
            'Ideal point analysis',
            'Subjective judgments'
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

    # When to use which method
    st.subheader("🎯 When to Use Each Method")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**SAW (Simple Additive Weighting)**")
        st.write("• First-time MCDM users")
        st.write("• Transparent decision process needed")
        st.write("• All criteria easily comparable")
        st.write("• Quick analysis required")

        st.write("**TOPSIS**")
        st.write("• Want to consider ideal solutions")
        st.write("• Need robust ranking method")
        st.write("• Dealing with conflicting criteria")
        st.write("• Geometric interpretation preferred")

    with col2:
        st.write("**WPM (Weighted Product Method)**")
        st.write("• Avoiding rank reversal issues")
        st.write("• Multiplicative relationships exist")
        st.write("• Dimensionally consistent results needed")
        st.write("• Zero values are problematic")

        st.write("**AHP (Analytic Hierarchy Process)**")
        st.write("• Subjective criteria importance")
        st.write("• Multiple stakeholders involved")
        st.write("• Consistency checking required")
        st.write("• Complex hierarchical problems")

    # Example problems showcase
    st.subheader("🌟 Featured Example Problems")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**🚗 Car Selection**")
        st.write("Classic consumer choice problem comparing vehicles on price, fuel economy, safety, performance, and comfort.")

        st.write("**🏭 Supplier Selection**")
        st.write("Business decision comparing suppliers on cost, quality, delivery time, and reliability.")

        st.write("**🎓 University Selection**")
        st.write("Educational choice comparing universities on tuition, ranking, location, research, and campus life.")

    with col2:
        st.write("**🌱 Renewable Energy Selection**")
        st.write("Environmental decision comparing energy technologies (Solar, Wind, Hydro, Biomass, Geothermal) on cost, output, environmental impact, reliability, maintenance, and land use.")

        st.write("**💻 Software Selection (AHP)**")
        st.write("Technology decision using pairwise comparisons to evaluate project management software options.")

        st.write("**📝 Custom Problems**")
        st.write("Create your own decision problems with the custom problems manager!")

    # Tips for beginners
    st.subheader("💡 Tips for Beginners")
    st.write("""
    1. **Start Simple**: Begin with SAW method to understand basic concepts
    2. **Try Examples**: Load the renewable energy example to see a real-world application
    3. **Define Clear Criteria**: Make sure your criteria are measurable and relevant
    4. **Consider Criterion Types**: Distinguish between benefit (higher is better) and cost (lower is better) criteria
    5. **Weight Carefully**: Spend time thinking about the relative importance of criteria
    6. **Validate Results**: Check if the results make intuitive sense
    7. **Try Different Methods**: Compare results from different MCDM methods
    8. **Check Consistency**: For AHP, ensure consistency ratios are acceptable
    9. **Create Custom Problems**: Use the custom problems manager to save your own scenarios
    10. **Sensitivity Analysis**: Test how changes in weights affect rankings
    """)

def render_mathematical_foundation(method_name, method_info):
    """Render mathematical foundation with proper LaTeX formatting"""

    if method_name == 'SAW':
        render_saw_formulas()
    elif method_name == 'WPM':
        render_wpm_formulas()
    elif method_name == 'TOPSIS':
        render_topsis_formulas()
    elif method_name == 'AHP':
        render_ahp_formulas()
    else:
        # Fallback to text format
        st.code(method_info['mathematical_foundation'], language='text')

def render_saw_formulas():
    """Render SAW mathematical formulas"""

    st.write("**Step 1: Normalize the decision matrix (Min-Max Normalization)**")
    st.write("For benefit criteria:")
    st.latex(r"r_{ij} = \frac{x_{ij} - \min_j(x_{ij})}{\max_j(x_{ij}) - \min_j(x_{ij})}")

    st.write("For cost criteria:")
    st.latex(r"r_{ij} = \frac{\max_j(x_{ij}) - x_{ij}}{\max_j(x_{ij}) - \min_j(x_{ij})}")

    st.write("**Step 2: Calculate weighted normalized values**")
    st.latex(r"v_{ij} = w_j \times r_{ij}")

    st.write("**Step 3: Calculate final scores**")
    st.latex(r"S_i = \sum_{j=1}^{n} v_{ij}")

    st.write("**Step 4: Rank alternatives**")
    st.write("Rank alternatives by $S_i$ (higher is better)")

def render_wpm_formulas():
    """Render WPM mathematical formulas"""

    st.write("**Step 1: Handle cost criteria (convert to benefit)**")
    st.write("For cost criteria:")
    st.latex(r"x'_{ij} = \frac{1}{x_{ij}}")

    st.write("**Step 2: Calculate weighted product scores**")
    st.latex(r"S_i = \prod_{j=1}^{n} (x'_{ij})^{w_j}")

    st.write("**Step 3: Rank alternatives**")
    st.write("Rank alternatives by $S_i$ (higher is better)")

    st.write("Where:")
    st.write("- $x'_{ij}$ = transformed value for alternative $i$ on criterion $j$")
    st.write("- $w_j$ = weight of criterion $j$")
    st.write("- $S_i$ = final score for alternative $i$")

def render_topsis_formulas():
    """Render TOPSIS mathematical formulas"""

    st.write("**Step 1: Vector normalization**")
    st.latex(r"r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{i=1}^{m} x_{ij}^2}}")

    st.write("**Step 2: Weighted normalized matrix**")
    st.latex(r"v_{ij} = w_j \times r_{ij}")

    st.write("**Step 3: Determine ideal solutions**")
    st.write("For benefit criteria:")
    st.latex(r"PIS_j = \max_i(v_{ij}), \quad NIS_j = \min_i(v_{ij})")

    st.write("For cost criteria:")
    st.latex(r"PIS_j = \min_i(v_{ij}), \quad NIS_j = \max_i(v_{ij})")

    st.write("**Step 4: Calculate separation measures**")
    st.latex(r"S_i^+ = \sqrt{\sum_{j=1}^{n} (v_{ij} - PIS_j)^2}")
    st.latex(r"S_i^- = \sqrt{\sum_{j=1}^{n} (v_{ij} - NIS_j)^2}")

    st.write("**Step 5: Calculate relative closeness**")
    st.latex(r"C_i^* = \frac{S_i^-}{S_i^+ + S_i^-}")

    st.write("**Step 6: Rank alternatives**")
    st.write("Rank alternatives by $C_i^*$ (higher is better)")

def render_ahp_formulas():
    """Render AHP mathematical formulas"""

    st.write("**Step 1: Pairwise comparison matrix**")
    st.latex(r"A = \begin{bmatrix} 1 & a_{12} & \cdots & a_{1n} \\ 1/a_{12} & 1 & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ 1/a_{1n} & 1/a_{2n} & \cdots & 1 \end{bmatrix}")

    st.write("**Step 2: Calculate priority weights (eigenvalue method)**")
    st.latex(r"A \mathbf{w} = \lambda_{max} \mathbf{w}")

    st.write(r"Where $\mathbf{w}$ is the principal eigenvector (normalized to sum to 1)")

    st.write("**Step 3: Consistency check**")
    st.latex(r"CI = \frac{\lambda_{max} - n}{n - 1}")
    st.latex(r"CR = \frac{CI}{RI}")

    st.write("**Step 4: Final synthesis**")
    st.latex(r"S_i = \sum_{j=1}^{n} w_j \times w_{ij}")

    st.write("Where:")
    st.write(r"- $CI$ = Consistency Index")
    st.write(r"- $CR$ = Consistency Ratio")
    st.write(r"- $RI$ = Random Index")
    st.write(r"- $w_j$ = weight of criterion $j$")
    st.write(r"- $w_{ij}$ = weight of alternative $i$ for criterion $j$")

    st.info("💡 **Consistency Guidelines:** CR < 0.1 indicates acceptable consistency")
