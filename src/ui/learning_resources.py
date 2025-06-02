"""
Learning resources and materials for MCDM education
"""

import streamlit as st

def render_learning_resources():
    """Render the learning resources page"""
    
    st.header("📚 MCDM Learning Resources")
    st.write("Comprehensive materials to deepen your understanding of Multi-Criteria Decision Making")
    
    # Create tabs for different types of resources
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📘 Reading Materials", 
        "🎥 Video Tutorials", 
        "💻 Online Courses", 
        "🔬 Research Papers",
        "🛠️ Tools & Software"
    ])
    
    with tab1:
        render_reading_materials()
    
    with tab2:
        render_video_tutorials()
    
    with tab3:
        render_online_courses()
    
    with tab4:
        render_research_papers()
    
    with tab5:
        render_tools_software()

def render_reading_materials():
    """Render reading materials section"""
    
    st.subheader("📘 Introductory Reading Materials")
    st.write("Essential readings to build your foundation in MCDM theory and practice")
    
    # Beginner Level
    st.write("### 🌱 Beginner Level")
    
    with st.expander("📖 Introduction to Multi-Criteria Decision Making and the Evidential Reasoning Approach", expanded=False):
        st.write("""
        **Authors:** Xu, D.L. & Yang, J.B.  
        **Description:** A comprehensive paper that introduces the fundamentals of MCDM and discusses various techniques, including the Evidential Reasoning (ER) approach.
        
        **Key Topics Covered:**
        - Basic MCDM concepts and terminology
        - Overview of major MCDM methods
        - Evidential Reasoning approach
        - Practical applications and case studies
        
        **Why Read This:** Excellent starting point for understanding the breadth of MCDM methods and their theoretical foundations.
        """)
        st.link_button("📄 Read Paper", "https://personalpages.manchester.ac.uk/staff/jian-bo.yang/JB%20Yang%20Book_Chapters/XuYang_MSM_WorkingPaperFinal.pdf")
    
    with st.expander("📖 Introduction to Multi-Criteria Decision Making: TOPSIS Method", expanded=False):
        st.write("""
        **Authors:** Lamrani Alaoui, Y. et al.  
        **Description:** This document provides an introduction to the TOPSIS method, a widely used MCDM technique, with practical examples.
        
        **Key Topics Covered:**
        - TOPSIS methodology step-by-step
        - Mathematical foundations
        - Worked examples and calculations
        - Advantages and limitations
        
        **Why Read This:** Perfect for understanding one of the most popular MCDM methods with clear examples.
        """)
        st.link_button("📄 Read Paper", "https://www.researchgate.net/profile/Youssef-Lamrani-Alaoui/publication/334726621_Introduction_to_Multi_Criteria_Decision_Making_TOPSIS_Method/links/5d3cc82ba6fdcc370a66091a/Introduction-to-Multi-Criteria-Decision-Making-TOPSIS-Method.pdf")
    
    with st.expander("📖 An Introductory Guide to Multi-Criteria Decision Analysis (MCDA)", expanded=False):
        st.write("""
        **Publisher:** UK Government Analysis Function  
        **Description:** A beginner-friendly guide that explains the principles of MCDA and its applications in decision-making processes.
        
        **Key Topics Covered:**
        - When to use MCDA
        - Setting up an MCDA problem
        - Choosing appropriate methods
        - Implementation guidelines
        - Real-world applications in government
        
        **Why Read This:** Practical guide with focus on real-world implementation and policy applications.
        """)
        st.link_button("📄 Read Guide", "https://analysisfunction.civilservice.gov.uk/policy-store/an-introductory-guide-to-mcda/")
    
    # Intermediate Level
    st.write("### 🌿 Intermediate Level")
    
    with st.expander("📖 Multi-Criteria Decision Analysis: An Operations Research Approach", expanded=False):
        st.write("""
        **Description:** A detailed paper discussing various MCDM methods from an operations research perspective.
        
        **Key Topics Covered:**
        - Mathematical foundations of MCDM
        - Comparison of different methods
        - Operations research applications
        - Advanced theoretical concepts
        
        **Why Read This:** Deeper mathematical treatment suitable for students with quantitative background.
        """)
        st.link_button("📄 Read Paper", "https://bit.csc.lsu.edu/trianta/EditedBook_CHAPTERS/EEEE1.pdf")
    
    # Recommended Books
    st.write("### 📚 Recommended Books")
    
    books = [
        {
            "title": "Multiple Criteria Decision Analysis: An Integrated Approach",
            "authors": "Belton, V. & Stewart, T.J.",
            "description": "Comprehensive textbook covering theory and practice of MCDA",
            "level": "Intermediate to Advanced"
        },
        {
            "title": "Multi-Criteria Decision Analysis: Methods and Software",
            "authors": "Ishizaka, A. & Nemery, P.",
            "description": "Practical guide with software implementations",
            "level": "Intermediate"
        },
        {
            "title": "The Analytic Hierarchy Process",
            "authors": "Saaty, T.L.",
            "description": "Definitive guide to AHP by its creator",
            "level": "Intermediate to Advanced"
        }
    ]
    
    for book in books:
        st.write(f"**{book['title']}**")
        st.write(f"*Authors:* {book['authors']}")
        st.write(f"*Level:* {book['level']}")
        st.write(f"*Description:* {book['description']}")
        st.write("---")

def render_video_tutorials():
    """Render video tutorials section"""
    
    st.subheader("🎥 Video Tutorials")
    st.write("Visual learning resources to understand MCDM concepts and applications")
    
    # Beginner Videos
    st.write("### 🎬 Beginner-Friendly Videos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Multi-Criteria Decision-Making (MCDM) Method | Simple Explanation**")
        st.write("Duration: ~15 minutes")
        st.write("This video offers a straightforward explanation of MCDM concepts using simple examples, making it ideal for beginners.")
        st.link_button("▶️ Watch Video", "https://www.youtube.com/watch?v=qDwldjPCOvk")
        
        st.write("---")
        
        st.write("**TOPSIS Method Explained with Example**")
        st.write("Duration: ~20 minutes")
        st.write("Step-by-step walkthrough of the TOPSIS method with a practical example.")
        st.write("*Search YouTube for: 'TOPSIS method tutorial'*")
    
    with col2:
        st.write("**Step-by-step Procedure of Multi-criteria Decision Making**")
        st.write("Duration: ~25 minutes")
        st.write("A tutorial that walks you through the process of conducting MCDM analysis, including practical demonstrations.")
        st.link_button("▶️ Watch Video", "https://www.youtube.com/watch?v=cETwyE9G_Vg")
        
        st.write("---")
        
        st.write("**AHP (Analytic Hierarchy Process) Tutorial**")
        st.write("Duration: ~30 minutes")
        st.write("Comprehensive introduction to AHP with pairwise comparisons and consistency checking.")
        st.write("*Search YouTube for: 'AHP tutorial pairwise comparison'*")
    
    # Method-Specific Videos
    st.write("### 🎯 Method-Specific Tutorials")
    
    methods_videos = {
        "SAW (Simple Additive Weighting)": [
            "SAW method step by step calculation",
            "Weighted sum model in decision making",
            "Multi-criteria scoring methods"
        ],
        "TOPSIS": [
            "TOPSIS method complete tutorial",
            "Ideal solution in decision making",
            "TOPSIS vs other MCDM methods"
        ],
        "AHP (Analytic Hierarchy Process)": [
            "AHP pairwise comparison tutorial",
            "Consistency ratio in AHP",
            "AHP hierarchy construction"
        ],
        "WPM (Weighted Product Method)": [
            "Weighted product model explanation",
            "WPM vs SAW comparison",
            "Geometric mean in decision making"
        ]
    }
    
    for method, searches in methods_videos.items():
        with st.expander(f"🎥 {method} Video Resources"):
            st.write(f"**Recommended YouTube searches for {method}:**")
            for search in searches:
                st.write(f"• {search}")
            st.info("💡 Tip: Look for videos with worked examples and step-by-step calculations")

def render_online_courses():
    """Render online courses section"""
    
    st.subheader("💻 Online Courses")
    st.write("Structured learning paths for comprehensive MCDM education")
    
    # Paid Courses
    st.write("### 💳 Paid Courses")
    
    with st.expander("🎓 Multi-Criteria Decision Making (MCDM) Using Matlab and Excel – Udemy", expanded=False):
        st.write("""
        **Platform:** Udemy  
        **Duration:** 8-10 hours  
        **Level:** Beginner to Intermediate  
        **Price:** ~$50-100 (often discounted)
        
        **Course Content:**
        - Introduction to MCDM concepts
        - Implementation in Excel and Matlab
        - SAW, TOPSIS, AHP methods
        - Practical case studies
        - Hands-on exercises
        
        **What You'll Learn:**
        - How to implement MCDM methods in spreadsheets
        - Programming MCDM algorithms
        - Real-world problem solving
        - Software tools for decision analysis
        
        **Best For:** Students who want practical implementation skills
        """)
        st.link_button("🔗 View Course", "https://www.udemy.com/course/multi-criteria-decision-making-mcdm-using-matlab-and-excel-h/")
    
    with st.expander("🎓 Multi-Criteria Decision Making (MCDM) – Tutorialspoint", expanded=False):
        st.write("""
        **Platform:** Tutorialspoint  
        **Duration:** Self-paced  
        **Level:** Beginner to Intermediate  
        **Price:** Subscription-based
        
        **Course Content:**
        - MCDM fundamentals
        - Excel and Matlab implementations
        - Multiple methods coverage
        - Practical examples
        
        **What You'll Learn:**
        - Software-based MCDM analysis
        - Problem formulation techniques
        - Method selection guidelines
        - Result interpretation
        
        **Best For:** Self-paced learners who prefer structured content
        """)
        st.link_button("🔗 View Course", "https://market.tutorialspoint.com/course/multi-criteria-decision-making-mcdm/index.asp")
    
    # Free Courses
    st.write("### 🆓 Free Courses")
    
    with st.expander("🎓 Multi-Criteria Decision Making and Applications – NPTEL", expanded=False):
        st.write("""
        **Platform:** NPTEL (National Programme on Technology Enhanced Learning)  
        **Duration:** 12 weeks  
        **Level:** Intermediate to Advanced  
        **Price:** Free (Certificate available for fee)
        
        **Course Content:**
        - Theoretical foundations of MCDM
        - Various MCDM methods and techniques
        - Applications across different domains
        - Case studies and assignments
        - Weekly quizzes and exams
        
        **What You'll Learn:**
        - Deep theoretical understanding
        - Mathematical foundations
        - Real-world applications
        - Research perspectives
        
        **Best For:** Academic learners seeking comprehensive theoretical knowledge
        """)
        st.link_button("🔗 View Course", "https://onlinecourses.nptel.ac.in/noc24_ge01/preview")
    
    # Course Recommendations by Level
    st.write("### 📊 Course Recommendations by Level")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**🌱 Beginner**")
        st.write("• Tutorialspoint MCDM Course")
        st.write("• YouTube tutorial series")
        st.write("• Basic Excel implementations")
        st.write("• This MCDM Learning Tool!")
    
    with col2:
        st.write("**🌿 Intermediate**")
        st.write("• Udemy MCDM Course")
        st.write("• NPTEL Course (first half)")
        st.write("• Method-specific tutorials")
        st.write("• Case study analysis")
    
    with col3:
        st.write("**🌳 Advanced**")
        st.write("• Complete NPTEL Course")
        st.write("• Research paper reviews")
        st.write("• Advanced software tools")
        st.write("• Original research projects")

def render_research_papers():
    """Render research papers section"""
    
    st.subheader("🔬 Research Papers & Academic Resources")
    st.write("Cutting-edge research and academic publications in MCDM")
    
    # Recent Research
    st.write("### 📊 Recent Research Trends")
    
    research_areas = [
        {
            "area": "Fuzzy MCDM",
            "description": "Integration of fuzzy logic with MCDM methods to handle uncertainty",
            "keywords": "fuzzy TOPSIS, fuzzy AHP, linguistic variables"
        },
        {
            "area": "Group Decision Making",
            "description": "MCDM methods for multiple decision makers and stakeholders",
            "keywords": "group AHP, consensus building, aggregation methods"
        },
        {
            "area": "Sustainability Assessment",
            "description": "Application of MCDM to environmental and sustainability problems",
            "keywords": "sustainable development, environmental MCDM, green technology"
        },
        {
            "area": "Supply Chain Management",
            "description": "MCDM applications in supplier selection and logistics",
            "keywords": "supplier selection, logistics optimization, supply chain MCDM"
        }
    ]
    
    for area in research_areas:
        with st.expander(f"🔬 {area['area']}"):
            st.write(f"**Description:** {area['description']}")
            st.write(f"**Key Search Terms:** {area['keywords']}")
            st.write("**Recommended Databases:** Google Scholar, IEEE Xplore, ScienceDirect, SpringerLink")
    
    # Classic Papers
    st.write("### 📜 Classic Papers (Must-Read)")
    
    classic_papers = [
        {
            "title": "A Scaling Method for Priorities in Hierarchical Structures",
            "author": "Saaty, T.L. (1977)",
            "significance": "Original AHP paper that introduced the method",
            "citation": "Journal of Mathematical Psychology, 15(3), 234-281"
        },
        {
            "title": "Multiple Attribute Decision Making Using TOPSIS",
            "author": "Hwang, C.L. & Yoon, K. (1981)",
            "significance": "Foundational TOPSIS methodology paper",
            "citation": "Springer-Verlag, Berlin"
        },
        {
            "title": "ELECTRE: A Comprehensive Literature Review",
            "author": "Govindan, K. & Jepsen, M.B. (2016)",
            "significance": "Comprehensive review of ELECTRE methods",
            "citation": "European Journal of Operational Research, 250(1), 1-29"
        }
    ]
    
    for paper in classic_papers:
        st.write(f"**{paper['title']}**")
        st.write(f"*Author(s):* {paper['author']}")
        st.write(f"*Significance:* {paper['significance']}")
        st.write(f"*Citation:* {paper['citation']}")
        st.write("---")
    
    # Research Databases
    st.write("### 🗄️ Academic Databases")
    
    databases = [
        {"name": "Google Scholar", "url": "https://scholar.google.com", "description": "Free access to academic papers"},
        {"name": "IEEE Xplore", "url": "https://ieeexplore.ieee.org", "description": "Engineering and technology papers"},
        {"name": "ScienceDirect", "url": "https://www.sciencedirect.com", "description": "Elsevier's database of scientific publications"},
        {"name": "SpringerLink", "url": "https://link.springer.com", "description": "Springer's collection of academic content"},
        {"name": "ResearchGate", "url": "https://www.researchgate.net", "description": "Academic social network with paper sharing"}
    ]
    
    col1, col2 = st.columns(2)
    
    for i, db in enumerate(databases):
        with col1 if i % 2 == 0 else col2:
            st.write(f"**{db['name']}**")
            st.write(db['description'])
            st.link_button(f"🔗 Visit {db['name']}", db['url'])
            st.write("---")

def render_tools_software():
    """Render tools and software section"""
    
    st.subheader("🛠️ MCDM Tools & Software")
    st.write("Software tools and platforms for implementing MCDM methods")
    
    # This Tool
    st.write("### 🎯 This Learning Tool")
    
    with st.expander("✨ MCDM Learning Tool Features", expanded=True):
        st.write("""
        **You're already using one of the best MCDM learning tools!**
        
        **Features:**
        - 4 MCDM methods (SAW, WPM, TOPSIS, AHP)
        - Interactive pairwise comparisons for AHP
        - Step-by-step calculations
        - Multiple visualization options
        - Custom problem creation
        - Example problems including energy planning
        - Educational content and explanations
        
        **Best For:** Learning, teaching, and small to medium-scale problems
        """)
    
    # Free Tools
    st.write("### 🆓 Free Tools")
    
    free_tools = [
        {
            "name": "Excel/Google Sheets",
            "description": "Spreadsheet implementation of MCDM methods",
            "pros": "Widely available, customizable, good for learning",
            "cons": "Manual setup required, limited automation",
            "best_for": "Educational purposes, simple problems"
        },
        {
            "name": "R (MCDM packages)",
            "description": "Statistical software with MCDM packages",
            "pros": "Powerful, many methods available, reproducible",
            "cons": "Programming knowledge required",
            "best_for": "Research, advanced analysis"
        },
        {
            "name": "Python (scikit-criteria)",
            "description": "Python library for MCDM methods",
            "pros": "Open source, extensible, integrates with data science tools",
            "cons": "Programming knowledge required",
            "best_for": "Research, integration with other systems"
        }
    ]
    
    for tool in free_tools:
        with st.expander(f"🔧 {tool['name']}"):
            st.write(f"**Description:** {tool['description']}")
            st.write(f"**Pros:** {tool['pros']}")
            st.write(f"**Cons:** {tool['cons']}")
            st.write(f"**Best For:** {tool['best_for']}")
    
    # Commercial Tools
    st.write("### 💼 Commercial Tools")
    
    commercial_tools = [
        {
            "name": "1000minds",
            "description": "Web-based MCDA software with PAPRIKA method",
            "url": "https://www.1000minds.com",
            "features": "User-friendly interface, pairwise comparisons, group decisions"
        },
        {
            "name": "Expert Choice",
            "description": "Professional AHP software",
            "url": "https://www.expertchoice.com",
            "features": "Advanced AHP features, sensitivity analysis, group support"
        },
        {
            "name": "PROMETHEE-GAIA",
            "description": "Software for PROMETHEE and GAIA methods",
            "url": "https://www.promethee-gaia.net",
            "features": "PROMETHEE methods, visual GAIA analysis"
        },
        {
            "name": "MACBETH",
            "description": "Measuring Attractiveness by a Categorical Based Evaluation Technique",
            "url": "https://www.m-macbeth.com",
            "features": "Qualitative approach, value measurement, sensitivity analysis"
        }
    ]
    
    for tool in commercial_tools:
        with st.expander(f"💼 {tool['name']}"):
            st.write(f"**Description:** {tool['description']}")
            st.write(f"**Key Features:** {tool['features']}")
            st.link_button(f"🔗 Visit {tool['name']}", tool['url'])
    
    # Programming Resources
    st.write("### 💻 Programming Resources")
    
    st.write("**Python Libraries:**")
    st.code("""
# Install MCDM libraries
pip install scikit-criteria
pip install pymcdm
pip install mcdm

# Example usage
from skcriteria import Data
from skcriteria.madm import simple
""")
    
    st.write("**R Packages:**")
    st.code("""
# Install MCDM packages
install.packages("MCDM")
install.packages("ahp")
install.packages("topsis")

# Example usage
library(MCDM)
result <- TOPSIS(decision_matrix, weights, criteria_types)
""")
    
    # Tool Comparison
    st.write("### 📊 Tool Comparison")
    
    comparison_data = {
        "Tool": ["This Learning Tool", "Excel", "R/Python", "1000minds", "Expert Choice"],
        "Ease of Use": ["⭐⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐"],
        "Methods Available": ["4 methods", "All (manual)", "Many", "PAPRIKA", "AHP focus"],
        "Cost": ["Free", "License fee", "Free", "Subscription", "License fee"],
        "Best For": ["Learning", "Custom analysis", "Research", "Business", "AHP projects"]
    }
    
    import pandas as pd
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    st.info("💡 **Recommendation:** Start with this learning tool, then explore programming solutions for advanced research or commercial tools for business applications.")
