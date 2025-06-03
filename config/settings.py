"""
Configuration settings for the MCDM Learning Tool
"""

# Application configuration
APP_CONFIG = {
    "page_title": "MCDM Learning Tool",
    "page_icon": "🎯",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# MCDM Methods configuration
MCDM_METHODS = {
    "SAW": {
        "name": "Simple Additive Weighting",
        "description": "A straightforward method that calculates weighted sums of normalized criteria values.",
        "complexity": "Beginner",
        "phase": 1,
        "enabled": True
    },
    "WPM": {
        "name": "Weighted Product Method",
        "description": "Uses weighted products instead of weighted sums for aggregation.",
        "complexity": "Beginner",
        "phase": 1,
        "enabled": True
    },
    "TOPSIS": {
        "name": "TOPSIS",
        "description": "Technique for Order Preference by Similarity to Ideal Solution.",
        "complexity": "Intermediate",
        "phase": 2,
        "enabled": True
    },
    "AHP": {
        "name": "Analytic Hierarchy Process",
        "description": "Uses pairwise comparisons to derive criteria weights and rankings.",
        "complexity": "Advanced",
        "phase": 3,
        "enabled": True  # Now enabled in Phase 3
    }
}

# Default example problems
EXAMPLE_PROBLEMS = {
    "Car Selection": {
        "description": "Choosing the best car based on multiple criteria",
        "alternatives": ["Toyota Camry", "Honda Accord", "BMW 3 Series", "Audi A4"],
        "criteria": ["Price", "Fuel Economy", "Safety", "Performance", "Comfort"],
        "criterion_types": ["cost", "benefit", "benefit", "benefit", "benefit"],
        "decision_matrix": [
            [25000, 32, 5, 7, 8],  # Toyota Camry
            [27000, 30, 5, 6, 7],  # Honda Accord
            [35000, 25, 4, 9, 9],  # BMW 3 Series
            [38000, 27, 4, 8, 9]   # Audi A4
        ],
        "weights": [0.3, 0.2, 0.25, 0.15, 0.1]
    },
    "Supplier Selection": {
        "description": "Selecting the best supplier for a manufacturing company",
        "alternatives": ["Supplier A", "Supplier B", "Supplier C"],
        "criteria": ["Cost", "Quality", "Delivery Time", "Reliability"],
        "criterion_types": ["cost", "benefit", "cost", "benefit"],
        "decision_matrix": [
            [100, 8, 5, 7],   # Supplier A
            [120, 9, 3, 8],   # Supplier B
            [90, 7, 4, 6]     # Supplier C
        ],
        "weights": [0.4, 0.3, 0.2, 0.1]
    },
    "University Selection": {
        "description": "Choosing the best university for higher education",
        "alternatives": ["University A", "University B", "University C", "University D"],
        "criteria": ["Tuition Fee", "Ranking", "Location", "Research", "Campus Life"],
        "criterion_types": ["cost", "benefit", "benefit", "benefit", "benefit"],
        "decision_matrix": [
            [50000, 85, 7, 8, 6],  # University A
            [45000, 90, 8, 9, 8],  # University B
            [40000, 75, 6, 7, 7],  # University C
            [55000, 95, 9, 9, 9]   # University D
        ],
        "weights": [0.25, 0.3, 0.15, 0.2, 0.1]
    },
    "Software Selection": {
        "description": "Selecting project management software using AHP",
        "alternatives": ["Software A", "Software B", "Software C"],
        "criteria": ["Cost", "Usability", "Features", "Support"],
        "criterion_types": ["cost", "benefit", "benefit", "benefit"],
        "decision_matrix": None,  # AHP doesn't use decision matrix
        "weights": None,  # AHP derives weights from pairwise comparisons
        "ahp_example": True
    },
    "Renewable Energy Selection": {
        "description": "Selecting the best renewable energy technology for a region",
        "alternatives": ["Solar PV", "Wind Turbines", "Hydroelectric", "Biomass", "Geothermal"],
        "criteria": ["Initial Cost", "Energy Output", "Environmental Impact", "Reliability", "Maintenance Cost", "Land Use"],
        "criterion_types": ["cost", "benefit", "benefit", "benefit", "cost", "cost"],
        "decision_matrix": [
            [850000, 2500, 9, 8, 25000, 15],    # Solar PV ($/MW, MWh/year, score 1-10, score 1-10, $/year, acres/MW)
            [1200000, 3200, 8, 7, 35000, 25],   # Wind Turbines
            [2500000, 4000, 9, 9, 15000, 5],    # Hydroelectric
            [600000, 1800, 7, 6, 45000, 30],    # Biomass
            [3000000, 3500, 10, 9, 20000, 8]    # Geothermal
        ],
        "weights": [0.25, 0.30, 0.20, 0.15, 0.05, 0.05]
    }
}

# UI Configuration
UI_CONFIG = {
    "max_alternatives": 10,
    "max_criteria": 8,
    "default_alternatives": 3,
    "default_criteria": 3,
    "number_format": "%.4f"
}

# Visualization settings
VIZ_CONFIG = {
    "color_palette": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                     "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
    "chart_height": 400,
    "chart_width": 600
}
