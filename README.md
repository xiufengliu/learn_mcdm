# MCDM Learning Tool

An interactive Streamlit application for learning Multi-Criteria Decision Making (MCDM) methods.

## Overview

This tool helps students understand and apply various MCDM methods through an interactive web interface. It provides step-by-step calculations, visualizations, and educational content to enhance learning.

## Features

### Phase 1 (MVP) - ✅ COMPLETED
- ✅ Simple Additive Weighting (SAW)
- ✅ Weighted Product Method (WPM)
- ✅ Basic data input and results display
- ✅ Method explanations

### Phase 2 - ✅ COMPLETED
- ✅ TOPSIS method with distance analysis
- ✅ Enhanced UI/UX with advanced visualizations
- ✅ Method-specific analysis charts
- ✅ Criteria impact analysis
- ✅ Improved educational content

### Phase 3 - ✅ COMPLETED
- ✅ Analytic Hierarchy Process (AHP)
- ✅ Pairwise comparison interface
- ✅ Consistency ratio analysis
- ✅ Step-by-step AHP calculations
- ✅ Saaty scale implementation

### Enhanced Features - ✅ COMPLETED
- ✅ **Custom Problem Management**: Create, save, load, and manage your own decision problems
- ✅ **Import/Export**: Share problems via JSON files
- ✅ **Renewable Energy Example**: Real-world sustainability decision scenario
- ✅ **6 Built-in Examples**: Car selection, supplier selection, university selection, renewable energy, software selection (AHP), and more
- ✅ **Problem Validation**: Comprehensive input validation and error handling

### Phase 4 (Future)
- Additional methods (ELECTRE, PROMETHEE)
- Sensitivity analysis
- Method comparison tools
- Advanced features

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
mcdm/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── config/
│   └── settings.py       # Application configuration
├── data/
│   └── examples/         # Example datasets
├── src/
│   ├── __init__.py
│   ├── methods/          # MCDM method implementations
│   ├── ui/              # UI components
│   ├── utils/           # Utility functions
│   └── visualization/   # Plotting and visualization
└── tests/               # Unit tests
```

## Usage

1. Select an MCDM method from the sidebar
2. Define your alternatives and criteria
3. Input the decision matrix values
4. Set criteria weights
5. View results and rankings
6. Explore visualizations and explanations

## Educational Features

- Method explanations and theory
- Step-by-step calculations
- Interactive examples
- Sensitivity analysis
- Visual comparisons

## Contributing

This is an educational tool. Contributions are welcome to improve the learning experience.

## License

MIT License
