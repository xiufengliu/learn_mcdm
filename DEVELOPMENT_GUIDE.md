# MCDM Learning Tool - Development Guide

## Project Overview

This is a comprehensive Streamlit application designed to help students learn Multi-Criteria Decision Making (MCDM) methods through interactive examples and step-by-step calculations.

## Project Structure

```
mcdm/
├── app.py                 # Main Streamlit application entry point
├── run.py                 # Simple script to run the application
├── setup.py               # Setup script for easy installation
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── DEVELOPMENT_GUIDE.md  # This file
├── config/
│   └── settings.py       # Application configuration and constants
├── data/
│   └── examples/         # Example CSV files for testing
├── src/
│   ├── __init__.py
│   ├── methods/          # MCDM method implementations
│   │   ├── __init__.py
│   │   ├── base.py       # Abstract base class for all methods
│   │   ├── saw.py        # Simple Additive Weighting
│   │   └── wpm.py        # Weighted Product Method
│   ├── ui/              # User interface components
│   │   ├── __init__.py
│   │   ├── sidebar.py    # Sidebar components
│   │   └── main_content.py # Main content area
│   ├── utils/           # Utility functions
│   │   ├── __init__.py
│   │   ├── session_state.py # Session state management
│   │   └── validation.py    # Input validation
│   └── visualization/   # Plotting and visualization
│       ├── __init__.py
│       └── charts.py    # Chart creation functions
└── tests/               # Unit tests
    ├── __init__.py
    └── test_methods.py  # Tests for MCDM methods
```

## Phase-wise Development Plan

### Phase 1 (MVP) - ✅ COMPLETED
- [x] Basic project structure
- [x] Simple Additive Weighting (SAW) method
- [x] Weighted Product Method (WPM)
- [x] Basic UI with data input and results display
- [x] Session state management
- [x] Input validation
- [x] Example problems
- [x] Method explanations

### Phase 2 - ✅ COMPLETED
- [x] TOPSIS method implementation
- [x] Enhanced visualizations with Plotly
- [x] Method-specific analysis charts
- [x] Criteria impact analysis
- [x] Improved UI/UX design with tabbed visualizations
- [x] Advanced educational content
- [x] Better error handling

### Phase 3 - ✅ COMPLETED
- [x] Analytic Hierarchy Process (AHP)
- [x] Pairwise comparison interface
- [x] Consistency ratio calculation
- [x] Saaty scale implementation
- [x] AHP-specific UI components
- [x] Advanced educational features
- [x] Step-by-step AHP tutorials

### Phase 4 (Advanced)
- [ ] ELECTRE method
- [ ] PROMETHEE method
- [ ] Sensitivity analysis
- [ ] What-if scenarios
- [ ] Comparison between methods

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Quick Setup:**
   ```bash
   python setup.py
   ```

2. **Manual Setup:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Using the run script:**
   ```bash
   python run.py
   ```

2. **Direct Streamlit command:**
   ```bash
   streamlit run app.py
   ```

3. **For development:**
   ```bash
   streamlit run app.py --server.runOnSave true
   ```

## Development Guidelines

### Adding New MCDM Methods

1. **Create method class** in `src/methods/`:
   ```python
   from .base import MCDMMethod
   
   class NewMethod(MCDMMethod):
       def calculate(self):
           # Implementation here
           pass
   ```

2. **Update configuration** in `config/settings.py`:
   ```python
   MCDM_METHODS["NEW"] = {
       "name": "New Method",
       "description": "Description here",
       "complexity": "Intermediate",
       "phase": 2,
       "enabled": True
   }
   ```

3. **Add to UI** in `src/ui/main_content.py`:
   ```python
   method_classes = {
       'SAW': SAW,
       'WPM': WPM,
       'NEW': NewMethod,  # Add here
   }
   ```

4. **Write tests** in `tests/test_methods.py`

### UI Component Guidelines

- **Sidebar**: Method selection, problem setup, configuration
- **Main Content**: Tabbed interface with data input, results, visualizations, learning
- **Validation**: Always validate inputs before processing
- **Session State**: Use session state for persistence across interactions

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Keep functions focused and modular
- Use meaningful variable names

## Testing

### Running Tests
```bash
python -m pytest tests/
```

### Test Coverage
```bash
python -m pytest tests/ --cov=src
```

### Manual Testing Checklist
- [ ] Load example problems
- [ ] Input custom data
- [ ] Calculate results for each method
- [ ] Verify step-by-step calculations
- [ ] Test input validation
- [ ] Check visualizations
- [ ] Test edge cases (equal weights, single criterion, etc.)

## Configuration

### Application Settings
Edit `config/settings.py` to modify:
- Available MCDM methods
- Example problems
- UI limits (max alternatives, criteria)
- Visualization settings

### Example Problems
Add new examples in `config/settings.py`:
```python
EXAMPLE_PROBLEMS["New Example"] = {
    "description": "Description",
    "alternatives": ["A1", "A2"],
    "criteria": ["C1", "C2"],
    "criterion_types": ["benefit", "cost"],
    "decision_matrix": [[1, 2], [3, 4]],
    "weights": [0.6, 0.4]
}
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the project root directory
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **Streamlit Issues**: Try `streamlit cache clear`
4. **Visualization Problems**: Check if plotly is installed correctly

### Debug Mode
Set `debug=True` in Streamlit config or use:
```bash
streamlit run app.py --logger.level=debug
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Educational Features

### Method Explanations
Each method includes:
- When to use it
- Advantages and disadvantages
- Mathematical foundation
- Step-by-step calculation

### Interactive Learning
- Real-time calculation updates
- Visual feedback
- Example problems with explanations
- Comparison between methods

## Future Enhancements

### Technical Improvements
- Database integration for storing problems
- User authentication and saved sessions
- REST API for external integration
- Mobile-responsive design

### Educational Features
- Interactive tutorials
- Gamification elements
- Progress tracking
- Collaborative features

### Advanced Analytics
- Sensitivity analysis
- Robustness testing
- Method comparison tools
- Statistical analysis of results

## License

MIT License - see LICENSE file for details.
