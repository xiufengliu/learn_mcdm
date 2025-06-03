# LaTeX Mathematical Formulas in MCDM Learning Tool

This document demonstrates how mathematical formulas are now rendered in the MCDM Learning Tool using LaTeX formatting.

## 🧮 Enhanced Mathematical Display

### Before (Plain Text):
```
r_ij = x_ij / sqrt(sum(x_ij^2)) for all i
S_i+ = sqrt(sum((v_ij - PIS_j)^2))
C*_i = S_i- / (S_i+ + S_i-)
```

### After (LaTeX Rendered):
The formulas are now displayed as properly formatted mathematical expressions using Streamlit's `st.latex()` function.

## 📊 Method-Specific Formula Rendering

### 1. SAW (Simple Additive Weighting)

**Normalization:**
- Benefit criteria: `r_{ij} = \frac{x_{ij}}{\max_i(x_{ij})}`
- Cost criteria: `r_{ij} = \frac{\min_i(x_{ij})}{x_{ij}}`

**Final Score:**
- `S_i = \sum_{j=1}^{n} w_j \times r_{ij}`

### 2. WPM (Weighted Product Method)

**Cost Conversion:**
- `x'_{ij} = \frac{1}{x_{ij}}`

**Final Score:**
- `S_i = \prod_{j=1}^{n} (x'_{ij})^{w_j}`

### 3. TOPSIS

**Vector Normalization:**
- `r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{i=1}^{m} x_{ij}^2}}`

**Weighted Normalized Matrix:**
- `v_{ij} = w_j \times r_{ij}`

**Ideal Solutions:**
- Benefit: `PIS_j = \max_i(v_{ij}), \quad NIS_j = \min_i(v_{ij})`
- Cost: `PIS_j = \min_i(v_{ij}), \quad NIS_j = \max_i(v_{ij})`

**Separation Measures:**
- `S_i^+ = \sqrt{\sum_{j=1}^{n} (v_{ij} - PIS_j)^2}`
- `S_i^- = \sqrt{\sum_{j=1}^{n} (v_{ij} - NIS_j)^2}`

**Relative Closeness:**
- `C_i^* = \frac{S_i^-}{S_i^+ + S_i^-}`

### 4. AHP (Analytic Hierarchy Process)

**Pairwise Comparison Matrix:**
```
A = [1      a_12    ...  a_1n  ]
    [1/a_12  1      ...  a_2n  ]
    [⋮       ⋮      ⋱    ⋮     ]
    [1/a_1n  1/a_2n ...   1    ]
```

**Eigenvalue Method:**
- `A \mathbf{w} = \lambda_{max} \mathbf{w}`

**Consistency Check:**
- `CI = \frac{\lambda_{max} - n}{n - 1}`
- `CR = \frac{CI}{RI}`

**Final Synthesis:**
- `S_i = \sum_{j=1}^{n} w_j \times w_{ij}`

## 🎯 Where LaTeX Formulas Appear

### 1. Method Guide Tab
- **Mathematical Foundation section** for each method
- Properly formatted formulas with explanations
- Symbol definitions and notation guide

### 2. Step-by-Step Calculations
- **Enhanced formula display** in calculation steps
- Context-aware LaTeX rendering based on the method
- Fallback to code format for complex expressions

### 3. Learning Resources
- **Mathematical Notation Guide** with common symbols
- **Decision matrix notation** and standard MCDM symbols
- **Interactive examples** with proper mathematical formatting

## 🔧 Implementation Details

### LaTeX Rendering Functions:
- `render_mathematical_foundation()` - Main formula renderer
- `render_saw_formulas()` - SAW-specific formulas
- `render_wpm_formulas()` - WPM-specific formulas
- `render_topsis_formulas()` - TOPSIS-specific formulas
- `render_ahp_formulas()` - AHP-specific formulas
- `render_step_formula()` - Step-by-step formula renderer

### Features:
- **Method-specific rendering** - Different formulas for different methods
- **Context-aware display** - Formulas match the current calculation step
- **Fallback support** - Plain text display if LaTeX fails
- **Symbol consistency** - Standardized mathematical notation

## 📚 Educational Benefits

### Enhanced Learning Experience:
1. **Professional Appearance** - Mathematical formulas look like textbook quality
2. **Better Comprehension** - Proper notation aids understanding
3. **Standard Notation** - Students learn correct mathematical symbols
4. **Visual Clarity** - Complex formulas are easier to read

### Improved Accessibility:
- **Screen Reader Compatible** - LaTeX can be read by assistive technology
- **Copy-Paste Friendly** - Formulas can be copied for assignments
- **Print Quality** - Formulas look good in printed materials
- **Mobile Responsive** - Formulas scale properly on different devices

## 🎓 Usage Examples

When students select TOPSIS method and view the Mathematical Foundation:

1. **Vector Normalization** formula appears as a properly formatted fraction
2. **Distance calculations** show with square root symbols and summations
3. **Relative closeness** displays as a clear fraction with superscripts

When viewing step-by-step calculations:

1. **Each step** shows the relevant formula in LaTeX format
2. **Calculations** are easier to follow with proper mathematical notation
3. **Results** are clearly linked to the formulas used

This enhancement significantly improves the educational value and professional appearance of the MCDM Learning Tool!
