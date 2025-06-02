"""
Weighted Product Method (WPM) Implementation
"""

import numpy as np
import pandas as pd
from .base import MCDMMethod

class WPM(MCDMMethod):
    """
    Weighted Product Method (WPM)
    
    The WPM method uses weighted products instead of weighted sums.
    Each criterion value is raised to the power of its weight, then
    all values for an alternative are multiplied together.
    
    Steps:
    1. Normalize the decision matrix (if needed)
    2. Raise each value to the power of its criterion weight
    3. Calculate the product of weighted values for each alternative
    4. Rank alternatives by total scores
    """
    
    def calculate(self):
        """
        Calculate WPM scores and rankings
        
        Returns:
            dict: Results with scores, rankings, and intermediate steps
        """
        # For WPM, we work with the original matrix but handle cost criteria
        matrix = self.decision_matrix.copy()
        
        # Step 1: Handle cost criteria by taking reciprocals
        processed_matrix = matrix.copy()
        for i, (col, ctype) in enumerate(zip(matrix.columns, self.criterion_types)):
            if ctype == 'cost':
                # For cost criteria, use reciprocal (1/x) to convert to benefit
                processed_matrix[col] = 1.0 / matrix[col]
        
        # Step 2: Raise each value to the power of its weight
        weighted_matrix = processed_matrix.copy()
        for i, col in enumerate(weighted_matrix.columns):
            weighted_matrix[col] = processed_matrix[col] ** self.weights[i]
        
        # Step 3: Calculate product scores
        scores = weighted_matrix.prod(axis=1).values
        
        # Step 4: Calculate rankings
        rankings = self.get_ranking(scores)
        
        # Store results
        self.results = {
            'scores': scores.tolist(),
            'rankings': rankings,
            'method': 'WPM'
        }
        
        # Store intermediate steps for educational purposes
        self.intermediate_steps = {
            'original_matrix': self.decision_matrix,
            'processed_matrix': processed_matrix,
            'weighted_matrix': weighted_matrix,
            'weights': self.weights,
            'criterion_types': self.criterion_types
        }
        
        return self.results
    
    def get_step_by_step_explanation(self):
        """
        Get detailed step-by-step explanation of the calculation
        
        Returns:
            dict: Step-by-step explanation with matrices and descriptions
        """
        if not self.results:
            self.calculate()
        
        explanation = {
            'method_name': 'Weighted Product Method (WPM)',
            'description': 'WPM calculates weighted products of criteria values.',
            'steps': []
        }
        
        # Step 1: Original matrix
        explanation['steps'].append({
            'step_number': 1,
            'title': 'Original Decision Matrix',
            'description': 'The original decision matrix with alternatives and criteria.',
            'matrix': self.intermediate_steps['original_matrix'],
            'formula': 'X = [x_ij] where i = alternatives, j = criteria'
        })
        
        # Step 2: Process cost criteria
        explanation['steps'].append({
            'step_number': 2,
            'title': 'Processed Matrix (Cost Criteria Handled)',
            'description': 'Convert cost criteria to benefit by taking reciprocals (1/x). Benefit criteria remain unchanged.',
            'matrix': self.intermediate_steps['processed_matrix'],
            'formula': 'For cost criteria: p_ij = 1/x_ij\nFor benefit criteria: p_ij = x_ij'
        })
        
        # Step 3: Weighted matrix (powers)
        explanation['steps'].append({
            'step_number': 3,
            'title': 'Weighted Matrix (Raised to Powers)',
            'description': 'Raise each processed value to the power of its criterion weight.',
            'matrix': self.intermediate_steps['weighted_matrix'],
            'formula': 'v_ij = (p_ij)^w_j'
        })
        
        # Step 4: Final scores
        scores_df = pd.DataFrame({
            'Alternative': self.alternatives,
            'Product Score': self.results['scores'],
            'Rank': self.results['rankings']
        }).sort_values('Rank')
        
        explanation['steps'].append({
            'step_number': 4,
            'title': 'Final Scores and Rankings',
            'description': 'Calculate the product of weighted values for each alternative.',
            'matrix': scores_df,
            'formula': 'S_i = Π(v_ij) for j = 1 to n'
        })
        
        return explanation
    
    @staticmethod
    def get_method_description():
        """
        Get detailed method description for educational purposes
        
        Returns:
            dict: Method description with theory and usage
        """
        return {
            'name': 'Weighted Product Method (WPM)',
            'other_names': ['Weighted Product Model'],
            'complexity': 'Beginner',
            'description': '''
            The Weighted Product Method (WPM) is similar to SAW but uses multiplication 
            instead of addition. Each criterion value is raised to the power of its 
            weight, and then all values for an alternative are multiplied together.
            ''',
            'when_to_use': [
                'When you want to avoid the rank reversal problem of SAW',
                'When criteria have multiplicative relationships',
                'When you prefer geometric aggregation over arithmetic',
                'For problems where zero values would eliminate alternatives'
            ],
            'advantages': [
                'Avoids rank reversal problem',
                'Dimensionally consistent (unit-free)',
                'Better handles multiplicative relationships',
                'More sensitive to poor performance in any criterion',
                'No need for normalization in some cases'
            ],
            'disadvantages': [
                'More complex to understand than SAW',
                'Sensitive to zero or very small values',
                'May amplify the effect of extreme values',
                'Requires careful handling of cost criteria',
                'Less intuitive interpretation of results'
            ],
            'mathematical_foundation': '''
            WPM Score = Π((x_ij)^w_j) for j = 1 to n
            
            Where:
            - x_ij = value of alternative i for criterion j
            - w_j = weight of criterion j
            - n = number of criteria
            - Π = product operator
            
            For cost criteria: x_ij is replaced by 1/x_ij
            ''',
            'key_differences_from_saw': [
                'Uses multiplication instead of addition',
                'Raises values to power of weights',
                'No normalization typically needed',
                'More sensitive to poor performance'
            ]
        }
