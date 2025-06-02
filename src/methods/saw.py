"""
Simple Additive Weighting (SAW) Method Implementation
"""

import numpy as np
import pandas as pd
from .base import MCDMMethod

class SAW(MCDMMethod):
    """
    Simple Additive Weighting (SAW) Method
    
    The SAW method calculates the weighted sum of normalized criteria values.
    It's one of the simplest and most intuitive MCDM methods.
    
    Steps:
    1. Normalize the decision matrix
    2. Multiply normalized values by criteria weights
    3. Sum weighted values for each alternative
    4. Rank alternatives by total scores
    """
    
    def calculate(self):
        """
        Calculate SAW scores and rankings
        
        Returns:
            dict: Results with scores, rankings, and intermediate steps
        """
        # Step 1: Normalize the decision matrix
        normalized_matrix = self.normalize_matrix(method='linear')
        
        # Step 2: Apply weights to normalized matrix
        weighted_matrix = normalized_matrix.copy()
        for i, col in enumerate(weighted_matrix.columns):
            weighted_matrix[col] = normalized_matrix[col] * self.weights[i]
        
        # Step 3: Calculate total scores (sum of weighted values)
        scores = weighted_matrix.sum(axis=1).values
        
        # Step 4: Calculate rankings
        rankings = self.get_ranking(scores)
        
        # Store results
        self.results = {
            'scores': scores.tolist(),
            'rankings': rankings,
            'method': 'SAW'
        }
        
        # Store intermediate steps for educational purposes
        self.intermediate_steps = {
            'original_matrix': self.decision_matrix,
            'normalized_matrix': normalized_matrix,
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
            'method_name': 'Simple Additive Weighting (SAW)',
            'description': 'SAW calculates weighted sums of normalized criteria values.',
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
        
        # Step 2: Normalization
        explanation['steps'].append({
            'step_number': 2,
            'title': 'Normalized Decision Matrix',
            'description': 'Normalize values to make criteria comparable. For benefit criteria: (x-min)/(max-min), for cost criteria: (max-x)/(max-min)',
            'matrix': self.intermediate_steps['normalized_matrix'],
            'formula': 'r_ij = (x_ij - min_j) / (max_j - min_j) for benefit criteria\nr_ij = (max_j - x_ij) / (max_j - min_j) for cost criteria'
        })
        
        # Step 3: Weighted matrix
        explanation['steps'].append({
            'step_number': 3,
            'title': 'Weighted Normalized Matrix',
            'description': 'Multiply normalized values by criteria weights.',
            'matrix': self.intermediate_steps['weighted_matrix'],
            'formula': 'v_ij = w_j × r_ij'
        })
        
        # Step 4: Final scores
        scores_df = pd.DataFrame({
            'Alternative': self.alternatives,
            'Total Score': self.results['scores'],
            'Rank': self.results['rankings']
        }).sort_values('Rank')
        
        explanation['steps'].append({
            'step_number': 4,
            'title': 'Final Scores and Rankings',
            'description': 'Sum weighted values for each alternative to get final scores.',
            'matrix': scores_df,
            'formula': 'S_i = Σ(v_ij) for j = 1 to n'
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
            'name': 'Simple Additive Weighting (SAW)',
            'other_names': ['Weighted Sum Model (WSM)', 'Scoring Method'],
            'complexity': 'Beginner',
            'description': '''
            The Simple Additive Weighting (SAW) method is one of the most straightforward 
            and widely used MCDM techniques. It calculates a weighted sum of normalized 
            criteria values for each alternative.
            ''',
            'when_to_use': [
                'When criteria can be easily compared and aggregated',
                'When you want a simple, transparent decision process',
                'When all criteria are measurable on ratio or interval scales',
                'For initial analysis or when stakeholders prefer simplicity'
            ],
            'advantages': [
                'Simple to understand and implement',
                'Transparent calculation process',
                'Allows direct comparison of alternatives',
                'Computationally efficient',
                'Widely accepted and used'
            ],
            'disadvantages': [
                'Assumes linear relationships between criteria',
                'May not handle interdependencies between criteria well',
                'Sensitive to the choice of normalization method',
                'Assumes perfect substitutability between criteria'
            ],
            'mathematical_foundation': '''
            SAW Score = Σ(w_j × r_ij) for j = 1 to n
            
            Where:
            - w_j = weight of criterion j
            - r_ij = normalized value of alternative i for criterion j
            - n = number of criteria
            ''',
            'normalization_methods': {
                'Linear (Min-Max)': 'Scales values to [0,1] range',
                'Vector': 'Normalizes using Euclidean norm'
            }
        }
