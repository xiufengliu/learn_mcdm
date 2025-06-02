"""
TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) Implementation
"""

import numpy as np
import pandas as pd
from .base import MCDMMethod

class TOPSIS(MCDMMethod):
    """
    TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
    
    TOPSIS is based on the concept that the chosen alternative should have the 
    shortest geometric distance from the positive ideal solution and the longest 
    geometric distance from the negative ideal solution.
    
    Steps:
    1. Normalize the decision matrix using vector normalization
    2. Calculate weighted normalized decision matrix
    3. Determine positive ideal solution (PIS) and negative ideal solution (NIS)
    4. Calculate separation measures (distances to PIS and NIS)
    5. Calculate relative closeness to ideal solution
    6. Rank alternatives based on relative closeness
    """
    
    def calculate(self):
        """
        Calculate TOPSIS scores and rankings
        
        Returns:
            dict: Results with scores, rankings, and intermediate steps
        """
        # Step 1: Normalize the decision matrix using vector normalization
        normalized_matrix = self._vector_normalize()
        
        # Step 2: Calculate weighted normalized decision matrix
        weighted_matrix = self._apply_weights(normalized_matrix)
        
        # Step 3: Determine ideal solutions
        pis, nis = self._calculate_ideal_solutions(weighted_matrix)
        
        # Step 4: Calculate separation measures
        s_plus, s_minus = self._calculate_separation_measures(weighted_matrix, pis, nis)
        
        # Step 5: Calculate relative closeness
        relative_closeness = self._calculate_relative_closeness(s_plus, s_minus)
        
        # Step 6: Calculate rankings
        rankings = self.get_ranking(relative_closeness)
        
        # Store results
        self.results = {
            'scores': relative_closeness.tolist(),
            'rankings': rankings,
            'method': 'TOPSIS'
        }
        
        # Store intermediate steps for educational purposes
        self.intermediate_steps = {
            'original_matrix': self.decision_matrix,
            'normalized_matrix': normalized_matrix,
            'weighted_matrix': weighted_matrix,
            'pis': pis,
            'nis': nis,
            's_plus': s_plus,
            's_minus': s_minus,
            'relative_closeness': relative_closeness,
            'weights': self.weights,
            'criterion_types': self.criterion_types
        }
        
        return self.results
    
    def _vector_normalize(self):
        """Vector normalization of decision matrix"""
        matrix = self.decision_matrix.copy().astype(float)
        normalized = matrix.copy()
        
        for col in matrix.columns:
            # Calculate the norm (square root of sum of squares)
            norm = np.sqrt(np.sum(matrix[col] ** 2))
            if norm != 0:
                normalized[col] = matrix[col] / norm
            else:
                normalized[col] = 0
        
        return normalized
    
    def _apply_weights(self, normalized_matrix):
        """Apply weights to normalized matrix"""
        weighted = normalized_matrix.copy()
        for i, col in enumerate(weighted.columns):
            weighted[col] = normalized_matrix[col] * self.weights[i]
        return weighted
    
    def _calculate_ideal_solutions(self, weighted_matrix):
        """Calculate Positive Ideal Solution (PIS) and Negative Ideal Solution (NIS)"""
        pis = []
        nis = []
        
        for i, (col, ctype) in enumerate(zip(weighted_matrix.columns, self.criterion_types)):
            col_values = weighted_matrix[col]
            
            if ctype == 'benefit':
                # For benefit criteria: PIS = max, NIS = min
                pis.append(col_values.max())
                nis.append(col_values.min())
            else:
                # For cost criteria: PIS = min, NIS = max
                pis.append(col_values.min())
                nis.append(col_values.max())
        
        return np.array(pis), np.array(nis)
    
    def _calculate_separation_measures(self, weighted_matrix, pis, nis):
        """Calculate separation measures (Euclidean distances)"""
        s_plus = []  # Distance to PIS
        s_minus = []  # Distance to NIS
        
        for idx in weighted_matrix.index:
            alternative_values = weighted_matrix.loc[idx].values
            
            # Distance to PIS
            d_plus = np.sqrt(np.sum((alternative_values - pis) ** 2))
            s_plus.append(d_plus)
            
            # Distance to NIS
            d_minus = np.sqrt(np.sum((alternative_values - nis) ** 2))
            s_minus.append(d_minus)
        
        return np.array(s_plus), np.array(s_minus)
    
    def _calculate_relative_closeness(self, s_plus, s_minus):
        """Calculate relative closeness to ideal solution"""
        # Avoid division by zero
        denominator = s_plus + s_minus
        relative_closeness = np.where(denominator != 0, s_minus / denominator, 0)
        return relative_closeness

    def get_step_by_step_explanation(self):
        """
        Get detailed step-by-step explanation of the calculation

        Returns:
            dict: Step-by-step explanation with matrices and descriptions
        """
        if not self.results:
            self.calculate()

        explanation = {
            'method_name': 'TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)',
            'description': 'TOPSIS selects alternatives that are closest to the ideal solution and farthest from the negative ideal solution.',
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

        # Step 2: Normalized matrix
        explanation['steps'].append({
            'step_number': 2,
            'title': 'Normalized Decision Matrix',
            'description': 'Normalize values using vector normalization (divide by the norm of each column).',
            'matrix': self.intermediate_steps['normalized_matrix'],
            'formula': 'r_ij = x_ij / sqrt(sum(x_ij^2)) for all i'
        })

        # Step 3: Weighted normalized matrix
        explanation['steps'].append({
            'step_number': 3,
            'title': 'Weighted Normalized Matrix',
            'description': 'Multiply normalized values by criteria weights.',
            'matrix': self.intermediate_steps['weighted_matrix'],
            'formula': 'v_ij = w_j × r_ij'
        })

        # Step 4: Ideal solutions
        pis_df = pd.DataFrame([self.intermediate_steps['pis']], columns=self.criteria, index=['PIS'])
        nis_df = pd.DataFrame([self.intermediate_steps['nis']], columns=self.criteria, index=['NIS'])
        ideal_solutions_df = pd.concat([pis_df, nis_df])

        explanation['steps'].append({
            'step_number': 4,
            'title': 'Positive and Negative Ideal Solutions',
            'description': 'Determine the ideal (PIS) and negative ideal (NIS) solutions for each criterion.',
            'matrix': ideal_solutions_df,
            'formula': 'For benefit criteria: PIS = max(v_ij), NIS = min(v_ij)\nFor cost criteria: PIS = min(v_ij), NIS = max(v_ij)'
        })

        # Step 5: Separation measures
        separation_df = pd.DataFrame({
            'Alternative': self.alternatives,
            'Distance to PIS (S+)': self.intermediate_steps['s_plus'],
            'Distance to NIS (S-)': self.intermediate_steps['s_minus']
        })

        explanation['steps'].append({
            'step_number': 5,
            'title': 'Separation Measures',
            'description': 'Calculate Euclidean distances from each alternative to PIS and NIS.',
            'matrix': separation_df,
            'formula': 'S_i+ = sqrt(sum((v_ij - PIS_j)^2))\nS_i- = sqrt(sum((v_ij - NIS_j)^2))'
        })

        # Step 6: Relative closeness
        results_df = pd.DataFrame({
            'Alternative': self.alternatives,
            'Relative Closeness (C*)': self.intermediate_steps['relative_closeness'],
            'Rank': self.results['rankings']
        }).sort_values('Rank')

        explanation['steps'].append({
            'step_number': 6,
            'title': 'Relative Closeness and Final Ranking',
            'description': 'Calculate relative closeness to the ideal solution and determine rankings.',
            'matrix': results_df,
            'formula': 'C*_i = S_i- / (S_i+ + S_i-)'
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
            'name': 'TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)',
            'other_names': ['Ideal Point Method'],
            'complexity': 'Intermediate',
            'description': '''
            TOPSIS is based on the concept that the chosen alternative should have the
            shortest geometric distance from the positive ideal solution (PIS) and the
            longest geometric distance from the negative ideal solution (NIS).
            ''',
            'when_to_use': [
                'When you want to consider both the best and worst possible scenarios',
                'When you need a method that handles both benefit and cost criteria well',
                'When you want a method with strong theoretical foundation',
                'For problems where the distance to ideal solutions is meaningful'
            ],
            'advantages': [
                'Considers both positive and negative ideal solutions',
                'Simple, rational, and comprehensible concept',
                'Easy computation process',
                'Allows for trade-offs between criteria',
                'Provides a cardinal ranking of alternatives'
            ],
            'disadvantages': [
                'Sensitive to normalization method',
                'Susceptible to rank reversal when alternatives are added or removed',
                'Euclidean distance does not account for correlation between criteria',
                'Requires careful definition of positive and negative ideals'
            ],
            'mathematical_foundation': '''
            1. Normalize the decision matrix using vector normalization:
               r_ij = x_ij / sqrt(sum(x_ij^2)) for all i

            2. Calculate weighted normalized values:
               v_ij = w_j × r_ij

            3. Determine ideal solutions:
               For benefit criteria: PIS = max(v_ij), NIS = min(v_ij)
               For cost criteria: PIS = min(v_ij), NIS = max(v_ij)

            4. Calculate separation measures:
               S_i+ = sqrt(sum((v_ij - PIS_j)^2))
               S_i- = sqrt(sum((v_ij - NIS_j)^2))

            5. Calculate relative closeness:
               C*_i = S_i- / (S_i+ + S_i-)

            6. Rank alternatives by C*_i (higher is better)
            ''',
            'key_differences_from_saw': [
                'Considers distance to both ideal and anti-ideal points',
                'Uses vector normalization instead of linear',
                'More robust to certain types of rank reversal',
                'Better handles problems with extreme values'
            ]
        }
