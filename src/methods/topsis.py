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
        
        # Step 4: Determine best and worst alternatives
        a_best, a_worst = self._calculate_ideal_solutions(weighted_matrix)

        # Step 5: Calculate L2-distances
        d_best, d_worst = self._calculate_separation_measures(weighted_matrix, a_best, a_worst)

        # Step 6: Calculate similarity to worst condition
        similarity_to_worst = self._calculate_relative_closeness(d_best, d_worst)
        
        # Step 7: Rank alternatives according to similarity to worst
        rankings = self.get_ranking(similarity_to_worst)

        # Store results
        self.results = {
            'scores': similarity_to_worst.tolist(),
            'rankings': rankings,
            'method': 'TOPSIS'
        }
        
        # Store intermediate steps for educational purposes
        self.intermediate_steps = {
            'original_matrix': self.decision_matrix,
            'normalized_matrix': normalized_matrix,
            'weighted_matrix': weighted_matrix,
            'a_best': a_best,
            'a_worst': a_worst,
            'd_best': d_best,
            'd_worst': d_worst,
            'similarity_to_worst': similarity_to_worst,
            'weights': self.weights,
            'criterion_types': self.criterion_types,
            # Keep old names for backward compatibility with visualizations
            'pis': a_best,
            'nis': a_worst,
            's_plus': d_best,
            's_minus': d_worst,
            'relative_closeness': similarity_to_worst
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
        """Calculate Best Alternative (Ab) and Worst Alternative (Aw)"""
        a_best = []  # Best alternative (Ab)
        a_worst = []  # Worst alternative (Aw)

        for i, (col, ctype) in enumerate(zip(weighted_matrix.columns, self.criterion_types)):
            col_values = weighted_matrix[col]

            if ctype == 'benefit':
                # For benefit criteria (J+): Ab = max, Aw = min
                a_best.append(col_values.max())
                a_worst.append(col_values.min())
            else:
                # For cost criteria (J-): Ab = min, Aw = max
                a_best.append(col_values.min())
                a_worst.append(col_values.max())

        return np.array(a_best), np.array(a_worst)
    
    def _calculate_separation_measures(self, weighted_matrix, a_best, a_worst):
        """Calculate L2-distance measures (Euclidean distances)"""
        d_best = []  # Distance to best alternative (d_ib)
        d_worst = []  # Distance to worst alternative (d_iw)

        for idx in weighted_matrix.index:
            alternative_values = weighted_matrix.loc[idx].values

            # L2-distance to best alternative
            d_ib = np.sqrt(np.sum((alternative_values - a_best) ** 2))
            d_best.append(d_ib)

            # L2-distance to worst alternative
            d_iw = np.sqrt(np.sum((alternative_values - a_worst) ** 2))
            d_worst.append(d_iw)

        return np.array(d_best), np.array(d_worst)
    
    def _calculate_relative_closeness(self, d_best, d_worst):
        """Calculate similarity to worst condition (s_iw)"""
        # s_iw = d_iw / (d_iw + d_ib)
        # Avoid division by zero
        denominator = d_worst + d_best
        similarity_to_worst = np.where(denominator != 0, d_worst / denominator, 0)
        return similarity_to_worst

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
            'title': 'Normalized Decision Matrix (R)',
            'description': 'Normalize values using vector normalization method.',
            'matrix': self.intermediate_steps['normalized_matrix'],
            'formula': 'r_ij = x_ij / sqrt(sum(x_kj^2)) for k=1 to m'
        })

        # Step 3: Weighted normalized matrix
        explanation['steps'].append({
            'step_number': 3,
            'title': 'Weighted Normalized Decision Matrix (T)',
            'description': 'Calculate weighted normalized decision matrix.',
            'matrix': self.intermediate_steps['weighted_matrix'],
            'formula': 't_ij = r_ij × w_j'
        })

        # Step 4: Best and worst alternatives
        ab_df = pd.DataFrame([self.intermediate_steps['a_best']], columns=self.criteria, index=['A_best'])
        aw_df = pd.DataFrame([self.intermediate_steps['a_worst']], columns=self.criteria, index=['A_worst'])
        ideal_solutions_df = pd.concat([ab_df, aw_df])

        explanation['steps'].append({
            'step_number': 4,
            'title': 'Best Alternative (Ab) and Worst Alternative (Aw)',
            'description': 'Determine the best and worst alternatives for each criterion.',
            'matrix': ideal_solutions_df,
            'formula': 'For benefit criteria (J+): Ab = max(t_ij), Aw = min(t_ij)\nFor cost criteria (J-): Ab = min(t_ij), Aw = max(t_ij)'
        })

        # Step 5: L2-distance measures
        separation_df = pd.DataFrame({
            'Alternative': self.alternatives,
            'Distance to Best (d_ib)': self.intermediate_steps['d_best'],
            'Distance to Worst (d_iw)': self.intermediate_steps['d_worst']
        })

        explanation['steps'].append({
            'step_number': 5,
            'title': 'L2-Distance Measures',
            'description': 'Calculate L2-norm distances from each alternative to best and worst conditions.',
            'matrix': separation_df,
            'formula': 'd_ib = sqrt(sum((t_ij - t_bj)^2))\nd_iw = sqrt(sum((t_ij - t_wj)^2))'
        })

        # Step 6: Similarity to worst condition
        results_df = pd.DataFrame({
            'Alternative': self.alternatives,
            'Similarity to Worst (s_iw)': self.intermediate_steps['similarity_to_worst'],
            'Rank': self.results['rankings']
        }).sort_values('Rank')

        explanation['steps'].append({
            'step_number': 6,
            'title': 'Similarity to Worst Condition and Final Ranking',
            'description': 'Calculate similarity to worst condition and rank alternatives (higher s_iw is better).',
            'matrix': results_df,
            'formula': 's_iw = d_iw / (d_iw + d_ib)'
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
