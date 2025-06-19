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
        Calculate WPM scores and rankings using the ratio-based approach

        Returns:
            dict: Results with scores, rankings, and intermediate steps
        """
        matrix = self.decision_matrix.copy()
        n_alternatives = len(self.alternatives)

        # Step 1: Handle cost criteria by taking reciprocals (convert to benefit)
        processed_matrix = matrix.copy().astype(float)
        for i, (col, ctype) in enumerate(zip(matrix.columns, self.criterion_types)):
            col_values = matrix[col].astype(float)

            # Ensure all values are positive to avoid invalid powers or divisions
            if np.any(col_values <= 0):
                positives = col_values[col_values > 0]
                if len(positives) > 0:
                    offset = positives.min() / 1000.0
                else:
                    offset = abs(col_values.min()) + 1e-6
                col_values = col_values + offset

            if ctype == 'cost':
                processed_matrix[col] = 1.0 / col_values
            else:
                processed_matrix[col] = col_values

        # Step 2: Calculate pairwise comparison ratios
        # Create a matrix to store P(Ak/Al) values
        comparison_matrix = np.zeros((n_alternatives, n_alternatives))

        for k in range(n_alternatives):
            for l in range(n_alternatives):
                if k == l:
                    comparison_matrix[k, l] = 1.0  # P(Ak/Ak) = 1
                else:
                    # Calculate P(Ak/Al) = ∏(akj/alj)^wj
                    ratio_product = 1.0
                    for j, col in enumerate(processed_matrix.columns):
                        ratio = processed_matrix.iloc[k, j] / processed_matrix.iloc[l, j]
                        ratio_product *= (ratio ** self.weights[j])
                    comparison_matrix[k, l] = ratio_product

        # Step 3: Calculate final scores
        # An alternative's score is how many times it's better than or equal to others
        scores = np.sum(comparison_matrix >= 1.0, axis=1)

        # Alternative approach: Use geometric mean of ratios as score
        # This gives more nuanced scoring
        geometric_scores = np.zeros(n_alternatives)
        for k in range(n_alternatives):
            # Calculate geometric mean of all ratios for alternative k
            ratios = comparison_matrix[k, :]
            geometric_scores[k] = np.power(np.prod(ratios), 1.0/n_alternatives)

        # Use geometric scores for final ranking (more discriminating)
        final_scores = geometric_scores

        # Step 4: Calculate rankings
        rankings = self.get_ranking(final_scores)

        # Store results
        self.results = {
            'scores': final_scores.tolist(),
            'rankings': rankings,
            'method': 'WPM'
        }

        # Store intermediate steps for educational purposes
        self.intermediate_steps = {
            'original_matrix': self.decision_matrix,
            'processed_matrix': processed_matrix,
            'comparison_matrix': comparison_matrix,
            'pairwise_scores': scores,
            'geometric_scores': geometric_scores,
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
            'description': 'WPM uses pairwise comparisons with ratio-based calculations to rank alternatives.',
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
            'title': 'Processed Matrix (Cost Criteria Converted)',
            'description': 'Convert cost criteria to benefit by taking reciprocals (1/x). Benefit criteria remain unchanged.',
            'matrix': self.intermediate_steps['processed_matrix'],
            'formula': 'For cost criteria: p_ij = 1/x_ij\nFor benefit criteria: p_ij = x_ij'
        })

        # Step 3: Pairwise comparison matrix
        comparison_df = pd.DataFrame(
            self.intermediate_steps['comparison_matrix'],
            index=[f"A{i+1}" for i in range(len(self.alternatives))],
            columns=[f"A{i+1}" for i in range(len(self.alternatives))]
        )

        explanation['steps'].append({
            'step_number': 3,
            'title': 'Pairwise Comparison Matrix P(Ak/Al)',
            'description': 'Calculate ratios between alternatives using the formula P(Ak/Al) = ∏(akj/alj)^wj. Values ≥1 indicate Ak is better than Al.',
            'matrix': comparison_df,
            'formula': 'P(Ak/Al) = ∏(akj/alj)^wj for j = 1 to n'
        })

        # Step 4: Final scores
        scores_df = pd.DataFrame({
            'Alternative': self.alternatives,
            'Geometric Score': self.results['scores'],
            'Rank': self.results['rankings']
        }).sort_values('Rank')

        explanation['steps'].append({
            'step_number': 4,
            'title': 'Final Scores and Rankings',
            'description': 'Calculate geometric mean of pairwise comparison ratios for each alternative.',
            'matrix': scores_df,
            'formula': 'Score_k = (∏P(Ak/Al))^(1/m) for l = 1 to m'
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
            The Weighted Product Method (WPM) uses pairwise comparisons between alternatives.
            For each pair of alternatives, it calculates ratios of their criterion values,
            raises each ratio to the power of the criterion weight, and multiplies them together.
            This approach avoids the "adding apples and oranges" problem of SAW.
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
            Primary WPM Formula (Pairwise Comparison):
            P(Ak/Al) = Π((akj/alj)^wj) for j = 1 to n

            Where:
            - P(Ak/Al) = preference of alternative k over alternative l
            - akj, alj = values of alternatives k and l for criterion j
            - wj = weight of criterion j
            - n = number of criteria
            - Π = product operator

            If P(Ak/Al) ≥ 1, then alternative k is preferred over alternative l.

            For cost criteria: values are replaced by their reciprocals (1/x).
            Final ranking uses geometric mean of all pairwise comparisons.
            ''',
            'key_differences_from_saw': [
                'Uses multiplication instead of addition',
                'Raises values to power of weights',
                'No normalization typically needed',
                'More sensitive to poor performance'
            ]
        }
