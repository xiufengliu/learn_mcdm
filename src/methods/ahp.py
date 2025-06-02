"""
Analytic Hierarchy Process (AHP) Implementation
"""

import numpy as np
import pandas as pd
from .base import MCDMMethod

class AHP(MCDMMethod):
    """
    Analytic Hierarchy Process (AHP)
    
    AHP is a structured technique for organizing and analyzing complex decisions.
    It uses pairwise comparisons to derive priority weights for criteria and alternatives.
    
    Steps:
    1. Create pairwise comparison matrices for criteria
    2. Calculate priority weights from comparison matrices
    3. Check consistency of comparisons
    4. Create pairwise comparison matrices for alternatives (for each criterion)
    5. Calculate final scores by combining weights
    6. Rank alternatives
    """
    
    def __init__(self, decision_matrix=None, weights=None, criterion_types=None, 
                 alternatives=None, criteria=None, pairwise_matrices=None):
        """
        Initialize AHP method
        
        Args:
            decision_matrix: Not used in AHP (can be None)
            weights: Not used in AHP (derived from pairwise comparisons)
            criterion_types: Not used in AHP
            alternatives: List of alternative names
            criteria: List of criteria names
            pairwise_matrices: Dictionary containing pairwise comparison matrices
        """
        # For AHP, we don't use the traditional decision matrix approach
        if decision_matrix is not None:
            super().__init__(decision_matrix, weights or [1], criterion_types or ['benefit'], alternatives, criteria)
        else:
            self.alternatives = alternatives or []
            self.criteria = criteria or []
            self.weights = None  # Will be calculated from pairwise comparisons
            self.criterion_types = None  # Not used in AHP
            self.decision_matrix = None
        
        self.pairwise_matrices = pairwise_matrices or {}
        self.results = {}
        self.intermediate_steps = {}
        
        # AHP-specific attributes
        self.criteria_matrix = None
        self.criteria_weights = None
        self.alternative_matrices = {}
        self.alternative_weights = {}
        self.consistency_ratios = {}
    
    def set_criteria_comparison_matrix(self, matrix):
        """Set the pairwise comparison matrix for criteria"""
        self.criteria_matrix = np.array(matrix)
    
    def set_alternative_comparison_matrix(self, criterion, matrix):
        """Set the pairwise comparison matrix for alternatives under a specific criterion"""
        self.alternative_matrices[criterion] = np.array(matrix)
    
    def calculate_priority_weights(self, matrix):
        """
        Calculate priority weights from pairwise comparison matrix using eigenvalue method
        
        Args:
            matrix (np.array): Pairwise comparison matrix
            
        Returns:
            tuple: (weights, consistency_ratio)
        """
        n = matrix.shape[0]
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        
        # Find the principal eigenvalue (largest real eigenvalue)
        max_eigenvalue_idx = np.argmax(eigenvalues.real)
        max_eigenvalue = eigenvalues[max_eigenvalue_idx].real
        principal_eigenvector = eigenvectors[:, max_eigenvalue_idx].real
        
        # Normalize the eigenvector to get weights
        weights = principal_eigenvector / np.sum(principal_eigenvector)
        weights = np.abs(weights)  # Ensure positive weights
        
        # Calculate consistency ratio
        consistency_index = (max_eigenvalue - n) / (n - 1) if n > 1 else 0
        random_index = self._get_random_index(n)
        consistency_ratio = consistency_index / random_index if random_index > 0 else 0
        
        return weights, consistency_ratio
    
    def _get_random_index(self, n):
        """Get random index for consistency calculation"""
        # Random indices for matrices of different sizes
        random_indices = {
            1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
        }
        return random_indices.get(n, 1.49)
    
    def calculate(self):
        """
        Calculate AHP scores and rankings
        
        Returns:
            dict: Results with scores, rankings, and intermediate steps
        """
        if self.criteria_matrix is None:
            raise ValueError("Criteria comparison matrix not set")
        
        # Step 1: Calculate criteria weights
        self.criteria_weights, criteria_cr = self.calculate_priority_weights(self.criteria_matrix)
        self.consistency_ratios['criteria'] = criteria_cr
        
        # Step 2: Calculate alternative weights for each criterion
        for i, criterion in enumerate(self.criteria):
            if criterion in self.alternative_matrices:
                weights, cr = self.calculate_priority_weights(self.alternative_matrices[criterion])
                self.alternative_weights[criterion] = weights
                self.consistency_ratios[criterion] = cr
            else:
                # If no pairwise matrix provided, use equal weights
                n_alternatives = len(self.alternatives)
                self.alternative_weights[criterion] = np.ones(n_alternatives) / n_alternatives
                self.consistency_ratios[criterion] = 0
        
        # Step 3: Calculate final scores
        final_scores = np.zeros(len(self.alternatives))
        
        for i, criterion in enumerate(self.criteria):
            criterion_weight = self.criteria_weights[i]
            alternative_weights = self.alternative_weights.get(criterion, 
                                                             np.ones(len(self.alternatives)) / len(self.alternatives))
            final_scores += criterion_weight * alternative_weights
        
        # Step 4: Calculate rankings
        rankings = self.get_ranking(final_scores)
        
        # Store results
        self.results = {
            'scores': final_scores.tolist(),
            'rankings': rankings,
            'method': 'AHP'
        }
        
        # Store intermediate steps
        self.intermediate_steps = {
            'criteria_matrix': self.criteria_matrix,
            'criteria_weights': self.criteria_weights,
            'alternative_matrices': self.alternative_matrices,
            'alternative_weights': self.alternative_weights,
            'consistency_ratios': self.consistency_ratios,
            'final_scores': final_scores
        }
        
        return self.results
    
    def get_results_dataframe(self):
        """Get results as a formatted DataFrame"""
        if not self.results:
            self.calculate()
        
        df = pd.DataFrame({
            'Alternative': self.alternatives,
            'Score': self.results['scores'],
            'Rank': self.results['rankings']
        })
        
        # Sort by rank
        df = df.sort_values('Rank').reset_index(drop=True)
        
        return df
    
    def is_consistent(self, threshold=0.1):
        """
        Check if all pairwise comparisons are consistent
        
        Args:
            threshold (float): Consistency ratio threshold (default 0.1)
            
        Returns:
            dict: Consistency check results
        """
        if not self.consistency_ratios:
            return {'overall_consistent': False, 'message': 'No consistency ratios calculated'}
        
        inconsistent_matrices = []
        
        for matrix_name, cr in self.consistency_ratios.items():
            if cr > threshold:
                inconsistent_matrices.append((matrix_name, cr))
        
        overall_consistent = len(inconsistent_matrices) == 0
        
        return {
            'overall_consistent': overall_consistent,
            'inconsistent_matrices': inconsistent_matrices,
            'threshold': threshold,
            'all_ratios': self.consistency_ratios
        }
    
    def create_pairwise_matrix_from_comparisons(self, comparisons):
        """
        Create pairwise comparison matrix from list of comparisons
        
        Args:
            comparisons (list): List of comparison values (upper triangular)
            
        Returns:
            np.array: Pairwise comparison matrix
        """
        n = int((1 + np.sqrt(1 + 8 * len(comparisons))) / 2)
        matrix = np.ones((n, n))
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i, j] = comparisons[idx]
                matrix[j, i] = 1.0 / comparisons[idx]
                idx += 1
        
        return matrix

    def get_step_by_step_explanation(self):
        """
        Get detailed step-by-step explanation of the AHP calculation

        Returns:
            dict: Step-by-step explanation with matrices and descriptions
        """
        if not self.results:
            self.calculate()

        explanation = {
            'method_name': 'Analytic Hierarchy Process (AHP)',
            'description': 'AHP uses pairwise comparisons to derive priority weights and make decisions.',
            'steps': []
        }

        # Step 1: Criteria comparison matrix
        criteria_df = pd.DataFrame(
            self.intermediate_steps['criteria_matrix'],
            index=self.criteria,
            columns=self.criteria
        )

        explanation['steps'].append({
            'step_number': 1,
            'title': 'Criteria Pairwise Comparison Matrix',
            'description': 'Matrix showing relative importance between criteria pairs.',
            'matrix': criteria_df,
            'formula': 'A[i,j] = importance of criterion i relative to criterion j'
        })

        # Step 2: Criteria weights
        criteria_weights_df = pd.DataFrame({
            'Criterion': self.criteria,
            'Weight': self.intermediate_steps['criteria_weights'],
            'Consistency Ratio': [self.consistency_ratios.get('criteria', 0)] * len(self.criteria)
        })

        explanation['steps'].append({
            'step_number': 2,
            'title': 'Criteria Priority Weights',
            'description': 'Weights derived from the principal eigenvector of the comparison matrix.',
            'matrix': criteria_weights_df,
            'formula': 'w = principal eigenvector of comparison matrix'
        })

        # Step 3: Alternative comparison matrices (show first criterion as example)
        if self.criteria and self.criteria[0] in self.alternative_matrices:
            first_criterion = self.criteria[0]
            alt_matrix_df = pd.DataFrame(
                self.alternative_matrices[first_criterion],
                index=self.alternatives,
                columns=self.alternatives
            )

            explanation['steps'].append({
                'step_number': 3,
                'title': f'Alternative Comparisons for "{first_criterion}"',
                'description': f'Pairwise comparison of alternatives with respect to {first_criterion}.',
                'matrix': alt_matrix_df,
                'formula': 'A[i,j] = preference of alternative i over j for this criterion'
            })

        # Step 4: Final synthesis
        synthesis_data = []
        for i, alt in enumerate(self.alternatives):
            row = {'Alternative': alt}
            for j, crit in enumerate(self.criteria):
                alt_weight = self.alternative_weights.get(crit, [0] * len(self.alternatives))[i]
                crit_weight = self.criteria_weights[j]
                contribution = alt_weight * crit_weight
                row[f'{crit} (Weight)'] = f'{alt_weight:.3f}'
                row[f'{crit} (Contribution)'] = f'{contribution:.3f}'
            row['Final Score'] = self.results['scores'][i]
            row['Rank'] = self.results['rankings'][i]
            synthesis_data.append(row)

        synthesis_df = pd.DataFrame(synthesis_data)

        explanation['steps'].append({
            'step_number': 4,
            'title': 'Final Synthesis and Ranking',
            'description': 'Combine criteria weights with alternative weights to get final scores.',
            'matrix': synthesis_df,
            'formula': 'Final Score = Σ(criteria_weight[j] × alternative_weight[i,j])'
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
            'name': 'Analytic Hierarchy Process (AHP)',
            'other_names': ['Pairwise Comparison Method'],
            'complexity': 'Advanced',
            'description': '''
            AHP is a structured technique for organizing and analyzing complex decisions.
            It breaks down a decision problem into a hierarchy of criteria and alternatives,
            then uses pairwise comparisons to derive priority weights.
            ''',
            'when_to_use': [
                'When you need to involve multiple stakeholders in decision making',
                'When criteria importance is subjective and hard to quantify directly',
                'When you want to check consistency of judgments',
                'For complex decisions with multiple levels of criteria',
                'When transparency in the decision process is important'
            ],
            'advantages': [
                'Handles both quantitative and qualitative criteria',
                'Provides consistency checking mechanism',
                'Involves stakeholders in the decision process',
                'Breaks complex problems into manageable parts',
                'Well-established theoretical foundation',
                'Widely accepted and used in practice'
            ],
            'disadvantages': [
                'Can be time-consuming for large problems',
                'Requires many pairwise comparisons',
                'Susceptible to rank reversal',
                'May be difficult for decision makers to provide consistent judgments',
                'Limited to 9-point comparison scale'
            ],
            'mathematical_foundation': '''
            1. Create pairwise comparison matrices:
               A[i,j] = relative importance of element i over element j

            2. Calculate priority weights using eigenvalue method:
               Aw = λ_max × w (where w is the principal eigenvector)

            3. Check consistency:
               CI = (λ_max - n) / (n - 1)
               CR = CI / RI (where RI is random index)
               CR < 0.1 indicates acceptable consistency

            4. Synthesize results:
               Final Score = Σ(criteria_weight[j] × alternative_weight[i,j])
            ''',
            'saaty_scale': {
                '1': 'Equal importance',
                '3': 'Moderate importance',
                '5': 'Strong importance',
                '7': 'Very strong importance',
                '9': 'Extreme importance',
                '2,4,6,8': 'Intermediate values'
            },
            'consistency_guidelines': {
                'CR < 0.1': 'Acceptable consistency',
                '0.1 ≤ CR < 0.2': 'Marginal consistency (review recommended)',
                'CR ≥ 0.2': 'Unacceptable consistency (revision required)'
            }
        }
