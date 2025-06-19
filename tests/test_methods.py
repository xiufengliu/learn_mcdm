"""
Unit tests for MCDM methods
"""

import unittest
import numpy as np
import pandas as pd
import warnings
from src.methods.saw import SAW
from src.methods.wpm import WPM
from src.methods.topsis import TOPSIS
from src.methods.ahp import AHP
from src.utils.validation import validate_mcdm_inputs, validate_decision_matrix, validate_weights

class TestSAW(unittest.TestCase):
    """Test cases for SAW method"""
    
    def setUp(self):
        """Set up test data"""
        self.decision_matrix = pd.DataFrame([
            [25000, 32, 5, 7, 8],
            [27000, 30, 5, 6, 7],
            [35000, 25, 4, 9, 9],
            [38000, 27, 4, 8, 9]
        ], 
        index=['Toyota Camry', 'Honda Accord', 'BMW 3 Series', 'Audi A4'],
        columns=['Price', 'Fuel Economy', 'Safety', 'Performance', 'Comfort'])
        
        self.weights = [0.3, 0.2, 0.25, 0.15, 0.1]
        self.criterion_types = ['cost', 'benefit', 'benefit', 'benefit', 'benefit']
    
    def test_saw_calculation(self):
        """Test SAW calculation"""
        saw = SAW(self.decision_matrix, self.weights, self.criterion_types)
        results = saw.calculate()
        
        # Check that results are returned
        self.assertIsNotNone(results)
        self.assertIn('scores', results)
        self.assertIn('rankings', results)
        
        # Check that we have the right number of scores and rankings
        self.assertEqual(len(results['scores']), 4)
        self.assertEqual(len(results['rankings']), 4)
        
        # Check that rankings are valid (1-4)
        rankings = results['rankings']
        self.assertEqual(set(rankings), {1, 2, 3, 4})
    
    def test_saw_normalization(self):
        """Test SAW normalization"""
        saw = SAW(self.decision_matrix, self.weights, self.criterion_types)
        normalized = saw.normalize_matrix()

        # Check that normalized values are between 0 and 1
        self.assertTrue((normalized >= 0).all().all())
        self.assertTrue((normalized <= 1).all().all())

    def test_linear_normalization_constant_column(self):
        """Linear normalization handles constant columns"""
        matrix = pd.DataFrame(
            [[5, 10], [5, 10], [5, 10]],
            columns=["C1", "C2"],
            index=["A1", "A2", "A3"],
        )
        weights = [0.5, 0.5]
        c_types = ["benefit", "cost"]
        saw = SAW(matrix, weights, c_types)
        normalized = saw.normalize_matrix()
        self.assertTrue((normalized["C1"] == 0).all())
        self.assertTrue((normalized["C2"] == 0).all())

    def test_vector_normalization_zero_values(self):
        """Vector normalization avoids division by zero"""
        matrix = pd.DataFrame(
            [[0, 1], [0, 2], [0, 3]],
            columns=["C1", "C2"],
            index=["A1", "A2", "A3"],
        )
        weights = [0.5, 0.5]
        c_types = ["benefit", "cost"]
        saw = SAW(matrix, weights, c_types)
        normalized = saw.normalize_matrix(method="vector")
        self.assertFalse(np.isinf(normalized.values).any())
        self.assertFalse(np.isnan(normalized.values).any())

    def test_vector_normalization_negative_costs(self):
        """Vector normalization handles negative or zero cost values gracefully"""
        matrix = pd.DataFrame(
            [[-1, 1], [0, 2], [1, 3]],
            columns=["Cost", "Benefit"],
            index=["A1", "A2", "A3"],
        )
        weights = [0.5, 0.5]
        c_types = ["cost", "benefit"]
        saw = SAW(matrix, weights, c_types)
        normalized = saw.normalize_matrix(method="vector")
        self.assertFalse(np.isinf(normalized.values).any())
        self.assertFalse(np.isnan(normalized.values).any())

class TestWPM(unittest.TestCase):
    """Test cases for WPM method"""
    
    def setUp(self):
        """Set up test data"""
        self.decision_matrix = pd.DataFrame([
            [25000, 32, 5, 7, 8],
            [27000, 30, 5, 6, 7],
            [35000, 25, 4, 9, 9],
            [38000, 27, 4, 8, 9]
        ], 
        index=['Toyota Camry', 'Honda Accord', 'BMW 3 Series', 'Audi A4'],
        columns=['Price', 'Fuel Economy', 'Safety', 'Performance', 'Comfort'])
        
        self.weights = [0.3, 0.2, 0.25, 0.15, 0.1]
        self.criterion_types = ['cost', 'benefit', 'benefit', 'benefit', 'benefit']
    
    def test_wpm_calculation(self):
        """Test WPM calculation"""
        wpm = WPM(self.decision_matrix, self.weights, self.criterion_types)
        results = wpm.calculate()
        
        # Check that results are returned
        self.assertIsNotNone(results)
        self.assertIn('scores', results)
        self.assertIn('rankings', results)
        
        # Check that we have the right number of scores and rankings
        self.assertEqual(len(results['scores']), 4)
        self.assertEqual(len(results['rankings']), 4)
        
        # Check that rankings are valid (1-4)
        rankings = results['rankings']
        self.assertEqual(set(rankings), {1, 2, 3, 4})
        
        # Check that scores are positive (since we use products)
        scores = results['scores']
        self.assertTrue(all(score > 0 for score in scores))

    def test_wpm_zero_cost_handling(self):
        """WPM handles zero values in cost criteria without infinities"""
        matrix = pd.DataFrame(
            [[0, 1], [1, 1], [2, 1]],
            columns=["Cost", "Benefit"],
            index=["A1", "A2", "A3"],
        )
        weights = [0.5, 0.5]
        c_types = ["cost", "benefit"]
        wpm = WPM(matrix, weights, c_types)
        results = wpm.calculate()
        vals = np.array(results["scores"], dtype=float)
        self.assertFalse(np.isinf(vals).any())
        self.assertFalse(np.isnan(vals).any())

class TestTOPSIS(unittest.TestCase):
    """Test cases for TOPSIS method"""

    def setUp(self):
        """Set up test data"""
        self.decision_matrix = pd.DataFrame([
            [25000, 32, 5, 7, 8],
            [27000, 30, 5, 6, 7],
            [35000, 25, 4, 9, 9],
            [38000, 27, 4, 8, 9]
        ],
        index=['Toyota Camry', 'Honda Accord', 'BMW 3 Series', 'Audi A4'],
        columns=['Price', 'Fuel Economy', 'Safety', 'Performance', 'Comfort'])

        self.weights = [0.3, 0.2, 0.25, 0.15, 0.1]
        self.criterion_types = ['cost', 'benefit', 'benefit', 'benefit', 'benefit']

    def test_topsis_calculation(self):
        """Test TOPSIS calculation"""
        topsis = TOPSIS(self.decision_matrix, self.weights, self.criterion_types)
        results = topsis.calculate()

        # Check that results are returned
        self.assertIsNotNone(results)
        self.assertIn('scores', results)
        self.assertIn('rankings', results)

        # Check that we have the right number of scores and rankings
        self.assertEqual(len(results['scores']), 4)
        self.assertEqual(len(results['rankings']), 4)

        # Check that rankings are valid (1-4)
        rankings = results['rankings']
        self.assertEqual(set(rankings), {1, 2, 3, 4})

        # Check that scores are between 0 and 1 (relative closeness)
        scores = results['scores']
        self.assertTrue(all(0 <= score <= 1 for score in scores))

    def test_topsis_intermediate_steps(self):
        """Test TOPSIS intermediate steps"""
        topsis = TOPSIS(self.decision_matrix, self.weights, self.criterion_types)
        topsis.calculate()

        # Check that intermediate steps are stored
        self.assertIn('normalized_matrix', topsis.intermediate_steps)
        self.assertIn('weighted_matrix', topsis.intermediate_steps)
        self.assertIn('pis', topsis.intermediate_steps)
        self.assertIn('nis', topsis.intermediate_steps)

class TestAHP(unittest.TestCase):
    """Test cases for AHP method"""

    def setUp(self):
        """Set up test data"""
        self.alternatives = ['Alternative A', 'Alternative B', 'Alternative C']
        self.criteria = ['Criterion 1', 'Criterion 2', 'Criterion 3']

        # Simple criteria comparison matrix (Criterion 1 > Criterion 2 > Criterion 3)
        self.criteria_matrix = np.array([
            [1, 3, 5],
            [1/3, 1, 3],
            [1/5, 1/3, 1]
        ])

        # Alternative comparison matrices
        self.alt_matrix_1 = np.array([
            [1, 2, 4],
            [1/2, 1, 3],
            [1/4, 1/3, 1]
        ])

    def test_ahp_priority_weights(self):
        """Test AHP priority weight calculation"""
        ahp = AHP(alternatives=self.alternatives, criteria=self.criteria)
        weights, cr = ahp.calculate_priority_weights(self.criteria_matrix)

        # Check that weights sum to 1
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)

        # Check that weights are positive
        self.assertTrue(all(w > 0 for w in weights))

        # Check that consistency ratio is calculated
        self.assertIsInstance(cr, float)
        self.assertGreaterEqual(cr, 0)

    def test_ahp_calculation(self):
        """Test full AHP calculation"""
        ahp = AHP(alternatives=self.alternatives, criteria=self.criteria)
        ahp.set_criteria_comparison_matrix(self.criteria_matrix)
        ahp.set_alternative_comparison_matrix(self.criteria[0], self.alt_matrix_1)

        results = ahp.calculate()

        # Check that results are returned
        self.assertIsNotNone(results)
        self.assertIn('scores', results)
        self.assertIn('rankings', results)

        # Check that we have the right number of scores and rankings
        self.assertEqual(len(results['scores']), 3)
        self.assertEqual(len(results['rankings']), 3)

    def test_ahp_consistency_check(self):
        """Test AHP consistency checking"""
        ahp = AHP(alternatives=self.alternatives, criteria=self.criteria)
        ahp.set_criteria_comparison_matrix(self.criteria_matrix)
        ahp.calculate()

        consistency_results = ahp.is_consistent()

        # Check that consistency results are returned
        self.assertIn('overall_consistent', consistency_results)
        self.assertIn('all_ratios', consistency_results)
        self.assertIsInstance(consistency_results['overall_consistent'], bool)

class TestValidation(unittest.TestCase):
    """Test cases for validation utilities"""

    def test_validate_decision_matrix_empty(self):
        """Test validation with empty matrix"""
        empty_matrix = pd.DataFrame()
        is_valid, errors, warnings = validate_decision_matrix(empty_matrix)
        self.assertFalse(is_valid)
        self.assertIn("empty", errors[0].lower())

    def test_validate_decision_matrix_missing_values(self):
        """Test validation with missing values"""
        matrix = pd.DataFrame([[1, 2], [np.nan, 4]], columns=['C1', 'C2'])
        is_valid, errors, warnings = validate_decision_matrix(matrix)
        self.assertFalse(is_valid)
        self.assertIn("missing values", errors[0].lower())

    def test_validate_decision_matrix_non_numeric(self):
        """Test validation with non-numeric values"""
        matrix = pd.DataFrame([['a', 2], [3, 4]], columns=['C1', 'C2'])
        is_valid, errors, warnings = validate_decision_matrix(matrix)
        self.assertFalse(is_valid)
        self.assertIn("non-numeric", errors[0].lower())

    def test_validate_weights_negative(self):
        """Test validation with negative weights"""
        weights = [0.5, -0.3, 0.8]
        is_valid, errors, warnings, normalized = validate_weights(weights)
        self.assertFalse(is_valid)
        self.assertIn("negative", errors[0].lower())

    def test_validate_weights_zero_sum(self):
        """Test validation with zero sum weights"""
        weights = [0, 0, 0]
        is_valid, errors, warnings, normalized = validate_weights(weights)
        self.assertFalse(is_valid)
        self.assertIn("zero", errors[0].lower())


class TestEdgeCases(unittest.TestCase):
    """Test cases for edge cases across all methods"""

    def setUp(self):
        """Set up edge case test data"""
        # Matrix with identical values in one column
        self.constant_column_matrix = pd.DataFrame([
            [5, 10, 3],
            [5, 20, 4],
            [5, 15, 5]
        ], columns=['Constant', 'Variable', 'Another'])

        # Matrix with extreme values
        self.extreme_values_matrix = pd.DataFrame([
            [1, 1000000, 0.001],
            [2, 1000001, 0.002],
            [3, 1000002, 0.003]
        ], columns=['Small', 'Large', 'Tiny'])

        # Matrix with negative values
        self.negative_values_matrix = pd.DataFrame([
            [-5, 10, 3],
            [-3, 20, 4],
            [-1, 15, 5]
        ], columns=['Negative', 'Positive', 'Another'])

        self.weights = [0.33, 0.33, 0.34]
        self.criterion_types = ['benefit', 'benefit', 'benefit']

    def test_saw_constant_column(self):
        """Test SAW with constant column values"""
        saw = SAW(self.constant_column_matrix, self.weights, self.criterion_types)
        results = saw.calculate()
        self.assertIsNotNone(results)
        self.assertEqual(len(results['scores']), 3)

    def test_saw_extreme_values(self):
        """Test SAW with extreme values"""
        saw = SAW(self.extreme_values_matrix, self.weights, self.criterion_types)
        results = saw.calculate()
        self.assertIsNotNone(results)
        self.assertFalse(any(np.isnan(results['scores'])))
        self.assertFalse(any(np.isinf(results['scores'])))

    def test_wpm_negative_values(self):
        """Test WPM with negative values (should handle gracefully)"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress expected warnings
            wpm = WPM(self.negative_values_matrix, self.weights, self.criterion_types)
            results = wpm.calculate()
            self.assertIsNotNone(results)
            self.assertFalse(any(np.isnan(results['scores'])))
            self.assertFalse(any(np.isinf(results['scores'])))

    def test_topsis_constant_column(self):
        """Test TOPSIS with constant column values"""
        topsis = TOPSIS(self.constant_column_matrix, self.weights, self.criterion_types)
        results = topsis.calculate()
        self.assertIsNotNone(results)
        self.assertTrue(all(0 <= score <= 1 for score in results['scores']))

    def test_single_criterion(self):
        """Test methods with single criterion"""
        single_criterion_matrix = pd.DataFrame([[5], [10], [15]], columns=['Only'])
        weights = [1.0]
        criterion_types = ['benefit']

        # Test SAW
        saw = SAW(single_criterion_matrix, weights, criterion_types)
        saw_results = saw.calculate()
        self.assertEqual(len(saw_results['scores']), 3)

        # Test TOPSIS
        topsis = TOPSIS(single_criterion_matrix, weights, criterion_types)
        topsis_results = topsis.calculate()
        self.assertEqual(len(topsis_results['scores']), 3)


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling"""

    def test_invalid_matrix_dimensions(self):
        """Test error handling for invalid matrix dimensions"""
        # Single alternative
        single_alt_matrix = pd.DataFrame([[1, 2, 3]], columns=['C1', 'C2', 'C3'])
        weights = [0.33, 0.33, 0.34]
        criterion_types = ['benefit', 'benefit', 'benefit']

        with self.assertRaises(ValueError):
            SAW(single_alt_matrix, weights, criterion_types)

    def test_mismatched_weights(self):
        """Test error handling for mismatched weights"""
        matrix = pd.DataFrame([[1, 2], [3, 4]], columns=['C1', 'C2'])
        wrong_weights = [0.5, 0.3, 0.2]  # 3 weights for 2 criteria
        criterion_types = ['benefit', 'benefit']

        with self.assertRaises(ValueError):
            SAW(matrix, wrong_weights, criterion_types)

    def test_invalid_criterion_types(self):
        """Test error handling for invalid criterion types"""
        matrix = pd.DataFrame([[1, 2], [3, 4]], columns=['C1', 'C2'])
        weights = [0.5, 0.5]
        invalid_types = ['benefit', 'invalid']

        with self.assertRaises(ValueError):
            SAW(matrix, weights, invalid_types)

    def test_ahp_invalid_pairwise_matrix(self):
        """Test AHP error handling for invalid pairwise matrices"""
        alternatives = ['A', 'B', 'C']
        criteria = ['C1', 'C2']

        # Invalid matrix (not reciprocal)
        invalid_matrix = np.array([
            [1, 3],
            [2, 1]  # Should be 1/3, not 2
        ])

        ahp = AHP(alternatives=alternatives, criteria=criteria)
        with self.assertRaises(ValueError):
            ahp.calculate_priority_weights(invalid_matrix)

    def test_wpm_all_zero_values(self):
        """Test WPM error handling for all zero values in a column"""
        matrix = pd.DataFrame([[0, 1], [0, 2]], columns=['Zero', 'Nonzero'])
        weights = [0.5, 0.5]
        criterion_types = ['benefit', 'benefit']

        with self.assertRaises(ValueError):
            wpm = WPM(matrix, weights, criterion_types)
            wmp.calculate()


if __name__ == '__main__':
    unittest.main()
