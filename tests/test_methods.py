"""
Unit tests for MCDM methods
"""

import unittest
import numpy as np
import pandas as pd
from src.methods.saw import SAW
from src.methods.wpm import WPM
from src.methods.topsis import TOPSIS
from src.methods.ahp import AHP

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

if __name__ == '__main__':
    unittest.main()
