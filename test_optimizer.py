import unittest
from sub_agents.text_generator.modules.optimizer.optimizer import Optimizer

class TestOptimizer(unittest.TestCase):
    def test_optimize(self):
        optimizer = Optimizer()
        # The Optimizer.optimize signature returns (optimized_text, score, notes)
        result = optimizer.optimize("Test content", "test topic")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], str)

if __name__ == "__main__":
    unittest.main()