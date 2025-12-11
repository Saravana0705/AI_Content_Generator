import unittest
from unittest.mock import patch, MagicMock

from sub_agents.text_generator.modules.optimizer.optimizer import Optimizer


class TestOptimizer(unittest.TestCase):

    @patch("sub_agents.text_generator.modules.optimizer.optimizer.Generator")
    def test_optimize(self, MockGenerator):
        mock_gen_instance = MockGenerator.return_value
        mock_gen_instance.generate.return_value = "Adjusted tone output"

        optimizer = Optimizer()

        optimized_text, final_score, notes = optimizer.optimize(
            text="Test content",
            original_topic="marketing",
            content_type="blog_article",
            tone="friendly"
        )

        # --- Assertions ---
        self.assertIsInstance(optimized_text, str)
        self.assertIsInstance(final_score, (int, float))
        self.assertTrue(0 <= final_score <= 100)
        self.assertIsInstance(notes, str)

        # Ensure LLM tone adjustment was used
        mock_gen_instance.generate.assert_called_once()

        # Ensure notes contain expected content
        self.assertIn("Optimization Analysis Report", notes)
        self.assertIn("Final Optimization Score", notes)


if __name__ == "__main__":
    unittest.main()
