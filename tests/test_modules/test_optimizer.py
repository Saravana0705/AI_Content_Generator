import unittest
from unittest.mock import patch, MagicMock
from sub_agents.text_generator.modules.optimizer.optimizer import Optimizer
class TestOptimizer(unittest.TestCase):

    @patch("sub_agents.text_generator.modules.optimizer.optimizer.Generator")
    def test_optimize_basic(self, MockGenerator):
        """Optimizer should return text, score and notes in the expected format."""
        mock_gen_instance = MockGenerator.return_value
        mock_gen_instance.generate.return_value = "Adjusted tone output"

        optimizer = Optimizer()

        optimized_text, final_score, notes = optimizer.optimize(
            text="Test content",
            original_topic="marketing",
            content_type="blog_article",
            tone="friendly",
            auto_revise=False,   # disable auto-revision in this unit test
        )

        # --- Assertions ---
        self.assertIsInstance(optimized_text, str)
        self.assertIsInstance(final_score, (int, float))
        self.assertTrue(0 <= final_score <= 100)
        self.assertIsInstance(notes, str)

        # Ensure tone-adjustment LLM call was used once in this controlled setting
        mock_gen_instance.generate.assert_called_once()

        # Ensure notes contain expected content
        self.assertIn("Optimization Analysis Report", notes)
        self.assertIn("Final Optimization Score", notes)


if __name__ == "__main__":
    unittest.main()
