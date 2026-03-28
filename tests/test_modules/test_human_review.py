import unittest
from sub_agents.text_generator.modules.human_review.reviewer import Reviewer

class TestHumanReview(unittest.TestCase):
    def setUp(self):
        self.reviewer = Reviewer(threshold=60.0)

    def test_high_score_is_approved(self):
        """Content with a high score should be approved."""
        content = "This is clear, simple marketing content."
        score = 85.0
        notes = "Optimization Analysis Report: readability good."

        result = self.reviewer.review(content=content, score=score, notes=notes)

        self.assertIsInstance(result, dict)
        self.assertTrue(result["approved"])
        self.assertEqual(result["decision"], "approve")
        # For 85, quality band should be 'excellent'
        self.assertEqual(result["quality_band"], "excellent")
        self.assertEqual(result["content"], content)

    def test_low_score_needs_revision(self):
        """Content with a low score should not be approved."""
        content = "Hard to read, very complex and not suitable for general readers."
        score = 40.0  # below 0.7 * threshold (42)
        notes = "Optimization Analysis Report: readability very difficult."

        result = self.reviewer.review(content=content, score=score, notes=notes)

        self.assertIsInstance(result, dict)
        self.assertFalse(result["approved"])
        self.assertEqual(result["decision"], "revise")
        # For 40, with threshold=60, quality band = 'needs_major_revision'
        self.assertEqual(result["quality_band"], "needs_major_revision")
        self.assertIn("below the quality benchmark", " ".join(result["comments"]).lower())

    def test_notes_trigger_readability_comment(self):
        """If 'readability' appears in notes, a readability hint should be added."""
        content = "Some text to test readability hint."
        score = 55.0  # below threshold but above 0.7 * threshold
        notes = "The main issue here is readability and sentence structure."

        result = self.reviewer.review(content=content, score=score, notes=notes)

        comments_text = " ".join(result["comments"]).lower()
        self.assertIn("readability", comments_text)
        self.assertFalse(result["approved"])  # still below threshold


if __name__ == "__main__":
    unittest.main()
