import unittest
from sub_agents.text_generator.modules.human_review.reviewer import Reviewer

class TestHumanReview(unittest.TestCase):
    def test_review(self):
        reviewer = Reviewer()
        result = reviewer.review("Test content")
        self.assertTrue("Reviewed" in result)

if __name__ == "__main__":
    unittest.main()