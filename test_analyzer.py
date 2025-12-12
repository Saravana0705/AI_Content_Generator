import unittest
from src.sub_agents.text_generator.modules.analyzer.analyzer import Analyzer

class TestAnalyzer(unittest.TestCase):
    def test_analyze_input(self):
        analyzer = Analyzer()
        result = analyzer.analyze_input("Test input")
        self.assertTrue("Analyzed" in result)
    
    def test_analyze_full(self):
        """Test full analyze method with comprehensive output."""
        analyzer = Analyzer()
        text = "This is a test sentence. It has multiple words and punctuation."
        result = analyzer.analyze(text)
        
        # Check that all expected keys are present
        self.assertIn("readability", result)
        self.assertIn("sentiment", result)
        self.assertIn("keywords", result)
        self.assertIn("spelling", result)
        self.assertIn("stats", result)
        
        # Check stats
        # expected word count is 11 for the provided sentence
        self.assertEqual(result["stats"]["word_count"], 11)
        self.assertGreater(result["stats"]["sentence_count"], 0)

if __name__ == "__main__":
    unittest.main()