import unittest
from sub_agents.text_generator.modules.analyzer.analyzer import Analyzer

class TestAnalyzer(unittest.TestCase):
    def test_analyze_input(self):
        analyzer = Analyzer()
        result = analyzer.analyze_input("Test input")
        self.assertTrue("Analyzed" in result)

if __name__ == "__main__":
    unittest.main()