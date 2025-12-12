import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sub_agents.text_generator.modules.exporter.exporter import Exporter


class TestExporter(unittest.TestCase):
    def setUp(self):
        self.exporter = Exporter()

    def test_export_wordpress(self):
        result = self.exporter.export("Test content", "WordPress")
        self.assertIn("Successfully exported to WordPress", result)
        self.assertIn("Content length: 12 characters", result)

    def test_export_wordpress_with_title(self):
        result = self.exporter.export("Test content", "WordPress",
                                      title="My Title")
        self.assertIn("Title: My Title", result)

    def test_export_medium(self):
        result = self.exporter.export("Test content", "Medium")
        self.assertIn("Successfully exported to Medium", result)

    def test_export_medium_with_tags(self):
        result = self.exporter.export("Test content", "Medium",
                                      tags=["AI", "Tech"])
        self.assertIn("Tags: AI, Tech", result)

    def test_export_linkedin(self):
        result = self.exporter.export("Test content", "LinkedIn")
        self.assertIn("Successfully posted to LinkedIn", result)

    def test_export_twitter(self):
        result = self.exporter.export("Test content", "Twitter")
        self.assertIn("Successfully tweeted", result)

    def test_export_twitter_long_content(self):
        long_content = "A" * 300
        result = self.exporter.export(long_content, "Twitter")
        self.assertIn("Successfully tweeted", result)
        self.assertLessEqual(len(result.split(": ")[1]), 280)

    def test_export_unknown_platform(self):
        result = self.exporter.export("Test content", "Unknown")
        self.assertEqual(result, "Exported to unknown: Test content")


if __name__ == "__main__":
    unittest.main()
