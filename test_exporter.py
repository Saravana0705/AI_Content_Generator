import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from src.sub_agents.text_generator.modules.exporter.exporter import Exporter


class TestExporter(unittest.TestCase):
    def test_export(self):
        exporter = Exporter()
        result = exporter.export("Test content", "WordPress")
        self.assertTrue("Exported to WordPress" in result)


if __name__ == "__main__":
    unittest.main()
