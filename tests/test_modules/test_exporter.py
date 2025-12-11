import unittest
from sub_agents.text_generator.modules.exporter.exporter import Exporter

class TestExporter(unittest.TestCase):
    def test_export(self):
        exporter = Exporter()
        result = exporter.export("Test content", "WordPress")
        self.assertTrue("Exported to WordPress" in result)

if __name__ == "__main__":
    unittest.main()