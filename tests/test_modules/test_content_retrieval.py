import unittest
from sub_agents.text_generator.modules.content_retrieval.llamaindex_retriever import Retriever

class TestContentRetrieval(unittest.TestCase):
    def test_retrieve(self):
        retriever = Retriever()
        result = retriever.retrieve("Test query")
        self.assertTrue(isinstance(result, str))

if __name__ == "__main__":
    unittest.main()