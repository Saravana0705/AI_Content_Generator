import unittest
from unittest.mock import patch, MagicMock

from sub_agents.text_generator.modules.generator.content_generator import Generator


class TestGenerator(unittest.TestCase):

    @patch(
        "sub_agents.text_generator.modules.generator.content_generator.client.chat.completions.create"
    )
    def test_generate_success(self, mock_create):
        generator = Generator()

        # Fake OpenAI-style response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Mocked blog output"))
        ]
        mock_create.return_value = mock_response

        result = generator.generate("Test blog post")

        mock_create.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Test blog post"}],
        )

        self.assertEqual(result, "Mocked blog output")

    @patch(
        "sub_agents.text_generator.modules.generator.content_generator.client.chat.completions.create"
    )
    def test_generate_error(self, mock_create):
        generator = Generator()

        mock_create.side_effect = Exception("API failure")

        result = generator.generate("Hello")

        self.assertIn("Error generating content", result)
        self.assertIn("API failure", result)


if __name__ == "__main__":
    unittest.main()
