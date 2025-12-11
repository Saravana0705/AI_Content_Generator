import unittest
from unittest.mock import patch, MagicMock

from sub_agents.text_generator.modules.generator.content_generator import Generator


class TestGenerator(unittest.TestCase):

    @patch(
        "sub_agents.text_generator.modules.generator.content_generator.client.chat.completions.create"
    )
    def test_generate_success(self, mock_create):
        # Arrange
        generator = Generator()

        # Fake OpenAI-style response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Mocked output"))
        ]
        mock_create.return_value = mock_response

        # Act
        result = generator.generate("Hello")

        # Assert
        mock_create.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        self.assertEqual(result, "Mocked output")

    @patch(
        "sub_agents.text_generator.modules.generator.content_generator.client.chat.completions.create"
    )
    def test_generate_error(self, mock_create):
        # Arrange
        generator = Generator()
        mock_create.side_effect = Exception("API failure")

        # Act
        result = generator.generate("Hello")

        # Assert
        self.assertIn("Error generating content", result)
        self.assertIn("API failure", result)


if __name__ == "__main__":
    unittest.main()
