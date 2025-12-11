from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
      raise ValueError("OPENAI_API_KEY not found in environment variables")

client = OpenAI(api_key=api_key)

class Generator:
      def generate(self, input_text):
          try:
              response = client.chat.completions.create(
                  model="gpt-4o",
                  messages=[{"role": "user", "content": input_text}]
              )
              return response.choices[0].message.content
          except Exception as e:
              return f"Error generating content: {str(e)}"

if __name__ == "__main__":
      generator = Generator()
      result = generator.generate("Test blog post")
      print(result)