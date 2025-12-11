from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CONTENT_TYPE = "text"
    CONTENT_CATEGORY = "blog_post"