import os
from dotenv import load_dotenv
from src.sub_agents.text_generator.modules.exporter.exporter import Exporter

load_dotenv()

md_path = r"exports\create_a_detailed_blog_post_about_the_benefits_of_ai_in_trav_blog_article_20260111_150126\create_a_detailed_blog_post_about_the_benefits_of_ai_in_trav_blog_article.md"

with open(md_path, "r", encoding="utf-8") as f:
    md = f.read()

exporter = Exporter()
res = exporter.publish_to_devto(
    title="Benefits of AI in Travel Industry (Published via Exporter)",
    body_markdown=md,
    published=False,  # draft for safety
    tags=["ai", "travel", "marketing"],
)

print(res)
