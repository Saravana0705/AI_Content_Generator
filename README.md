# AI Content Generator

## Overview
The AI Content Generator is a modular system designed to automatically generate marketing text and images. It uses a multi-agent architecture where a central workflow coordinates specialized agents responsible for text and image generation.
The system integrates prompt processing, retrieval mechanisms, quality evaluation and optimization to produce reliable and high-quality outputs. It is designed to be extensible and supports multiple models and languages.


## Key Features
- Text generation for marketing content (blogs, ads, descriptions, etc.)
- Image generation from text prompts
- Multi-agent workflow with modular components
- Retrieval-augmented generation for improved text quality
- Automatic quality scoring and optimization
- Human-readable review output
- Export functionality (text and metadata)
- Multi-language support (English and German)
- Benchmarking and evaluation framework


## System Architecture
The system follows a layered architecture:

1. User Interface Layer  
   - Streamlit-based web application  
   - Input forms and output viewer  

2. Application Layer  
   - Supervisor: Coordinates workflow execution  
   - Router: Selects appropriate sub-agent  

3. Sub-Agent Layer  
   - Text Generator Agent  
   - Image Generator Agent  

4. Shared Services  
   - Logging and metadata tracking  
   - Configuration management  
   - Scoring utilities  

5. Data Layer  
   - Generated outputs  
   - Benchmark data  
   - Prompt and run logs  


## Workflow
The execution flow of the system is:
User Input → Supervisor → Router → Sub-Agent → Optimization → Review → Export

Each sub-agent follows a structured pipeline:

### Text Pipeline
- Analyze input
- Retrieve context (optional)
- Generate content
- Optimize output
- Review quality
- Export result

### Image Pipeline
- Analyze prompt
- Enhance prompt (style retrieval)
- Generate image
- Evaluate (CLIP + aesthetic score)
- Review output
- Export result


## Text Generation Module
The text generation system includes:
- Analyzer: Cleans input, extracts keywords, builds prompts  
- Retriever: Fetches relevant context using TF-IDF  
- Generator: Calls LLM APIs (OpenAI, Groq)  
- Optimizer: Improves quality using readability, sentiment, and repetition checks  
- Reviewer: Converts scores into approval decisions  
- Exporter: Saves outputs and metadata  


## Image Generation Module
The image generation system includes:
- Analyzer: Extracts structured information from prompts  
- Style Retriever: Enhances prompts with style templates  
- Generator: Produces images using external APIs  
- Optimizer: Evaluates using CLIP and aesthetic scoring  
- Reviewer: Provides quality feedback  
- Exporter: Stores images and metadata  


## Quality Evaluation

### Text Evaluation
- Readability (Flesch score)
- Sentiment (VADER)
- Repetition detection
- Content-type-specific scoring

### Image Evaluation
- CLIP score (semantic alignment)
- Aesthetic score (visual quality)
- Technical validation

## Benchmarking
The project includes benchmarking for both text and image models.

### Text Models Evaluated
- GPT-4o  
- LLaMA 3.3 70B  
- Moonshot Kimi K2  
- Qwen 3 32B  

### Image Models Evaluated
- OpenAI Image-1 Mini  
- Stability SDXL  
- Freepik Mystic  

Key observations:
- Optimization significantly improves output quality  
- Open models can achieve competitive performance with lower latency  
- Image models show trade-offs between quality, speed, and reliability  


## Technologies Used
- Python
- Streamlit
- OpenAI API
- Groq API
- LangGraph
- LlamaIndex
- scikit-learn (TF-IDF)
- PyTorch
- OpenCLIP
- Pillow (PIL)


## How to Run
1. Clone the repository:
```bash
git clone https://github.com/Saravana0705/AI_Content_Generator.git
cd AI_Content_Generator
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate # Linux/Mac
.venv\Scripts\activate # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set environment variables:
```bash
OPENAI_API_KEY=your_key
```

5. Run the application:
```bash
streamlit run app.py
```


```markdown
## Project Structure
```
```text
AI_Content_Generator/
├── assets/              # icons and UI assets
├── benchmark/           # benchmark data, plots and runs
├── calibration/         # threshold calibration files
├── data/                # retrieval knowledge base
├── exports/             # exported outputs
├── metrics/             # benchmarking and evaluation scripts
├── models/              # model artifacts
├── runs/                # text/image run logs
├── src/
│   ├── main_agent/      # supervisor, router, main agent
│   ├── shared/          # shared logging utilities
│   ├── sub_agents/
│   │   ├── text_generator/
│   │   ├── image_generator/
│   │   └── future_video_generator/
│   └── utils/
├── tests/               # unit tests
├── app.py               # Streamlit application
├── Dockerfile
├── requirements.txt
└── README.md
```


## Future Improvements
- Add support for additional languages  
- Extend to video and audio generation  
- Improve retrieval with vector databases  
- Enhance UI and user interaction  
- Integrate deployment pipeline  


## Conclusion
This project demonstrates a scalable and modular approach to automated content generation. The combination of structured workflows, evaluation mechanisms and multiple models enables consistent and high-quality output for both text and image generation tasks.
