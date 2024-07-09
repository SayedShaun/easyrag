---

# EasyRAG Python Package

EasyRAG is a Python package designed to facilitate information retrieval and generation tasks, particularly in natural language processing applications. With RAG, users can input a PDF file along with a Hugging Face model, enabling the extraction of relevant data from the PDF and responding to user queries based on the extracted information. The "EasyRAG" framework is designed for quick rag prototype and to check retrieval performance with different open source models, including Llama, Mistral, Phi, and other 1 to 10 billion parameter causal models depanding on the hardware. It also supports Googgle Gemini and OpenAi models through API call.

## Features

- **PDF Parsing**: RAG can parse PDF files to extract textual information.
- **Information Retrieval**: Using Hugging Face models, RAG retrieves relevant data from the parsed PDF.
- **Query Response**: Users can ask questions or input queries, and RAG will provide responses based on the extracted information.

## Installation

To install rag, paste the link below into your terminal and press enter.

```bash
pip install git+https://github.com/SayedShaun/easyrag-python.git
```
or 
```bash
pip install easyrag-python
```

## Usage

Using RAG is straightforward. Here's a basic example of how to use it:

```python
from easyrag import HuggingFaceModel

# Initialize and Provide a PDF file and Hugging Face model
rag = HuggingFaceModel(
    model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    hf_token="your huggingface token",
    pdf_path="sample resume.pdf"
)

# Retrieve data from the PDF
rag.retrieve_answer("what skills she has?")

# Response
"""
Donna Robbins has skills in Microsoft NAV Dynamics,
Cashflow planning & management, State & federal tax codes,
Bookkeeping, Exceptional communication, and Fluent in German.
"""
```

## Bugs
All open-source models might not be compatible at this moment. This package is specifically designed to work with causal language models often decoder-only models.

## Contributing

We welcome contributions from the community to enhance RAG's functionality, improve its performance, or fix any issues. To contribute, please follow these steps:

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
