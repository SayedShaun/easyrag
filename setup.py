from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'RAG Python Package'

setup(
    name="rag",
    author="Sayed Shaun",
    version=VERSION,
    packages=find_packages(),
    description=DESCRIPTION,
    install_requires=[
        "langchain==0.1.17",
        "langchain_google_genai==1.0.3",
        "pydantic==2.7.1",
        "pypdf==4.2.0",
        "setuptools==68.2.2",
        "torch==2.1.1",
        "transformers==4.40.1",
        "faiss-cpu",
        "sentence-transformers",
        "bitsandbytes",
        "accelerate"
    ],
)

