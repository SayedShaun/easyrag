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
        "langchain",
        "pypdf",
        "torch",
        "transformers",
        "faiss-gpu",
        "sentence-transformers",
        "bitsandbytes",
        "accelerate"
    ],
)

