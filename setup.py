from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'RAG Python Package'

setup(
    name="easy-rag",
    author="Sayed Shaun",
    version=VERSION,
    packages=find_packages(),
    description=DESCRIPTION,
    install_requires=[
        "langchain==0.1.17",
        "langchain_google_genai==1.0.3",
        "pypdf",
        "torch",
        "transformers",
        "faiss-cpu",
        "sentence-transformers",
        "bitsandbytes",
        "accelerate"
    ],
)

