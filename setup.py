from setuptools import setup, find_packages

setup(
    name="rag",
    version=0.1,
    packages=find_packages(),
    install_requires=[
        "langchain",
        "pypdf",
        "torch",
        "transformers",
        "faiss-gpu",
        "sentence-transformers",
        "bitsandbytes",
        "accelerate"
    ]
)

