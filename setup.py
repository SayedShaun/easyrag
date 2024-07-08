import subprocess
from setuptools import setup, find_packages


VERSION = 1.0.0 
DESCRIPTION = 'OpenRAG Python Package'

setup(
    name="openrag",
    author="Sayed Shaun",
    version=VERSION,
    packages=find_packages(),
    description=DESCRIPTION,
    install_requires=[
        "langchain",
        "langchain-community",
        "langchain_google_genai",
        "pypdf",
        "torch",
        "transformers",
        "faiss-cpu",
        "sentence-transformers",
        "bitsandbytes",
        "accelerate"
    ],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    python_requires=">=3.10",
)

