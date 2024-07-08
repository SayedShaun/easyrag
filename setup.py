from setuptools import setup, find_packages
import subprocess

VERSION = subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8")
DESCRIPTION = 'OpenRAG Python Package'

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

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
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires=">=3.10",
)

