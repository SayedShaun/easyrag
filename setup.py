import subprocess
from setuptools import setup, find_packages


def get_version_from_git():
    version = subprocess.check_output(['git', 'describe', '--tags']).strip().decode('utf-8')
    if version.startswith('v'):
        version = version[1:]  # Remove the 'v' prefix if it exists
    return version


VERSION = get_version_from_git() 
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

