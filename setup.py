# from setuptools import setup, find_packages
# import subprocess

# VERSION = subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8")
# DESCRIPTION = 'OpenRAG Python Package'

# with open("README.md", "r", encoding="utf-8") as f:
#     long_description = f.read()

# setup(
#     name="openrag",
#     author="Sayed Shaun",
#     version=VERSION,
#     packages=find_packages(),
#     description=DESCRIPTION,
#     install_requires=[
#         "langchain",
#         "langchain-community",
#         "langchain_google_genai",
#         "pypdf",
#         "torch",
#         "transformers",
#         "faiss-cpu",
#         "sentence-transformers",
#         "bitsandbytes",
#         "accelerate"
#     ],
#     long_description=long_description,
#     long_description_content_type='text/markdown',
# )



from setuptools import setup, find_packages
import subprocess

# Get version from Git describe
try:
    VERSION = subprocess.check_output(["git", "describe", "--tags", "--always"]).strip().decode("utf-8")
    if '-' in VERSION:  # Check if it's a dev version
        parts = VERSION.split('-')
        VERSION = parts[0]  # Use only the tag part
except subprocess.CalledProcessError:
    VERSION = "0.1.0"  # Fallback version if Git command fails

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
)
