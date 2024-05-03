import os

# Check if the current directory contains a directory named "pyrag"
if os.path.isdir("pyrag"):
    # If it does, check if there is a file named "rag.py" within the "pyrag" directory
    if os.path.isfile(os.path.join("pyrag", "rag.py")):
        print("Files are aligned correctly: 'rag.py' is inside the 'pyrag' directory.")
    else:
        print("Error: 'rag.py' file not found inside the 'pyrag' directory.")
else:
    print("Error: Directory named 'pyrag' not found.")
