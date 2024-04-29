import sys
import os

# Get the absolute path of the current file (drone.py)
current_file_path = os.path.abspath(__file__)

# Get the parent directory of the current file
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
# Append the parent directory to the Python path
sys.path.append(parent_directory)