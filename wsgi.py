import sys

# Add your project directory to the Python path
path = '/home/your_username/pregnancy_modal'  # Replace 'your_username' with your PythonAnywhere username
if path not in sys.path:
    sys.path.append(path)

from api import app as application 