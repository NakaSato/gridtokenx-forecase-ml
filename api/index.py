import os
import sys

# Add the project root to sys.path so that absolute imports work on Vercel
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

# Vercel requires the FastAPI instance to be named 'app'
# and exported from this file.
