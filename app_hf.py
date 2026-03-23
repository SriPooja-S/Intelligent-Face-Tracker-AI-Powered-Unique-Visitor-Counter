"""
app_hf.py
---------
Hugging Face Spaces entry point.
This runs a DASHBOARD-ONLY version of the Face Tracker on HF Spaces.
"""

import sys, os

# Add src to path so main.py can find its modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# MUST initialize globals BEFORE importing the flask app from main
from main import _init_globals
try:
    # Initialize the config and DB paths pointing to the local files you uploaded
    _init_globals(config_path="config/config.json", port=7860)
except Exception as e:
    print(f"Config init warning: {e}")

# Now it is safe to import the flask app
from main import flask_app

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Face Tracker — Hugging Face Spaces Demo")
    print("  Dashboard running on port 7860")
    print("="*55 + "\n")
    
    # Hugging Face requires host="0.0.0.0" and port=7860
    flask_app.run(host="0.0.0.0", port=7860, debug=False)