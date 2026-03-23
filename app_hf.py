"""
app_hf.py
---------
Hugging Face Spaces entry point.

This runs a DASHBOARD-ONLY version of the Face Tracker on HF Spaces.
It shows the web UI, sample output data, and explains the project.

Full video/RTSP processing requires GPU + InsightFace which is not
available on the free HF CPU tier. Use Google Colab for live processing.

HF Spaces uses port 7860.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import Flask app from main
from main import flask_app, _init_globals, _DB_PATH

# Initialise with default config
try:
    _init_globals(config_path="config/config.json", port=7860)
except Exception as e:
    print(f"Config init warning: {e}")

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Face Tracker — Hugging Face Spaces Demo")
    print("  Dashboard running on port 7860")
    print("="*55 + "\n")
    flask_app.run(host="0.0.0.0", port=7860, debug=False)
