FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    libgl1-mesa-glx wget curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install (CPU-only — no GPU on HF free tier)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    flask \
    opencv-python-headless \
    Pillow \
    && echo "Base deps installed"

# Copy all project files
COPY . .

# Create log directories
RUN mkdir -p logs/entries logs/exits logs/registered

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# Run the dashboard-only mode (no video processing on HF — just shows the UI)
CMD ["python", "app_hf.py"]
