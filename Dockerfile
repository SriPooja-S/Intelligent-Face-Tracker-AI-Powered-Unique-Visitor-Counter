FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    libgl1-mesa-glx wget curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# We must install CPU-only versions of onnx and insightface for Hugging Face free tier.
# We also explicitly install the requirements list.
RUN pip install --no-cache-dir -r requirements.txt
RUN pip uninstall -y onnxruntime-gpu
RUN pip install --no-cache-dir onnxruntime

# Copy all project files
COPY . .

# Ensure log directories exist (even though they are in the repo, this is safe)
RUN mkdir -p logs/entries logs/exits logs/registered

# Hugging Face Spaces exposes port 7860
EXPOSE 7860

# Run the dashboard-only mode
CMD ["python", "app_hf.py"]