FROM python:3.10-slim

# Install system dependencies required for OpenCV and PyTorch
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency requirements
COPY Backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY Backend/ /app/Backend/
COPY Models/ /app/Models/

EXPOSE 8000

ENV PYTHONPATH=/app
CMD ["python", "Backend/main.py"]
