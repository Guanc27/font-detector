FROM python:3.11-slim

# System deps for OpenCV headless
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download EasyOCR CRAFT model so it's baked into the image
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

# Copy application code
COPY api_server.py preprocessing.py ./

# Copy model and vector database
COPY models/best_model2.pt models/best_model2.pt
COPY vector_db/ vector_db/

# Default env vars (can be overridden at deploy time)
ENV CHECKPOINT_PATH=models/best_model2.pt
ENV VECTOR_DB_DIR=vector_db
ENV MODEL_NAME=ViT-B-32
ENV PRETRAINED=openai

EXPOSE 7860

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "7860"]
