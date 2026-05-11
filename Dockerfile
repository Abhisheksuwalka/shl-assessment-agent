FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Pre-build FAISS index at image build time (if not already present)
RUN python scripts/build_index.py

# Expose port
EXPOSE 8000

# Health check for container orchestrators
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); assert r.status_code == 200"

# Start the server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
