FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app
COPY requirements.txt ./

# CPU wheels avoid downloading CUDA dependencies for this local demo.
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 10001 appuser
RUN mkdir -p /app/data /app/uploads /app/chroma_store /app/.cache/huggingface
COPY --chown=appuser:appuser . .
RUN chown -R appuser:appuser /app

USER appuser
EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
