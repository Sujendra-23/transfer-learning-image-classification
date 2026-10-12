# Prediction service for one exported model. The model file is not in the image (it is not in the
# repo either): mount it and point MODEL_PATH at it.
#
#   docker build -t tl-classifier .
#   docker run --rm -p 8000:8000 -v /path/to/models:/models:ro \
#     -e MODEL_PATH=/models/best_model_resnet50.keras -e BACKBONE=ResNet50 tl-classifier
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Same TensorFlow/Keras pair the models were exported with (tensorflow-metal is Apple-only and omitted).
COPY serving/requirements.txt serving/requirements.txt
RUN pip install tensorflow==2.16.2 keras==3.15.0 "numpy<2" -r serving/requirements.txt

COPY scripts scripts
COPY serving serving
RUN useradd --create-home --uid 10001 appuser
USER appuser

ENV MODEL_PATH=/models/model.keras \
    BACKBONE=ResNet50

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

# One process: the model lives in this process's memory and predictions are serialized, so extra
# workers in one container only multiply memory.
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
