FROM python:3.11-slim
WORKDIR /app

# pytesseract is a wrapper; install the OCR engine and language data too.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-hin \
        tesseract-ocr-tel \
        tesseract-ocr-tam \
        tesseract-ocr-urd \
        tesseract-ocr-ben \
        tesseract-ocr-mar \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 10000
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --timeout 180 backend.app:app"]
