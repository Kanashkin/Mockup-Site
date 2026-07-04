FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pillow scipy opencv-python-headless numpy python-multipart

COPY . .

CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}
