# RULE: can be change according to use cases
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.base.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libx11-6 libxcb1 libgl1 libgl1-mesa-dri libglx0 libglu1-mesa \
  && rm -rf /var/lib/apt/lists/* \
  && pip install --no-cache-dir -r requirements.base.txt

COPY . .
RUN mkdir -p compiler_generated && chmod +x run-grpc-compiler.sh && ./run-grpc-compiler.sh

EXPOSE 50051
CMD ["python", "host.py"]
