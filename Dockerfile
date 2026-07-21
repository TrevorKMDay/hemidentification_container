# Use a lightweight Python base image
FROM python:3.13-slim

# Install system dependencies if required by pandas wheel builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pandas
COPY requirements.txt ./requirements.txt
COPY *.py /app/

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1
ENTRYPOINT [ "python", "/app/fingerprinting_manager.py" ]

# CMD ["--help"]