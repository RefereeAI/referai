FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . .

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
