FROM python:3.11-slim

# Установка системных зависимостей (для torch, transformers и milvus)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /indexing

# Скопировать файлы приложения
COPY main.py .  
COPY requirements.txt .

# Установить Python-зависимости
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8500

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8500"]
