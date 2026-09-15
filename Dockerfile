FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libegl1 \ 
    libgles2 \ 
    libsm6 \ 
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python entrenar_final.py
RUN python entrenar_identificador.py

EXPOSE 5000

CMD ["python", "app.py"]