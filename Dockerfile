# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN curl -L "https://drive.google.com/file/d/14aB1TXaUzpDqAyp_cBIgpInIRUzSW3dL/view?usp=sharing" -o model_save

CMD ["uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]