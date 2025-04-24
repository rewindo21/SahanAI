FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install gdown to download from Google Drive
RUN pip install gdown

# Copy project files
COPY . .

<<<<<<< HEAD
# Download the zip file from Google Drive and extract it
RUN gdown --id 14aB1TXaUzpDqAyp_cBIgpInIRUzSW3dL -O model_save.zip \
    && unzip model_save.zip -d model_save \
    && rm model_save.zip

# Run the app
=======
RUN curl -L "https://drive.google.com/file/d/14aB1TXaUzpDqAyp_cBIgpInIRUzSW3dL/view?usp=sharing" -o model_save

>>>>>>> 21a7b1ef6770c71b396b74ee269bb01fb9d8d6b5
CMD ["uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]