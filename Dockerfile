FROM python:3.10-slim

WORKDIR /app


# Install unzip and any other required OS packages
RUN apt-get update && apt-get install -y unzip

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install gdown to download from Google Drive
RUN pip install gdown

# Copy project files
COPY . .

# Download the zip file from Google Drive and extract it
# RUN gdown --id 14aB1TXaUzpDqAyp_cBIgpInIRUzSW3dL -O model_save.zip \
#     && unzip model_save.zip -d . \
#     && rm model_save.zip

RUN curl -L "https://drive.google.com/uc?export=download&id=14aB1TXaUzpDqAyp_cBIgpInIRUzSW3dL" \
    -o model_save.zip && \
    unzip model_save.zip -d model_save && \
    mv model_save/model_save/* model_save/ && \
    rm model_save.zip

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]