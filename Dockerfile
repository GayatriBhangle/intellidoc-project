#use python 3.11 slim image (lightweight)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# COpy requirements file
COPY requirements.txt .

# install pyton packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Create uploads directory for file storage
RUN mkdir -p uploads

# Set Flask app
ENV FLASK_APP=app/main.py

# Expose port 10000 (Render uses dynamic ports)
EXPOSE 10000

# Run the flask app
CMD["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=10000"]