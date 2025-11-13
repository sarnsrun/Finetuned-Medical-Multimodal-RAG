# Use Python 3.10 slim image to keep it relatively small
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for building wheels (like for Levenshtein/Numpy)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create directories for saving feedback/images if they don't exist
RUN mkdir -p roco-dataset-master/data/test/radiology/images

# Expose the port Flask runs on
EXPOSE 5000

# Define the command to run the app using Gunicorn (Production Server)
# Timeout is set high (120s) because your model inference might be slow
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "300", "app:app"]