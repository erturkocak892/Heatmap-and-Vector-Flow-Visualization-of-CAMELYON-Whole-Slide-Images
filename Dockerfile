FROM continuumio/miniconda3:latest

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies for openslide and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    openslide-tools \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# First, install the main FastAPI application dependencies into the base Python environment
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now, we need to set up the 'hovernet' conda environment for the inference subprocess
# We expect the user to have already cloned hover_net locally before building this Dockerfile
COPY hover_net/environment.yml ./hover_net/
COPY hover_net/requirements.txt ./hover_net/
RUN conda env create -f hover_net/environment.yml

# Copy the rest of the application
COPY server/ ./server/
COPY web/ ./web/
COPY hover_net/ ./hover_net/

EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
