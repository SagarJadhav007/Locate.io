FROM python:3.11-slim

# Install system C libraries for GEOS / PROJ / Spatial extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Generate Chembur facilities seed dataset
RUN python -c "from src.dataset import generate_facilities, save_geojson; gdf = generate_facilities(n_samples=500); save_geojson(gdf, 'data/facilities.geojson')"

# Use dynamic PORT provided by cloud environments (Railway, Render, Fly.io, Heroku)
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
