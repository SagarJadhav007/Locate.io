"""
FastAPI Web Application & REST API for Location-Based Service.
Serves:
- REST API endpoints for k-NN and Radius searches
- Interactive HTML frontend allowing live queries, pin clicks, and layer toggles
- Dynamic Folium map generation
"""

import os
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import geopandas as gpd
from typing import Optional

from src.service import LocationBasedService
from src.dataset import generate_facilities, load_geojson, save_geojson
from src.config import DEFAULT_CENTER_LAT, DEFAULT_CENTER_LON, CATEGORY_CONFIG

app = FastAPI(
    title="Location-Based Facility Discovery Service",
    description="Spatial Indexing & Nearest Neighbor Service using GeoPandas, SciPy, and Folium",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize service singleton
GEOJSON_FILE = os.path.join("data", "facilities.geojson")
if not os.path.exists(GEOJSON_FILE):
    gdf = generate_facilities(n_samples=1500)
    save_geojson(gdf, GEOJSON_FILE)
else:
    gdf = load_geojson(GEOJSON_FILE)

service = LocationBasedService(gdf=gdf)


@app.get("/api/stats")
def get_stats():
    """Returns dataset summary statistics."""
    return service.get_summary_stats()


@app.get("/api/categories")
def get_categories():
    """Returns supported facility categories and visual configurations."""
    return CATEGORY_CONFIG


@app.get("/api/search/knn")
def search_knn(
    lat: float = Query(DEFAULT_CENTER_LAT, description="Query Latitude"),
    lon: float = Query(DEFAULT_CENTER_LON, description="Query Longitude"),
    k: int = Query(5, ge=1, le=50, description="Number of neighbors"),
    category: Optional[str] = Query(None, description="Filter category"),
    method: str = Query("ckdtree", pattern="^(ckdtree|strtree|bruteforce)$")
):
    """Finds the k nearest facilities to coordinates."""
    results = service.find_nearest_facilities(lat=lat, lon=lon, k=k, category=category, method=method)
    
    # Format output as clean JSON
    data = []
    for _, row in results.iterrows():
        data.append({
            "facility_id": row.get("facility_id"),
            "name": row.get("name"),
            "category": row.get("category"),
            "rating": row.get("rating"),
            "distance_km": row.get("distance_km"),
            "distance_m": row.get("distance_m"),
            "address": row.get("address"),
            "lat": row.get("lat"),
            "lon": row.get("lon")
        })
    return {
        "query": {"lat": lat, "lon": lon, "k": k, "category": category, "method": method},
        "count": len(data),
        "results": data
    }


@app.get("/api/search/radius")
def search_radius(
    lat: float = Query(DEFAULT_CENTER_LAT, description="Query Latitude"),
    lon: float = Query(DEFAULT_CENTER_LON, description="Query Longitude"),
    radius_km: float = Query(3.0, ge=0.1, le=50.0, description="Radius in kilometers"),
    category: Optional[str] = Query(None, description="Filter category"),
    method: str = Query("strtree", pattern="^(strtree|ckdtree|bruteforce)$")
):
    """Finds all facilities within radius_km."""
    results = service.find_facilities_in_radius(lat=lat, lon=lon, radius_km=radius_km, category=category, method=method)
    data = []
    for _, row in results.iterrows():
        data.append({
            "facility_id": row.get("facility_id"),
            "name": row.get("name"),
            "category": row.get("category"),
            "rating": row.get("rating"),
            "distance_km": row.get("distance_km"),
            "distance_m": row.get("distance_m"),
            "address": row.get("address"),
            "lat": row.get("lat"),
            "lon": row.get("lon")
        })
    return {
        "query": {"lat": lat, "lon": lon, "radius_km": radius_km, "category": category, "method": method},
        "count": len(data),
        "results": data
    }


@app.get("/map/interactive", response_class=HTMLResponse)
def get_dynamic_map(
    lat: float = DEFAULT_CENTER_LAT,
    lon: float = DEFAULT_CENTER_LON,
    k: int = 5,
    radius_km: float = 2.5,
    category: Optional[str] = None
):
    """Generates on-the-fly Folium map for the given query parameters."""
    nearest = service.find_nearest_facilities(lat, lon, k=k, category=category)
    within_rad = service.find_facilities_in_radius(lat, lon, radius_km=radius_km, category=category)
    
    tmp_path = os.path.join("output", "dynamic_query_map.html")
    service.render_map(
        output_html_path=tmp_path,
        query_lat=lat,
        query_lon=lon,
        nearest_df=nearest,
        radius_df=within_rad,
        radius_km=radius_km,
        include_all=True
    )
    with open(tmp_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/", response_class=HTMLResponse)
def index_page():
    """Serves the clean, flat-design frontend dashboard."""
    template_path = os.path.join("templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    raise HTTPException(status_code=404, detail="Template not found")
