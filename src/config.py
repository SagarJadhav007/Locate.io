"""
Configuration module for Location-Based Facility Service.
Defines constants, category styling for Folium visualization,
CRS projections, and coordinate defaults.
"""

from typing import Dict, Any

# Earth radius in kilometers (mean spherical radius)
EARTH_RADIUS_KM = 6371.0088

# Coordinate Reference Systems
CRS_WGS84 = "EPSG:4326"        # Lat/Lon degrees
CRS_PROJECTED = "EPSG:3857"    # Web Mercator in meters (for metric buffers)

# Category styling configuration for Folium interactive maps
CATEGORY_CONFIG: Dict[str, Dict[str, Any]] = {
    "Hospital": {
        "color": "red",
        "icon": "hospital-o",
        "prefix": "fa",
        "description": "Medical Center / Emergency Hospital"
    },
    "College": {
        "color": "blue",
        "icon": "graduation-cap",
        "prefix": "fa",
        "description": "Colleges & Higher Education Universities"
    },
    "Restaurant": {
        "color": "orange",
        "icon": "cutlery",
        "prefix": "fa",
        "description": "Dining, Bistros & Cafes"
    },
    "Pharmacy": {
        "color": "green",
        "icon": "medkit",
        "prefix": "fa",
        "description": "Pharmacies & Drugstores"
    },
    "Fire Station": {
        "color": "darkred",
        "icon": "fire-extinguisher",
        "prefix": "fa",
        "description": "Fire & Rescue Stations"
    },
    "Police Station": {
        "color": "darkblue",
        "icon": "shield",
        "prefix": "fa",
        "description": "Law Enforcement & Police Stations"
    },
    "Supermarket": {
        "color": "purple",
        "icon": "shopping-cart",
        "prefix": "fa",
        "description": "Groceries & Supermarkets"
    },
    "Library": {
        "color": "cadetblue",
        "icon": "book",
        "prefix": "fa",
        "description": "Public & Research Libraries"
    }
}

# Default query center (Chembur, Mumbai, Maharashtra, India)
DEFAULT_CENTER_LAT = 19.0596
DEFAULT_CENTER_LON = 72.8997
DEFAULT_BBOX = {
    "min_lat": 18.98,
    "max_lat": 19.14,
    "min_lon": 72.82,
    "max_lon": 72.98
}
