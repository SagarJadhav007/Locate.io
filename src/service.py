"""
Location-Based Facility Service.
High-level service API connecting data loading, spatial indexing algorithms,
nearest-neighbor lookups, radius searches, and interactive map visualizations.
"""

import os
import geopandas as gpd
from typing import Optional, Dict, Any, List
from src.dataset import create_seed_geodataframe, generate_facilities, save_geojson, load_geojson
from src.spatial_engine import SpatialIndexEngine
from src.visualizer import build_facility_map
from src.config import DEFAULT_CENTER_LAT, DEFAULT_CENTER_LON


class LocationBasedService:
    """
    Facade service providing location-based facility discovery and spatial analytics.
    """

    def __init__(self, gdf: Optional[gpd.GeoDataFrame] = None, data_path: Optional[str] = None):
        """
        Initializes the service. Loads from GeoDataFrame, or from file, or creates seed dataset.
        """
        if gdf is not None:
            self.gdf = gdf
        elif data_path and os.path.exists(data_path):
            self.gdf = load_geojson(data_path)
        else:
            self.gdf = create_seed_geodataframe()

        self.engine = SpatialIndexEngine(self.gdf)

    @classmethod
    def with_synthetic_data(cls, n_samples: int = 2500, center_lat: float = DEFAULT_CENTER_LAT, center_lon: float = DEFAULT_CENTER_LON):
        """Constructs a service instance populated with n_samples synthetic facilities."""
        gdf = generate_facilities(n_samples=n_samples, center_lat=center_lat, center_lon=center_lon)
        return cls(gdf=gdf)

    def find_nearest_facilities(
        self,
        lat: float,
        lon: float,
        k: int = 5,
        category: Optional[str] = None,
        method: str = "ckdtree"
    ) -> gpd.GeoDataFrame:
        """
        Finds the k closest facilities from the given coordinates.
        Methods:
        - 'ckdtree': 3D Cartesian spherical k-d tree (O(k log N), recommended)
        - 'strtree': GeoPandas / Shapely STRtree R-tree
        - 'bruteforce': O(N) linear scan
        """
        if method == "ckdtree":
            return self.engine.knn_ckdtree(lat, lon, k=k, category=category)
        elif method == "strtree":
            return self.engine.knn_strtree(lat, lon, k=k, category=category)
        elif method == "bruteforce":
            return self.engine.knn_bruteforce(lat, lon, k=k, category=category)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'ckdtree', 'strtree', or 'bruteforce'.")

    def find_facilities_in_radius(
        self,
        lat: float,
        lon: float,
        radius_km: float = 3.0,
        category: Optional[str] = None,
        method: str = "strtree"
    ) -> gpd.GeoDataFrame:
        """
        Finds all facilities within radius_km from the given coordinates.
        Methods:
        - 'strtree': GeoPandas STRtree bounding-box filter + exact Haversine refine
        - 'ckdtree': 3D Cartesian spherical ball point query
        - 'bruteforce': Linear scan
        """
        if method == "strtree":
            return self.engine.radius_strtree(lat, lon, radius_km=radius_km, category=category)
        elif method == "ckdtree":
            return self.engine.radius_ckdtree(lat, lon, radius_km=radius_km, category=category)
        elif method == "bruteforce":
            return self.engine.radius_bruteforce(lat, lon, radius_km=radius_km, category=category)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'strtree', 'ckdtree', or 'bruteforce'.")

    def render_map(
        self,
        output_html_path: str,
        query_lat: Optional[float] = None,
        query_lon: Optional[float] = None,
        nearest_df: Optional[gpd.GeoDataFrame] = None,
        radius_df: Optional[gpd.GeoDataFrame] = None,
        radius_km: Optional[float] = None,
        include_all: bool = True
    ) -> str:
        """
        Generates and saves an interactive Folium map visualization.
        Returns the absolute file path of the saved HTML file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)
        fmap = build_facility_map(
            all_facilities=self.gdf if include_all else None,
            query_lat=query_lat,
            query_lon=query_lon,
            nearest_facilities=nearest_df,
            radius_km=radius_km,
            radius_facilities=radius_df
        )
        fmap.save(output_html_path)
        return os.path.abspath(output_html_path)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Returns statistics on the current facility dataset."""
        cat_counts = self.gdf["category"].value_counts().to_dict()
        return {
            "total_facilities": len(self.gdf),
            "categories": cat_counts,
            "bounds": {
                "min_lat": float(self.gdf["lat"].min()),
                "max_lat": float(self.gdf["lat"].max()),
                "min_lon": float(self.gdf["lon"].min()),
                "max_lon": float(self.gdf["lon"].max())
            }
        }
