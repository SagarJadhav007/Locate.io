"""
Spatial Engine for Location-Based Service.
Implements:
1. Exact Vectorized Haversine distance calculation (Spherical Great-Circle)
2. 3D Cartesian coordinate projection for spherical KD-Tree / cKDTree
3. GeoPandas / Shapely STRtree (R-Tree Sort-Tile-Recursive) spatial index
4. SciPy cKDTree spatial index (Spherical 3D metric)
5. Brute-Force scan (O(N) baseline for correctness verification and benchmarking)
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box
from scipy.spatial import cKDTree
from typing import Tuple, List, Dict, Optional, Any
from src.config import EARTH_RADIUS_KM, CRS_WGS84


def haversine_vectorized(
    lat1: float,
    lon1: float,
    lat2_arr: np.ndarray,
    lon2_arr: np.ndarray,
    radius: float = EARTH_RADIUS_KM
) -> np.ndarray:
    """
    Computes exact great-circle distance in kilometers between a reference coordinate
    (lat1, lon1) and arrays of coordinates (lat2_arr, lon2_arr) using Haversine formula.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2_arr)
    delta_phi = np.radians(lat2_arr - lat1)
    delta_lambda = np.radians(lon2_arr - lon1)

    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    # Numerical clip to prevent NaN from tiny floating point inaccuracies
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return radius * c


def latlon_to_cartesian_3d(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    radius: float = EARTH_RADIUS_KM
) -> np.ndarray:
    """
    Converts spherical latitude/longitude coordinates (degrees) to 3D Cartesian coordinates (X, Y, Z).
    Enables Euclidean spatial indexing (KD-Tree / cKDTree) on the sphere without distortion.
    """
    phi = np.radians(latitudes)
    lam = np.radians(longitudes)
    x = radius * np.cos(phi) * np.cos(lam)
    y = radius * np.cos(phi) * np.sin(lam)
    z = radius * np.sin(phi)
    return np.column_stack((x, y, z))


def great_circle_to_chord_distance(dist_km: float, radius: float = EARTH_RADIUS_KM) -> float:
    """Converts a surface great-circle distance into a 3D Cartesian chord distance."""
    theta = dist_km / radius
    return 2.0 * radius * np.sin(theta / 2.0)


class SpatialIndexEngine:
    """
    Unified Spatial Indexing Engine wrapping GeoPandas STRtree,
    SciPy 3D cKDTree, and Brute-force scanning.
    """

    def __init__(self, gdf: gpd.GeoDataFrame):
        """Initializes and builds spatial indexes over the given GeoDataFrame."""
        if gdf.crs is None:
            gdf = gdf.set_crs(CRS_WGS84)
        elif gdf.crs.to_string() != CRS_WGS84:
            gdf = gdf.to_crs(CRS_WGS84)

        self.gdf = gdf.copy().reset_index(drop=True)
        self.size = len(self.gdf)
        
        # Pre-extract coordinate arrays for fast vectorized operations
        self.lat_arr = self.gdf["lat"].to_numpy(dtype=np.float64)
        self.lon_arr = self.gdf["lon"].to_numpy(dtype=np.float64)

        # 1. GeoPandas / Shapely STRtree R-Tree index
        # Accessing .sindex triggers building the STRtree
        self.sindex = self.gdf.sindex

        # 2. 3D Cartesian cKDTree index
        self.cartesian_coords = latlon_to_cartesian_3d(self.lat_arr, self.lon_arr)
        self.ckdtree = cKDTree(self.cartesian_coords)

    # ==========================================
    # 1. NEAREST-NEIGHBOR (k-NN) SEARCH
    # ==========================================

    def knn_ckdtree(
        self,
        query_lat: float,
        query_lon: float,
        k: int = 5,
        category: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        k-Nearest Neighbors search using SciPy cKDTree in 3D Cartesian space.
        Time complexity: O(k * log N) average.
        """
        if category:
            mask = (self.gdf["category"] == category).to_numpy()
            subset_indices = np.where(mask)[0]
            if len(subset_indices) == 0:
                return self.gdf.iloc[0:0].copy()
            
            # Query against subset or build temporary sub-tree if large
            sub_coords = self.cartesian_coords[subset_indices]
            sub_tree = cKDTree(sub_coords)
            q_3d = latlon_to_cartesian_3d(np.array([query_lat]), np.array([query_lon]))[0]
            
            k_actual = min(k, len(subset_indices))
            dists_chord, local_indices = sub_tree.query(q_3d, k=k_actual)
            if np.isscalar(local_indices):
                local_indices = [local_indices]
            
            matched_indices = subset_indices[local_indices]
        else:
            q_3d = latlon_to_cartesian_3d(np.array([query_lat]), np.array([query_lon]))[0]
            k_actual = min(k, self.size)
            dists_chord, matched_indices = self.ckdtree.query(q_3d, k=k_actual)
            if np.isscalar(matched_indices):
                matched_indices = [matched_indices]

        # Calculate exact Haversine distance
        results = self.gdf.iloc[matched_indices].copy()
        exact_distances = haversine_vectorized(
            query_lat, query_lon,
            results["lat"].to_numpy(),
            results["lon"].to_numpy()
        )
        results["distance_km"] = np.round(exact_distances, 3)
        results["distance_m"] = np.round(exact_distances * 1000.0, 1)
        results = results.sort_values(by="distance_km").reset_index(drop=True)
        return results

    def knn_strtree(
        self,
        query_lat: float,
        query_lon: float,
        k: int = 5,
        category: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        k-Nearest Neighbors using GeoPandas / Shapely STRtree R-Tree
        via expanding bounding-box spatial index search and exact refinement.
        Time complexity: O(k + log N) average.
        """
        if category:
            mask = self.gdf["category"] == category
            active_gdf = self.gdf[mask].reset_index(drop=True)
            if len(active_gdf) == 0:
                return self.gdf.iloc[0:0].copy()
            sindex = active_gdf.sindex
        else:
            active_gdf = self.gdf
            sindex = self.sindex

        k_actual = min(k, len(active_gdf))
        query_point = Point(query_lon, query_lat)

        # Expanding bounding-box search around query point
        # Initial radius ~ 1.0 km, expands until at least k candidates are found
        radius_km = 1.0
        while True:
            lat_delta = radius_km / 111.0
            lon_delta = radius_km / (111.0 * max(0.1, np.cos(np.radians(query_lat))))
            search_box = box(
                query_lon - lon_delta,
                query_lat - lat_delta,
                query_lon + lon_delta,
                query_lat + lat_delta
            )
            candidate_indices = sindex.query(search_box, predicate="intersects")
            
            # If candidate_indices is 2D, take the tree indices (row 1)
            if isinstance(candidate_indices, np.ndarray) and candidate_indices.ndim > 1:
                candidate_indices = candidate_indices[1]
                
            if len(candidate_indices) >= k_actual or radius_km >= 100.0:
                break
            radius_km *= 2.0

        # If candidates are still fewer than k (e.g. edge of world), fallback to all
        if len(candidate_indices) == 0:
            candidate_indices = np.arange(len(active_gdf))

        candidates = active_gdf.iloc[candidate_indices].copy()
        exact_distances = haversine_vectorized(
            query_lat, query_lon,
            candidates["lat"].to_numpy(),
            candidates["lon"].to_numpy()
        )
        candidates["distance_km"] = np.round(exact_distances, 3)
        candidates["distance_m"] = np.round(exact_distances * 1000.0, 1)
        results = candidates.sort_values(by="distance_km").head(k_actual).reset_index(drop=True)
        return results

    def knn_bruteforce(
        self,
        query_lat: float,
        query_lon: float,
        k: int = 5,
        category: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        k-Nearest Neighbors via Brute-Force scan.
        Computes Haversine distance across all N elements in O(N) time.
        """
        if category:
            mask = (self.gdf["category"] == category).to_numpy()
            target_gdf = self.gdf[mask].copy()
        else:
            target_gdf = self.gdf.copy()

        if len(target_gdf) == 0:
            return self.gdf.iloc[0:0].copy()

        dists = haversine_vectorized(
            query_lat, query_lon,
            target_gdf["lat"].to_numpy(),
            target_gdf["lon"].to_numpy()
        )
        target_gdf["distance_km"] = np.round(dists, 3)
        target_gdf["distance_m"] = np.round(dists * 1000.0, 1)
        return target_gdf.sort_values(by="distance_km").head(k).reset_index(drop=True)

    # ==========================================
    # 2. RADIUS / RANGE SEARCH
    # ==========================================

    def radius_strtree(
        self,
        query_lat: float,
        query_lon: float,
        radius_km: float = 3.0,
        category: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Radius / Range search using GeoPandas STRtree bounding-box filtering + Haversine refinement.
        Two-stage filter & refine spatial paradigm:
        1. Primary filter: Calculate bounding box in degrees for radius_km and query R-Tree.
        2. Secondary refine: Compute exact Haversine distance on candidates and filter <= radius_km.
        """
        # Bounding box in degrees
        lat_delta = radius_km / 111.0
        lon_delta = radius_km / (111.0 * np.cos(np.radians(query_lat)))
        
        search_box = box(
            query_lon - lon_delta,
            query_lat - lat_delta,
            query_lon + lon_delta,
            query_lat + lat_delta
        )

        candidate_indices = self.sindex.query(search_box, predicate="intersects")
        if isinstance(candidate_indices, np.ndarray) and candidate_indices.ndim > 1:
            candidate_indices = candidate_indices[1]

        if len(candidate_indices) == 0:
            return self.gdf.iloc[0:0].copy()

        candidates = self.gdf.iloc[candidate_indices].copy()

        if category:
            candidates = candidates[candidates["category"] == category]
            if len(candidates) == 0:
                return self.gdf.iloc[0:0].copy()

        # Refine step: calculate exact Haversine distances
        dists = haversine_vectorized(
            query_lat, query_lon,
            candidates["lat"].to_numpy(),
            candidates["lon"].to_numpy()
        )
        candidates["distance_km"] = np.round(dists, 3)
        candidates["distance_m"] = np.round(dists * 1000.0, 1)

        # Retain only those strictly within the circle
        filtered = candidates[candidates["distance_km"] <= radius_km]
        return filtered.sort_values(by="distance_km").reset_index(drop=True)

    def radius_ckdtree(
        self,
        query_lat: float,
        query_lon: float,
        radius_km: float = 3.0,
        category: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Radius / Range search using 3D Cartesian cKDTree ball query.
        Converts spherical radius_km to 3D chord distance.
        """
        chord_r = great_circle_to_chord_distance(radius_km)
        q_3d = latlon_to_cartesian_3d(np.array([query_lat]), np.array([query_lon]))[0]

        matched_indices = self.ckdtree.query_ball_point(q_3d, r=chord_r)
        if len(matched_indices) == 0:
            return self.gdf.iloc[0:0].copy()

        results = self.gdf.iloc[matched_indices].copy()
        if category:
            results = results[results["category"] == category]
            if len(results) == 0:
                return self.gdf.iloc[0:0].copy()

        exact_dists = haversine_vectorized(
            query_lat, query_lon,
            results["lat"].to_numpy(),
            results["lon"].to_numpy()
        )
        results["distance_km"] = np.round(exact_dists, 3)
        results["distance_m"] = np.round(exact_dists * 1000.0, 1)
        return results.sort_values(by="distance_km").reset_index(drop=True)

    def radius_bruteforce(
        self,
        query_lat: float,
        query_lon: float,
        radius_km: float = 3.0,
        category: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Radius search via brute-force linear scan.
        """
        if category:
            mask = (self.gdf["category"] == category).to_numpy()
            target_gdf = self.gdf[mask].copy()
        else:
            target_gdf = self.gdf.copy()

        if len(target_gdf) == 0:
            return self.gdf.iloc[0:0].copy()

        dists = haversine_vectorized(
            query_lat, query_lon,
            target_gdf["lat"].to_numpy(),
            target_gdf["lon"].to_numpy()
        )
        target_gdf["distance_km"] = np.round(dists, 3)
        target_gdf["distance_m"] = np.round(dists * 1000.0, 1)
        matched = target_gdf[target_gdf["distance_km"] <= radius_km]
        return matched.sort_values(by="distance_km").reset_index(drop=True)
