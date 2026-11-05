"""
Unit and Integration Tests for Location-Based Facility Service.
Tests:
- Haversine geodesic distance accuracy
- 3D Cartesian coordinates conversion
- Seed & synthetic dataset validation
- k-NN query equivalence (cKDTree vs STRtree vs Brute-force)
- Radius search equivalence across spatial engines
- Category filtering
- Folium HTML map rendering
"""

import os
import unittest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from src.config import EARTH_RADIUS_KM, CRS_WGS84, DEFAULT_CENTER_LAT, DEFAULT_CENTER_LON
from src.dataset import create_seed_geodataframe, generate_facilities
from src.spatial_engine import (
    haversine_vectorized,
    latlon_to_cartesian_3d,
    great_circle_to_chord_distance,
    SpatialIndexEngine
)
from src.service import LocationBasedService


class TestLocationBasedService(unittest.TestCase):

    def setUp(self):
        self.gdf = generate_facilities(n_samples=200, seed=123)
        self.service = LocationBasedService(gdf=self.gdf)
        self.query_lat = DEFAULT_CENTER_LAT
        self.query_lon = DEFAULT_CENTER_LON

    def test_haversine_accuracy(self):
        # San Francisco (37.7749, -122.4194) to Los Angeles (34.0522, -118.2437)
        # Expected distance is ~559 km
        sf_lat, sf_lon = 37.7749, -122.4194
        la_lat, la_lon = 34.0522, -118.2437
        dist = haversine_vectorized(sf_lat, sf_lon, np.array([la_lat]), np.array([la_lon]))[0]
        self.assertAlmostEqual(dist, 559.0, delta=5.0)

        # Distance from a point to itself must be 0
        dist_zero = haversine_vectorized(sf_lat, sf_lon, np.array([sf_lat]), np.array([sf_lon]))[0]
        self.assertAlmostEqual(dist_zero, 0.0, places=6)

    def test_cartesian_3d_properties(self):
        coords_3d = latlon_to_cartesian_3d(np.array([0.0, 90.0]), np.array([0.0, 0.0]))
        # Point on equator (0, 0) should be at (R, 0, 0)
        np.testing.assert_allclose(coords_3d[0], [EARTH_RADIUS_KM, 0.0, 0.0], atol=1e-3)
        # Point on North Pole (90, 0) should be at (0, 0, R)
        np.testing.assert_allclose(coords_3d[1], [0.0, 0.0, EARTH_RADIUS_KM], atol=1e-3)

    def test_dataset_generation(self):
        gdf = generate_facilities(n_samples=100, seed=42)
        self.assertEqual(len(gdf), 100)
        self.assertEqual(gdf.crs.to_string(), CRS_WGS84)
        self.assertIn("geometry", gdf.columns)
        self.assertIn("category", gdf.columns)
        self.assertIn("rating", gdf.columns)
        self.assertTrue(all(isinstance(geom, Point) for geom in gdf.geometry))

    def test_knn_equivalence(self):
        k = 5
        res_ckd = self.service.find_nearest_facilities(self.query_lat, self.query_lon, k=k, method="ckdtree")
        res_bf = self.service.find_nearest_facilities(self.query_lat, self.query_lon, k=k, method="bruteforce")

        self.assertEqual(len(res_ckd), k)
        self.assertEqual(len(res_bf), k)
        
        # Verify distances match identically between cKDTree and Brute-force
        np.testing.assert_allclose(res_ckd["distance_km"], res_bf["distance_km"], atol=1e-3)
        # Verify the 1st nearest neighbor is identical
        self.assertEqual(res_ckd.iloc[0]["facility_id"], res_bf.iloc[0]["facility_id"])

    def test_radius_search_equivalence(self):
        radius_km = 3.0
        res_str = self.service.find_facilities_in_radius(self.query_lat, self.query_lon, radius_km=radius_km, method="strtree")
        res_ckd = self.service.find_facilities_in_radius(self.query_lat, self.query_lon, radius_km=radius_km, method="ckdtree")
        res_bf = self.service.find_facilities_in_radius(self.query_lat, self.query_lon, radius_km=radius_km, method="bruteforce")

        ids_str = set(res_str["facility_id"])
        ids_ckd = set(res_ckd["facility_id"])
        ids_bf = set(res_bf["facility_id"])

        self.assertEqual(ids_str, ids_bf)
        self.assertEqual(ids_ckd, ids_bf)
        self.assertTrue(all(d <= radius_km for d in res_str["distance_km"]))

    def test_category_filtering(self):
        hospitals = self.service.find_nearest_facilities(
            self.query_lat, self.query_lon, k=5, category="Hospital"
        )
        self.assertTrue(all(cat == "Hospital" for cat in hospitals["category"]))

    def test_folium_map_rendering(self):
        output_file = os.path.join("output", "test_map.html")
        nearest = self.service.find_nearest_facilities(self.query_lat, self.query_lon, k=3)
        saved_path = self.service.render_map(
            output_html_path=output_file,
            query_lat=self.query_lat,
            query_lon=self.query_lon,
            nearest_df=nearest,
            radius_km=2.0
        )
        self.assertTrue(os.path.exists(saved_path))
        with open(saved_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("<!DOCTYPE html>", content)
        self.assertIn("folium", content.lower())
        # Clean up test output
        if os.path.exists(output_file):
            os.remove(output_file)


if __name__ == "__main__":
    unittest.main()
