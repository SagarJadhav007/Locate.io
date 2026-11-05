"""
Main Entry Point & Demonstration Pipeline.
Demonstrates:
1. Seed & synthetic data generation saved to GeoJSON
2. k-Nearest Neighbors (k-NN) queries for Hospitals and Colleges
3. Radius / Range spatial queries with GeoPandas STRtree bounding-box filtering
4. Rich interactive Folium map generation with radius circles and rays
5. Comprehensive spatial indexing performance benchmark & graph generation
"""

import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
from src.dataset import generate_facilities, save_geojson
from src.service import LocationBasedService
from src.benchmark import run_scalability_benchmark
from src.config import DEFAULT_CENTER_LAT, DEFAULT_CENTER_LON


def print_banner(title: str):
    print("\n" + "=" * 78)
    print(f" {title.upper()} ")
    print("=" * 78)


def print_results_table(title: str, df: gpd.GeoDataFrame):
    print(f"\n--- {title} (Total: {len(df)}) ---")
    if len(df) == 0:
        print("  No facilities matched the criteria.")
        return

    cols = ["facility_id", "name", "category", "rating", "distance_km", "address"]
    # Filter available columns
    avail = [c for c in cols if c in df.columns]
    
    # Formatted display
    formatters = {
        "distance_km": "{:.2f} km".format,
        "rating": "{:.1f}".format
    }
    print(df[avail].to_string(index=False, formatters={k: v for k, v in formatters.items() if k in avail}))


def main():
    print_banner("Location-Based Service for Finding Nearby Facilities")
    print("[*] Initializing service with realistic metropolitan facility dataset...")

    # 1. Generate & save dataset
    geojson_path = os.path.join("data", "facilities.geojson")
    gdf = generate_facilities(n_samples=1500, center_lat=DEFAULT_CENTER_LAT, center_lon=DEFAULT_CENTER_LON)
    save_geojson(gdf, geojson_path)
    print(f"[OK] Created dataset with {len(gdf)} facilities saved to {geojson_path}")

    service = LocationBasedService(gdf=gdf)
    stats = service.get_summary_stats()
    print("[OK] Spatial indexes built (STRtree + 3D Spherical cKDTree).")
    print(f"     Facility distribution across categories: {stats['categories']}")

    # Query coordinates: Chembur Central / Diamond Garden, Mumbai, India
    query_lat = DEFAULT_CENTER_LAT
    query_lon = DEFAULT_CENTER_LON
    print(f"\n[*] Reference Query Coordinates (Chembur, Mumbai): Latitude {query_lat:.4f}, Longitude {query_lon:.4f}")

    # Query 1: Nearest 5 Hospitals
    print_banner("1. k-NN Search: Top 5 Nearest Hospitals")
    nearest_hospitals = service.find_nearest_facilities(query_lat, query_lon, k=5, category="Hospital")
    print_results_table("Top 5 Nearest Hospitals", nearest_hospitals)

    # Query 2: Nearest 5 Colleges / Universities
    print_banner("2. k-NN Search: Top 5 Nearest Colleges / Universities")
    nearest_colleges = service.find_nearest_facilities(query_lat, query_lon, k=5, category="College")
    print_results_table("Top 5 Nearest Colleges", nearest_colleges)

    # Query 3: Radius Query - Facilities within 2.0 km
    print_banner("3. Radius / Range Search: Facilities within 2.0 km")
    radius_km = 2.0
    within_radius = service.find_facilities_in_radius(query_lat, query_lon, radius_km=radius_km)
    print_results_table(f"All Facilities within {radius_km} km", within_radius.head(10))
    print(f"\n  ... and {max(0, len(within_radius) - 10)} more facilities within {radius_km} km.")

    # Query 4: Radius Query - Specific category: Top Restaurants within 1.5 km
    nearby_restaurants = service.find_facilities_in_radius(query_lat, query_lon, radius_km=1.5, category="Restaurant")
    print_results_table("Restaurants within 1.5 km", nearby_restaurants.head(8))

    # 5. Interactive Folium Map Generation
    print_banner("4. Interactive Map Generation (Folium)")
    map_output_path = os.path.join("output", "nearby_facilities_map.html")
    top_nearest = service.find_nearest_facilities(query_lat, query_lon, k=6)
    saved_map_path = service.render_map(
        output_html_path=map_output_path,
        query_lat=query_lat,
        query_lon=query_lon,
        nearest_df=top_nearest,
        radius_df=within_radius.head(30),
        radius_km=radius_km,
        include_all=True
    )
    print(f"[OK] Interactive HTML map successfully generated:")
    print(f"     -> {saved_map_path}")

    # 6. Spatial Indexing Performance Benchmark
    print_banner("5. Spatial Indexing & Query Performance Benchmark")
    print("[*] Running scalability benchmark across varying facility counts (N = 500 to 25,000)...")
    benchmark_df = run_scalability_benchmark(
        sample_sizes=[500, 2000, 10000, 25000],
        output_chart_path=os.path.join("output", "query_performance.png")
    )
    print("\n--- Performance Benchmark Summary Table ---")
    print(benchmark_df.to_string(index=False))

    print_banner("Prototype Execution Complete")
    print("Files created:")
    print(f" - Facility GeoJSON:  {geojson_path}")
    print(f" - Interactive Map:   {map_output_path}")
    print(f" - Performance Chart: output/query_performance.png")
    print("\nTo explore the interactive web interface, run:")
    print("   python -m uvicorn app:app --port 8000")


if __name__ == "__main__":
    main()

