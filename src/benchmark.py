"""
Performance Benchmarking Suite for Spatial Indexing.
Compares query latency, throughput, and speedup across:
1. Brute-Force Linear Scan (O(N))
2. GeoPandas / Shapely STRtree R-Tree (O(log N))
3. SciPy 3D cKDTree (O(log N))
Generates performance metrics tables and a matplotlib visualization chart.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from src.dataset import generate_facilities
from src.spatial_engine import SpatialIndexEngine
from src.config import DEFAULT_CENTER_LAT, DEFAULT_CENTER_LON


def benchmark_knn(
    engine: SpatialIndexEngine,
    query_lat: float,
    query_lon: float,
    k: int = 5,
    iterations: int = 50
) -> Dict[str, float]:
    """Measures average execution time in milliseconds for k-NN queries across methods."""
    # 1. cKDTree
    times_ckd = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = engine.knn_ckdtree(query_lat, query_lon, k=k)
        times_ckd.append(time.perf_counter() - t0)

    # 2. GeoPandas STRtree
    times_str = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = engine.knn_strtree(query_lat, query_lon, k=k)
        times_str.append(time.perf_counter() - t0)

    # 3. Brute Force
    # Adjust iterations if dataset is very large
    bf_iter = min(iterations, 20)
    times_bf = []
    for _ in range(bf_iter):
        t0 = time.perf_counter()
        _ = engine.knn_bruteforce(query_lat, query_lon, k=k)
        times_bf.append(time.perf_counter() - t0)

    return {
        "ckdtree_ms": float(np.mean(times_ckd) * 1000.0),
        "strtree_ms": float(np.mean(times_str) * 1000.0),
        "bruteforce_ms": float(np.mean(times_bf) * 1000.0),
        "speedup_ckdtree": float(np.mean(times_bf) / np.mean(times_ckd)),
        "speedup_strtree": float(np.mean(times_bf) / np.mean(times_str))
    }


def benchmark_radius(
    engine: SpatialIndexEngine,
    query_lat: float,
    query_lon: float,
    radius_km: float = 3.0,
    iterations: int = 50
) -> Dict[str, float]:
    """Measures execution time for radius queries."""
    # 1. cKDTree
    times_ckd = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = engine.radius_ckdtree(query_lat, query_lon, radius_km=radius_km)
        times_ckd.append(time.perf_counter() - t0)

    # 2. STRtree
    times_str = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = engine.radius_strtree(query_lat, query_lon, radius_km=radius_km)
        times_str.append(time.perf_counter() - t0)

    # 3. Brute-force
    bf_iter = min(iterations, 20)
    times_bf = []
    for _ in range(bf_iter):
        t0 = time.perf_counter()
        _ = engine.radius_bruteforce(query_lat, query_lon, radius_km=radius_km)
        times_bf.append(time.perf_counter() - t0)

    return {
        "ckdtree_ms": float(np.mean(times_ckd) * 1000.0),
        "strtree_ms": float(np.mean(times_str) * 1000.0),
        "bruteforce_ms": float(np.mean(times_bf) * 1000.0),
        "speedup_ckdtree": float(np.mean(times_bf) / np.mean(times_ckd)),
        "speedup_strtree": float(np.mean(times_bf) / np.mean(times_str))
    }


def run_scalability_benchmark(
    sample_sizes: List[int] = [500, 2000, 10000, 30000],
    output_chart_path: str = "output/query_performance.png"
) -> pd.DataFrame:
    """
    Evaluates performance across multiple dataset sizes and produces a plot.
    """
    records = []
    query_lat = DEFAULT_CENTER_LAT
    query_lon = DEFAULT_CENTER_LON

    for n in sample_sizes:
        print(f"[*] Benchmarking dataset size N = {n}...")
        gdf = generate_facilities(n_samples=n, center_lat=query_lat, center_lon=query_lon)
        
        # Measure index build time
        t_build_start = time.perf_counter()
        engine = SpatialIndexEngine(gdf)
        t_build_total = (time.perf_counter() - t_build_start) * 1000.0

        knn_res = benchmark_knn(engine, query_lat, query_lon, k=5, iterations=30)
        rad_res = benchmark_radius(engine, query_lat, query_lon, radius_km=3.0, iterations=30)

        records.append({
            "N": n,
            "BuildTime_ms": round(t_build_total, 2),
            "kNN_cKDTree_ms": round(knn_res["ckdtree_ms"], 4),
            "kNN_STRtree_ms": round(knn_res["strtree_ms"], 4),
            "kNN_BruteForce_ms": round(knn_res["bruteforce_ms"], 4),
            "kNN_Speedup_cKDTree": round(knn_res["speedup_ckdtree"], 1),
            "Radius_STRtree_ms": round(rad_res["strtree_ms"], 4),
            "Radius_cKDTree_ms": round(rad_res["ckdtree_ms"], 4),
            "Radius_BruteForce_ms": round(rad_res["bruteforce_ms"], 4),
            "Radius_Speedup_STRtree": round(rad_res["speedup_strtree"], 1)
        })

    df = pd.DataFrame(records)

    # Plot results
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: k-NN Latency Comparison (Log-Log Scale)
    ax1.plot(df["N"], df["kNN_BruteForce_ms"], "r-o", label="Brute Force O(N)", linewidth=2.2)
    ax1.plot(df["N"], df["kNN_STRtree_ms"], "g-s", label="GeoPandas STRtree R-Tree", linewidth=2.2)
    ax1.plot(df["N"], df["kNN_cKDTree_ms"], "b-^", label="3D Spherical cKDTree", linewidth=2.2)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of Facilities (N)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Query Latency (ms, log-scale)", fontsize=11, fontweight="bold")
    ax1.set_title("k-NN Query Latency vs Dataset Size", fontsize=12, fontweight="bold")
    ax1.grid(True, which="both", ls="--", alpha=0.6)
    ax1.legend(fontsize=10)

    # Subplot 2: Speedup Factor Multiplier
    ax2.plot(df["N"], df["kNN_Speedup_cKDTree"], "b-^", label="cKDTree Speedup (vs Brute-force)", linewidth=2.2)
    ax2.plot(df["N"], df["Radius_Speedup_STRtree"], "g-s", label="STRtree Radius Speedup", linewidth=2.2)
    ax2.set_xlabel("Number of Facilities (N)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Speedup Multiplier (x times faster)", fontsize=11, fontweight="bold")
    ax2.set_title("Indexing Speedup Multiplier vs Brute Force", fontsize=12, fontweight="bold")
    ax2.grid(True, which="both", ls="--", alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_chart_path, dpi=200)
    plt.close()
    print(f"[OK] Benchmark chart saved to {output_chart_path}")

    return df
