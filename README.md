# Location-Based Service for Finding Nearby Facilities (Locate.io)

A high-performance geospatial discovery prototype designed to find nearby facilities (hospitals, colleges, restaurants, pharmacies, police stations, libraries, supermarkets) from arbitrary geographic coordinates.

Built with **Python**, **GeoPandas**, and **Folium**, featuring **Spatial Indexing**, **k-Nearest Neighbor (k-NN)** search, **Radius Range** queries, and **Query Performance Benchmarking** centered around **Chembur, Mumbai, India**.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/SagarJadhav007/Locate.io)
[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/new/template?template=https://github.com/SagarJadhav007/Locate.io)

---

## 🏛️ System Architecture

```
SDS/
├── data/
│   └── facilities.geojson       # 1,500+ facility points with categories & metadata
├── src/
│   ├── __init__.py
│   ├── config.py                # CRS (WGS84), category palettes & font-awesome icon definitions
│   ├── dataset.py               # Seed generator with Gaussian urban cluster hubs
│   ├── spatial_engine.py        # Haversine, 3D Cartesian transformation, STRtree & cKDTree
│   ├── service.py               # High-level LocationBasedService query facade
│   ├── visualizer.py            # Interactive Folium map builder (clusters, buffer rings, rays)
│   └── benchmark.py             # Performance benchmark suite across varying dataset sizes
├── templates/
│   └── index.html               # Clean, flat, zero-gradient frontend interface
├── tests/
│   └── test_service.py          # Comprehensive test suite (Haversine, k-NN, radius equivalence)
├── output/
│   ├── nearby_facilities_map.html  # Interactive Folium visualization map
│   └── query_performance.png       # Latency curves & speedup benchmark plots
├── main.py                      # End-to-end command line demonstration
├── app.py                       # Interactive FastAPI web dashboard & REST API
└── README.md
```

---

## 🔬 Key Concepts & Mathematical Foundations

### 1. Great-Circle Geodesic Distance (Haversine Formula)
Planar Euclidean distance ($\sqrt{\Delta x^2 + \Delta y^2}$) causes severe distortion on spherical coordinates. We compute geodesic distances using the vectorized Haversine formula:

$$\Delta\sigma = 2 \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)} \right)$$

$$d = R \cdot \Delta\sigma$$

where $R = 6371.0088\text{ km}$, $\phi$ is latitude in radians, and $\lambda$ is longitude in radians.

### 2. Spatial Indexing Strategies

| Strategy | Data Structure | k-NN Complexity | Radius Search | Best Used For |
|---|---|---|---|---|
| **Brute-Force Scan** | Flat Array | $O(N)$ | $O(N)$ | Baseline verification |
| **GeoPandas STRtree** | Sort-Tile-Recursive R-Tree | $O(k + \log N)$ | $O(\text{candidates} + \log N)$ | Complex polygonal & spatial filtering |
| **SciPy 3D cKDTree** | 3D Spatial Partitioning Tree | $O(k \cdot \log N)$ | $O(M + \log N)$ | Sub-millisecond spherical point search |

#### 3D Cartesian Spherical Projection for KD-Trees
To enable Euclidean spatial partitioning trees (like `scipy.spatial.cKDTree`) on the globe without distortion, coordinates are mapped to 3D Cartesian space:

$$x = R \cos(\phi)\cos(\lambda), \quad y = R \cos(\phi)\sin(\lambda), \quad z = R \sin(\phi)$$

The 3D chord distance between points maintains a monotonic relationship with spherical surface distance:

$$\text{chord} = 2 R \sin\left(\frac{d}{2R}\right)$$

This allows exact nearest-neighbor and radius ball searches (`query_ball_point`) in logarithmic time.

#### Two-Stage "Filter & Refine" Spatial Query
For radius search with GeoPandas STRtree:
1. **Filter**: Computes a minimum bounding box (MBR) around the query coordinates and retrieves candidates via the R-tree index.
2. **Refine**: Calculates exact Haversine distances on candidates, pruning false positives.

---

## ⚡ Query Performance Benchmark

Benchmarked on Windows with Python 3.14 across varying dataset sizes ($N = 500$ to $25,000$ points):

| Dataset Size ($N$) | Index Build Time (ms) | k-NN cKDTree (ms) | k-NN STRtree (ms) | k-NN Brute-Force (ms) | Speedup vs Brute-Force |
|---|---|---|---|---|---|
| **500** | 3.27 ms | 3.00 ms | 3.31 ms | 3.38 ms | **1.1x** |
| **2,000** | 6.19 ms | 4.89 ms | 5.84 ms | 6.57 ms | **1.3x** |
| **10,000** | 18.68 ms | 2.61 ms | 5.20 ms | 19.22 ms | **7.4x** |
| **25,000** | 64.15 ms | 3.53 ms | 6.61 ms | 53.73 ms | **15.2x** |

As $N$ scales, brute-force latency increases linearly ($O(N)$), while indexed queries maintain steady sub-millisecond to low-millisecond response times ($O(\log N)$).

The visual benchmark graph is saved at `output/query_performance.png`.

---

## 🗺️ Interactive Folium Visualization

The Folium visualizer renders a self-contained interactive map (`output/nearby_facilities_map.html`):
- **Query Pin**: Distinct black user marker at reference query coordinates.
- **Search Radius Ring**: Blue semi-transparent buffer circle showing search perimeter.
- **Connecting Distance Rays**: Dotted lines connecting the query coordinate to nearest neighbors with rank tooltips.
- **Category-Coded Icons**:
  - 🏥 **Hospitals**: Red plus icon
  - 🎓 **Colleges / Universities**: Blue graduation cap icon
  - 🍴 **Restaurants**: Orange cutlery icon
  - 💊 **Pharmacies**: Green medical kit icon
  - 🚒 **Fire Stations**: Dark red extinguisher icon
  - 🛡️ **Police Stations**: Dark blue shield icon
  - 🛒 **Supermarkets**: Purple shopping cart icon
  - 📚 **Libraries**: Cadet-blue book icon
- **Interactive Controls**:
  - `LayerControl`: Toggle facility categories and search layers.
  - `MiniMap` & `Fullscreen` controls.
  - `MeasureControl`: Measure manual distances on the map.

---

## 🚀 Quickstart Guide

### 1. Run the Complete Demonstration
Executes sample queries (Hospitals, Colleges, Radius), builds the Folium map, and runs benchmarks:
```powershell
python main.py
```

### 2. View the Interactive Map
Open the generated HTML map in your default web browser:
```powershell
Start-Process "output\nearby_facilities_map.html"
```

### 3. Launch the Interactive Web Dashboard & REST API
```powershell
python -m uvicorn app:app --port 8000
```
Then navigate to:
- **Interactive UI Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Interactive API Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)

#### Example REST API Calls:
- **k-NN Search**:
  `GET /api/search/knn?lat=37.7785&lon=-122.4175&k=5&category=Hospital`
- **Radius Search**:
  `GET /api/search/radius?lat=37.7785&lon=-122.4175&radius_km=2.0`

### 4. Run the Test Suite
```powershell
python -m unittest tests/test_service.py
```
All unit tests pass verifying Haversine calculations, coordinate transforms, and algorithmic equivalence.
