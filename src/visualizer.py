"""
Folium Map Visualizer for Location-Based Service.
Generates interactive, rich maps featuring:
- Query location marker with pulsating/distinct styling
- Search radius buffer circles (semi-transparent geodesic buffers)
- Category-coded icons and markers (Hospitals, Colleges, Restaurants, etc.)
- Rich HTML popup cards (rating, distance, category, address)
- Polyline connecting rays between query point and nearest neighbors
- LayerControl, MiniMap, Fullscreen, and MeasureControl plugins
"""

import folium
from folium import plugins
import geopandas as gpd
from typing import Optional, List, Dict, Any, Tuple
from src.config import CATEGORY_CONFIG, DEFAULT_CENTER_LAT, DEFAULT_CENTER_LON


def create_facility_popup_html(
    row: Dict[str, Any],
    query_dist_km: Optional[float] = None
) -> str:
    """Generates a styled HTML popup card for a facility."""
    cat = row.get("category", "Facility")
    config = CATEGORY_CONFIG.get(cat, {"color": "blue", "icon": "info-circle"})
    color = config["color"]
    
    rating_val = row.get("rating", 4.0)
    stars_html = "★" * int(round(rating_val)) + "☆" * (5 - int(round(rating_val)))

    dist_badge = ""
    if query_dist_km is not None or "distance_km" in row:
        dist = query_dist_km if query_dist_km is not None else row["distance_km"]
        dist_str = f"{dist * 1000.0:.0f} m" if dist < 1.0 else f"{dist:.2f} km"
        dist_badge = f"""
        <div style="background-color: #f1f5f9; border-radius: 6px; padding: 4px 8px; margin-top: 6px; font-weight: 600; font-size: 13px; color: #0f172a;">
            📍 Distance: <span style="color: #2563eb;">{dist_str}</span>
        </div>
        """

    html = f"""
    <div style="font-family: 'Segoe UI', -apple-system, sans-serif; width: 240px; padding: 4px;">
        <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 2px solid {color}; padding-bottom: 6px; margin-bottom: 8px;">
            <span style="font-weight: 700; font-size: 14px; color: #1e293b;">{row.get('name', 'Facility')}</span>
        </div>
        <div style="display: inline-block; background-color: {color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; text-transform: uppercase;">
            {cat}
        </div>
        <div style="margin-top: 6px; font-size: 13px; color: #475569;">
            <span style="color: #eab308; font-size: 14px;">{stars_html}</span> ({rating_val})
        </div>
        <div style="margin-top: 6px; font-size: 12px; color: #64748b; line-height: 1.4;">
            🏠 {row.get('address', 'No address')}
        </div>
        {dist_badge}
    </div>
    """
    return html


def build_facility_map(
    all_facilities: Optional[gpd.GeoDataFrame] = None,
    query_lat: Optional[float] = None,
    query_lon: Optional[float] = None,
    nearest_facilities: Optional[gpd.GeoDataFrame] = None,
    radius_km: Optional[float] = None,
    radius_facilities: Optional[gpd.GeoDataFrame] = None,
    title: str = "Nearby Facilities Search"
) -> folium.Map:
    """
    Builds an interactive Folium map illustrating spatial queries.
    """
    # Determine center coordinates
    if query_lat is not None and query_lon is not None:
        center = [query_lat, query_lon]
        zoom_start = 14 if (radius_km and radius_km <= 3.0) else 13
    elif all_facilities is not None and len(all_facilities) > 0:
        center = [all_facilities["lat"].mean(), all_facilities["lon"].mean()]
        zoom_start = 13
    else:
        center = [DEFAULT_CENTER_LAT, DEFAULT_CENTER_LON]
        zoom_start = 13

    # Initialize Folium Map
    fmap = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # Add standard plugins
    plugins.Fullscreen(position="topright").add_to(fmap)
    plugins.MeasureControl(position="topleft", primary_length_unit="kilometers").add_to(fmap)
    plugins.MiniMap(toggle_display=True, position="bottomright").add_to(fmap)

    # 1. Background layer of all available facilities (clustered by category)
    if all_facilities is not None and len(all_facilities) > 0:
        all_layer = folium.FeatureGroup(name="All Facilities (Background)", show=False)
        cluster = plugins.MarkerCluster().add_to(all_layer)

        for _, row in all_facilities.iterrows():
            cat = row.get("category", "Facility")
            cfg = CATEGORY_CONFIG.get(cat, {"color": "blue", "icon": "info", "prefix": "fa"})
            
            folium.Marker(
                location=[row["lat"], row["lon"]],
                popup=folium.Popup(create_facility_popup_html(row), max_width=280),
                tooltip=f"{row.get('name', 'Facility')} ({cat})",
                icon=folium.Icon(color=cfg["color"], icon=cfg["icon"], prefix=cfg["prefix"])
            ).add_to(cluster)

        all_layer.add_to(fmap)

    # 2. Query Location Marker
    if query_lat is not None and query_lon is not None:
        query_group = folium.FeatureGroup(name="Query Location", show=True)
        
        query_popup_html = f"""
        <div style="font-family: sans-serif; width: 200px;">
            <h4 style="margin: 0 0 6px 0; color: #dc2626;">📍 Query Origin</h4>
            <b>Latitude:</b> {query_lat:.6f}<br/>
            <b>Longitude:</b> {query_lon:.6f}<br/>
            {"<b>Search Radius:</b> " + str(radius_km) + " km<br/>" if radius_km else ""}
        </div>
        """

        folium.Marker(
            location=[query_lat, query_lon],
            popup=folium.Popup(query_popup_html, max_width=250),
            tooltip="User Location / Query Point",
            icon=folium.Icon(color="black", icon="user", prefix="fa")
        ).add_to(query_group)

        # Draw circle buffer if radius specified
        if radius_km is not None and radius_km > 0:
            folium.Circle(
                location=[query_lat, query_lon],
                radius=radius_km * 1000.0,
                color="#2563eb",
                weight=2,
                fill=True,
                fill_color="#3b82f6",
                fill_opacity=0.15,
                tooltip=f"Search Buffer: {radius_km} km radius"
            ).add_to(query_group)

        query_group.add_to(fmap)

    # 3. Radius Query Results Layer
    if radius_facilities is not None and len(radius_facilities) > 0:
        radius_layer = folium.FeatureGroup(name=f"Within Radius ({len(radius_facilities)} found)", show=True)
        for _, row in radius_facilities.iterrows():
            cat = row.get("category", "Facility")
            cfg = CATEGORY_CONFIG.get(cat, {"color": "blue", "icon": "info", "prefix": "fa"})
            
            folium.Marker(
                location=[row["lat"], row["lon"]],
                popup=folium.Popup(create_facility_popup_html(row), max_width=280),
                tooltip=f"{row.get('name', 'Facility')} - {row.get('distance_m', 0):.0f}m",
                icon=folium.Icon(color=cfg["color"], icon=cfg["icon"], prefix=cfg["prefix"])
            ).add_to(radius_layer)
        radius_layer.add_to(fmap)

    # 4. Nearest Neighbors (k-NN) Layer with Connecting Polylines
    if nearest_facilities is not None and len(nearest_facilities) > 0:
        knn_layer = folium.FeatureGroup(name=f"Top {len(nearest_facilities)} Nearest Facilities", show=True)
        
        for rank, (_, row) in enumerate(nearest_facilities.iterrows(), start=1):
            cat = row.get("category", "Facility")
            cfg = CATEGORY_CONFIG.get(cat, {"color": "blue", "icon": "info", "prefix": "fa"})
            dist_km = row.get("distance_km", 0.0)
            dist_m = row.get("distance_m", dist_km * 1000.0)

            # Draw connecting ray from query point to facility
            if query_lat is not None and query_lon is not None:
                folium.PolyLine(
                    locations=[[query_lat, query_lon], [row["lat"], row["lon"]]],
                    color="#ef4444" if rank == 1 else "#3b82f6",
                    weight=3 if rank == 1 else 2,
                    dash_array="6, 6",
                    opacity=0.8,
                    tooltip=f"Rank #{rank}: {row.get('name')} ({dist_m:.0f}m away)"
                ).add_to(knn_layer)

            # Add marker for nearest facility
            folium.Marker(
                location=[row["lat"], row["lon"]],
                popup=folium.Popup(create_facility_popup_html(row, dist_km), max_width=280),
                tooltip=f"#{rank} Nearest: {row.get('name')} ({dist_m:.0f}m)",
                icon=folium.Icon(color=cfg["color"], icon=cfg["icon"], prefix=cfg["prefix"])
            ).add_to(knn_layer)

        knn_layer.add_to(fmap)

    # Layer Control to allow user toggling
    folium.LayerControl(position="topright", collapsed=False).add_to(fmap)
    return fmap
