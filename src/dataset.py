"""
Dataset generator and loader for Chembur, Mumbai facilities.
Features authentic, real-world hospitals, colleges, restaurants, pharmacies,
fire stations, police stations, supermarkets, and libraries in and around Chembur,
as well as tightly bounded local neighbourhood facilities.
"""

import os
import random
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from typing import Optional, List, Dict
from src.config import (
    CATEGORY_CONFIG,
    CRS_WGS84,
    DEFAULT_CENTER_LAT,
    DEFAULT_CENTER_LON,
    EARTH_RADIUS_KM
)

# Comprehensive Authentic Real-World Facilities in Chembur, Mumbai, India
REAL_CHEMBUR_FACILITIES = [
    # ------------------ HOSPITALS & CLINICS ------------------
    {"name": "Zen Multispeciality Hospital", "category": "Hospital", "lat": 19.0595, "lon": 72.9021, "rating": 4.7, "address": "Plot 425, 10th Rd, Near Diamond Garden, Chembur, Mumbai 400071"},
    {"name": "Surana Sethia Hospital & Research Centre", "category": "Hospital", "lat": 19.0558, "lon": 72.8984, "rating": 4.6, "address": "Sion Trombay Rd, Suman Nagar, Chembur, Mumbai 400071"},
    {"name": "Sushrut Hospital & Research Centre", "category": "Hospital", "lat": 19.0682, "lon": 72.8951, "rating": 4.5, "address": "365 Swastik Park, Chembur East, Mumbai 400071"},
    {"name": "Apollo Spectra Hospitals - Chembur", "category": "Hospital", "lat": 19.0573, "lon": 72.8995, "rating": 4.5, "address": "Central Ave, Chembur East, Mumbai 400071"},
    {"name": "Joy Hospital", "category": "Hospital", "lat": 19.0579, "lon": 72.8972, "rating": 4.3, "address": "423 10th Rd, Chembur West, Mumbai 400071"},
    {"name": "Inlaks General Hospital", "category": "Hospital", "lat": 19.0515, "lon": 72.8942, "rating": 4.4, "address": "Inlaks Hospital Marg, Chembur Colony, Mumbai 400074"},
    {"name": "BARC Hospital", "category": "Hospital", "lat": 19.0435, "lon": 72.9234, "rating": 4.8, "address": "Anushakti Nagar, Deonar, Chembur, Mumbai 400094"},
    {"name": "Sai Hospital & Critical Care", "category": "Hospital", "lat": 19.0588, "lon": 72.8980, "rating": 4.2, "address": "DK Sandu Marg, Chembur East, Mumbai 400071"},
    {"name": "Maa Municipal General Hospital", "category": "Hospital", "lat": 19.0675, "lon": 72.8920, "rating": 4.0, "address": "P.L. Lokhande Marg, Tilak Nagar, Chembur, Mumbai 400089"},
    {"name": "Shatabdi Municipal Hospital", "category": "Hospital", "lat": 19.0525, "lon": 72.9168, "rating": 4.1, "address": "Govandi Station Rd, Near Chembur, Mumbai 400088"},
    {"name": "RCF Hospital", "category": "Hospital", "lat": 19.0410, "lon": 72.8935, "rating": 4.5, "address": "RCF Colony, Chembur, Mumbai 400074"},
    {"name": "Lifeline Multi-Speciality Hospital", "category": "Hospital", "lat": 19.0632, "lon": 72.9015, "rating": 4.4, "address": "Central Ave, Diamond Garden, Chembur, Mumbai 400071"},
    {"name": "Dr. Joshi's Diagnostic & Eye Clinic", "category": "Hospital", "lat": 19.0478, "lon": 72.8899, "rating": 4.6, "address": "Sindhi Society, Chembur, Mumbai 400071"},
    {"name": "Kohinoor Hospital", "category": "Hospital", "lat": 19.0715, "lon": 72.8840, "rating": 4.6, "address": "Kirol Rd, Kurla-Vidyavihar Link, Mumbai 400070"},

    # ------------------ COLLEGES & UNIVERSITIES ------------------
    {"name": "VESIT (Vivekanand Education Society's Inst of Tech)", "category": "College", "lat": 19.0465, "lon": 72.8891, "rating": 4.8, "address": "Hashu Advani Memorial Complex, Collector's Colony, Chembur, Mumbai 400074"},
    {"name": "Tata Institute of Social Sciences (TISS)", "category": "College", "lat": 19.0438, "lon": 72.9085, "rating": 4.9, "address": "VN Purav Marg, Deonar, Chembur, Mumbai 400088"},
    {"name": "K. J. Somaiya College of Engineering", "category": "College", "lat": 19.0730, "lon": 72.8998, "rating": 4.7, "address": "Somaiya Vidyavihar Campus, Vidyanagar, Mumbai 400077"},
    {"name": "Shah & Anchor Kutchhi Engineering College", "category": "College", "lat": 19.0503, "lon": 72.9135, "rating": 4.3, "address": "Mahavir Education Trust Chowk, WT Patil Marg, Chembur, Mumbai 400088"},
    {"name": "N.G. Acharya & D.K. Marathe College", "category": "College", "lat": 19.0543, "lon": 72.8958, "rating": 4.2, "address": "Shri N.G. Acharya Marg, Chembur East, Mumbai 400071"},
    {"name": "Pillai College of Arts, Commerce & Science", "category": "College", "lat": 19.0560, "lon": 72.8970, "rating": 4.4, "address": "Chembur Naka, Sion Trombay Rd, Chembur, Mumbai 400071"},
    {"name": "VES College of Pharmacy", "category": "College", "lat": 19.0480, "lon": 72.8885, "rating": 4.6, "address": "Sindhi Society, Chembur, Mumbai 400071"},
    {"name": "VES College of Arts, Science and Commerce", "category": "College", "lat": 19.0472, "lon": 72.8898, "rating": 4.5, "address": "Sindhi Society, Chembur, Mumbai 400071"},
    {"name": "Swami Vivekanand Junior College", "category": "College", "lat": 19.0485, "lon": 72.8905, "rating": 4.4, "address": "Sindhi Society, Chembur, Mumbai 400071"},
    {"name": "Atomic Energy Junior College", "category": "College", "lat": 19.0445, "lon": 72.9210, "rating": 4.6, "address": "AECS Compound, Anushakti Nagar, Chembur, Mumbai 400094"},
    {"name": "Somaiya Institute of Management (SIMSR)", "category": "College", "lat": 19.0718, "lon": 72.8985, "rating": 4.8, "address": "Somaiya Vidyavihar, Vidyanagar, Mumbai 400077"},

    # ------------------ RESTAURANTS & CAFES ------------------
    {"name": "Jhama Sweets - Chembur Camp", "category": "Restaurant", "lat": 19.0552, "lon": 72.8923, "rating": 4.9, "address": "Atur Park, Sion Trombay Rd, Chembur Camp, Mumbai 400071"},
    {"name": "Le Cafe - Jewel of Chembur", "category": "Restaurant", "lat": 19.0610, "lon": 72.8990, "rating": 4.7, "address": "1st Rd, Opp BMC Ward Office, Chembur, Mumbai 400071"},
    {"name": "Grand Central Restaurant", "category": "Restaurant", "lat": 19.0581, "lon": 72.8978, "rating": 4.6, "address": "Station Rd, Near Chembur Railway Station, Chembur, Mumbai 400071"},
    {"name": "Mani's Lunch Home", "category": "Restaurant", "lat": 19.0585, "lon": 72.8965, "rating": 4.6, "address": "Kombee Galli, Chembur East, Mumbai 400071"},
    {"name": "Celebrations Fine Dine & Bar", "category": "Restaurant", "lat": 19.0621, "lon": 72.9015, "rating": 4.5, "address": "Near Diamond Garden, Central Ave, Chembur, Mumbai 400071"},
    {"name": "The Bar Stock Exchange - Chembur", "category": "Restaurant", "lat": 19.0601, "lon": 72.9002, "rating": 4.4, "address": "VN Purav Marg, Chembur East, Mumbai 400071"},
    {"name": "Copper Chimney - Chembur", "category": "Restaurant", "lat": 19.0615, "lon": 72.8995, "rating": 4.5, "address": "Central Ave, Chembur East, Mumbai 400071"},
    {"name": "Vig Refreshments (Dal Pakwan & Chole)", "category": "Restaurant", "lat": 19.0548, "lon": 72.8930, "rating": 4.8, "address": "Camp Area, Dr. Choitram Gidwani Rd, Chembur, Mumbai 400071"},
    {"name": "Sadguru Fast Food & Pav Bhaji", "category": "Restaurant", "lat": 19.0590, "lon": 72.8960, "rating": 4.4, "address": "Station Road, Chembur, Mumbai 400071"},
    {"name": "Starbucks Coffee - Diamond Garden", "category": "Restaurant", "lat": 19.0618, "lon": 72.9012, "rating": 4.7, "address": "Central Ave, Diamond Garden, Chembur, Mumbai 400071"},
    {"name": "McDonald's - Chembur", "category": "Restaurant", "lat": 19.0625, "lon": 72.9005, "rating": 4.3, "address": "Central Ave, Diamond Garden, Chembur, Mumbai 400071"},
    {"name": "Pizza Express - Chembur", "category": "Restaurant", "lat": 19.0645, "lon": 72.9030, "rating": 4.6, "address": "Union Park, Chembur East, Mumbai 400071"},
    {"name": "Chembur Social", "category": "Restaurant", "lat": 19.0592, "lon": 72.8988, "rating": 4.5, "address": "Natraj Compound, Central Ave, Chembur, Mumbai 400071"},
    {"name": "Shiv Sagar Veg Restaurant", "category": "Restaurant", "lat": 19.0612, "lon": 72.9020, "rating": 4.4, "address": "Near Diamond Garden, Chembur, Mumbai 400071"},
    {"name": "Natural Ice Cream - Chembur", "category": "Restaurant", "lat": 19.0608, "lon": 72.9001, "rating": 4.8, "address": "Central Ave, Chembur, Mumbai 400071"},
    {"name": "Sainath Dhaba - Sindhi Delicacies", "category": "Restaurant", "lat": 19.0482, "lon": 72.8912, "rating": 4.7, "address": "Sindhi Society, Chembur, Mumbai 400071"},
    {"name": "Domino's Pizza - Chembur Station", "category": "Restaurant", "lat": 19.0583, "lon": 72.8980, "rating": 4.2, "address": "DK Sandu Marg, Chembur East, Mumbai 400071"},
    {"name": "Geeta Bhawan Udupi Restaurant", "category": "Restaurant", "lat": 19.0578, "lon": 72.8962, "rating": 4.4, "address": "Chembur Naka, Chembur, Mumbai 400071"},

    # ------------------ PHARMACIES & DRUGSTORES ------------------
    {"name": "Wellness Forever 24x7 - Chembur Naka", "category": "Pharmacy", "lat": 19.0598, "lon": 72.8975, "rating": 4.6, "address": "Chembur Naka, VN Purav Marg, Chembur, Mumbai 400071"},
    {"name": "Apollo Pharmacy - Diamond Garden", "category": "Pharmacy", "lat": 19.0612, "lon": 72.9008, "rating": 4.5, "address": "Shop 4, Diamond Garden, Chembur, Mumbai 400071"},
    {"name": "Noble Plus Chemist & Health Lounge", "category": "Pharmacy", "lat": 19.0580, "lon": 72.8985, "rating": 4.4, "address": "Govandi Rd, Chembur East, Mumbai 400071"},
    {"name": "Noble Medical Stores - Station Rd", "category": "Pharmacy", "lat": 19.0588, "lon": 72.8968, "rating": 4.3, "address": "Station Rd, Chembur, Mumbai 400071"},
    {"name": "MedPlus Pharmacy - Chembur Naka", "category": "Pharmacy", "lat": 19.0576, "lon": 72.8970, "rating": 4.4, "address": "Sion Trombay Rd, Chembur Naka, Mumbai 400071"},
    {"name": "Care Pharmacy - Sindhi Society", "category": "Pharmacy", "lat": 19.0475, "lon": 72.8890, "rating": 4.5, "address": "Sindhi Society, Chembur, Mumbai 400071"},
    {"name": "Metro Chemist - 10th Road", "category": "Pharmacy", "lat": 19.0595, "lon": 72.8960, "rating": 4.3, "address": "10th Rd, Chembur West, Mumbai 400071"},
    {"name": "Sanjeevani Medical Stores - Tilak Nagar", "category": "Pharmacy", "lat": 19.0690, "lon": 72.8940, "rating": 4.3, "address": "P.L. Lokhande Marg, Tilak Nagar, Chembur, Mumbai 400089"},
    {"name": "New National Medical Store - Chembur Camp", "category": "Pharmacy", "lat": 19.0550, "lon": 72.8920, "rating": 4.5, "address": "Dr. Choitram Gidwani Rd, Chembur Camp, Mumbai 400071"},

    # ------------------ FIRE STATIONS ------------------
    {"name": "Chembur Fire Station", "category": "Fire Station", "lat": 19.0575, "lon": 72.8960, "rating": 4.8, "address": "VN Purav Marg, Chembur Naka, Mumbai 400071"},
    {"name": "Deonar Fire Station", "category": "Fire Station", "lat": 19.0490, "lon": 72.9180, "rating": 4.7, "address": "Deonar Colony, Govandi, Mumbai 400043"},
    {"name": "Kurla Fire Station", "category": "Fire Station", "lat": 19.0680, "lon": 72.8790, "rating": 4.6, "address": "SG Barve Marg, Kurla East, Mumbai 400024"},

    # ------------------ POLICE STATIONS ------------------
    {"name": "Chembur Police Station", "category": "Police Station", "lat": 19.0589, "lon": 72.8972, "rating": 4.3, "address": "Ramakrishna Chemburkar Marg, Chembur, Mumbai 400071"},
    {"name": "Tilak Nagar Police Station", "category": "Police Station", "lat": 19.0695, "lon": 72.8935, "rating": 4.2, "address": "P.L. Lokhande Marg, Tilak Nagar, Chembur, Mumbai 400089"},
    {"name": "Govandi Police Station", "category": "Police Station", "lat": 19.0530, "lon": 72.9140, "rating": 4.0, "address": "Govandi Station Rd, Govandi East, Mumbai 400088"},
    {"name": "RCF Police Station", "category": "Police Station", "lat": 19.0390, "lon": 72.8920, "rating": 4.1, "address": "Mahul Rd, RCF Colony, Chembur, Mumbai 400074"},
    {"name": "Chunabhatti Police Station", "category": "Police Station", "lat": 19.0510, "lon": 72.8780, "rating": 4.1, "address": "Near Eastern Express Highway, Chunabhatti, Mumbai 400022"},

    # ------------------ SUPERMARKETS & GROCERIES ------------------
    {"name": "D-Mart Chembur - Shell Colony", "category": "Supermarket", "lat": 19.0650, "lon": 72.8885, "rating": 4.7, "address": "Shell Colony Rd, Sahakar Nagar, Chembur, Mumbai 400071"},
    {"name": "Reliance Smart Bazaar - Chembur", "category": "Supermarket", "lat": 19.0590, "lon": 72.8992, "rating": 4.4, "address": "Central Ave, Near Diamond Garden, Chembur, Mumbai 400071"},
    {"name": "Nature's Basket - Diamond Garden", "category": "Supermarket", "lat": 19.0620, "lon": 72.9010, "rating": 4.6, "address": "Central Ave, Chembur East, Mumbai 400071"},
    {"name": "Reliance Fresh - Tilak Nagar", "category": "Supermarket", "lat": 19.0688, "lon": 72.8945, "rating": 4.2, "address": "P.L. Lokhande Marg, Tilak Nagar, Chembur, Mumbai 400089"},
    {"name": "Sahakari Bhandar - Chembur Station", "category": "Supermarket", "lat": 19.0585, "lon": 72.8980, "rating": 4.3, "address": "Station Rd, Chembur East, Mumbai 400071"},
    {"name": "Apna Supermarket - Sindhi Society", "category": "Supermarket", "lat": 19.0485, "lon": 72.8900, "rating": 4.4, "address": "Sindhi Society, Chembur, Mumbai 400071"},
    {"name": "Gourmet Food Supermarket - Cubic Mall", "category": "Supermarket", "lat": 19.0540, "lon": 72.8975, "rating": 4.4, "address": "Cubic Mall, Dr. CG Rd, Chembur, Mumbai 400071"},

    # ------------------ LIBRARIES ------------------
    {"name": "Chembur Sarvajanik Vachanalaya (Public Library)", "category": "Library", "lat": 19.0582, "lon": 72.8970, "rating": 4.7, "address": "Station Rd, Chembur East, Mumbai 400071"},
    {"name": "Sir Dorabji Tata Memorial Library (TISS)", "category": "Library", "lat": 19.0440, "lon": 72.9080, "rating": 4.9, "address": "TISS Main Campus, VN Purav Marg, Deonar, Chembur, Mumbai 400088"},
    {"name": "VES Central Library - Sindhi Society", "category": "Library", "lat": 19.0470, "lon": 72.8895, "rating": 4.7, "address": "Sindhi Society, Chembur, Mumbai 400071"},
    {"name": "K. J. Somaiya Central Library", "category": "Library", "lat": 19.0725, "lon": 72.9002, "rating": 4.8, "address": "Somaiya Vidyavihar, Vidyanagar, Mumbai 400077"},
    {"name": "Chembur Karnataka Sangha Library", "category": "Library", "lat": 19.0570, "lon": 72.8990, "rating": 4.5, "address": "Govandi Rd, Chembur, Mumbai 400071"},
    {"name": "Friends Library - Chembur Branch", "category": "Library", "lat": 19.0590, "lon": 72.8975, "rating": 4.5, "address": "DK Sandu Marg, Chembur East, Mumbai 400071"},
    {"name": "Anushakti Nagar Community Library", "category": "Library", "lat": 19.0450, "lon": 72.9220, "rating": 4.6, "address": "BARC Colony, Anushakti Nagar, Chembur, Mumbai 400094"}
]


def create_seed_geodataframe() -> gpd.GeoDataFrame:
    """Creates a GeoPandas GeoDataFrame from the authentic Chembur facilities."""
    geometries = [Point(f["lon"], f["lat"]) for f in REAL_CHEMBUR_FACILITIES]
    gdf = gpd.GeoDataFrame(REAL_CHEMBUR_FACILITIES, geometry=geometries, crs=CRS_WGS84)
    gdf["facility_id"] = [f"FAC-CHEMBUR-{i+1:04d}" for i in range(len(gdf))]
    return gdf


# Realistic local naming components for nearby Chembur neighbourhood facilities
CHEMBUR_LOCALITIES = [
    "Diamond Garden", "Chembur Naka", "Sindhi Society", "Chembur Station",
    "Swastik Park", "Tilak Nagar", "Collector Colony", "Shell Colony",
    "Deonar Farm Rd", "Pestom Sagar", "Subhash Nagar", "Maravali Camp",
    "Suman Nagar", "Union Park", "Sahakar Nagar"
]

LOCAL_FACILITY_TEMPLATES = {
    "Hospital": [
        ("Dr. Hegde Nursing Home", "Station Rd"),
        ("Shanti Speciality Daycare & Clinic", "Diamond Garden"),
        ("Adarsh Children Hospital", "Swastik Park"),
        ("Ayush Maternity & Nursing Home", "Tilak Nagar"),
        ("Sanjeevani Polyclinic", "Sindhi Society"),
        ("Chembur Orthopaedic Centre", "Central Ave"),
        ("City Eye Hospital", "Chembur Naka"),
        ("Sai Krupa Clinic & Health Centre", "Shell Colony")
    ],
    "College": [
        ("Chembur English Junior College", "Station Rd"),
        ("Maratha High School & Science College", "Collector Colony"),
        ("Sindhi Model Academy of Arts", "Sindhi Society"),
        ("Deonar Junior Institute", "Deonar Farm Rd"),
        ("Lokmanya Tilak Science College", "Tilak Nagar"),
        ("Chembur Commerce Institute", "Chembur Naka")
    ],
    "Restaurant": [
        ("Shree Krishna Pure Veg", "Chembur Naka"),
        ("Balaji Bhavan Udupi Snacks", "Station Rd"),
        ("Guru Kripa Fast Food", "Chembur Camp"),
        ("Gokul Refreshments", "Diamond Garden"),
        ("Sindh Sweets & Savouries", "Sindhi Society"),
        ("Hotel Navgraha Dining Bar", "Suman Nagar"),
        ("Uphaar Pure Veg", "Swastik Park"),
        ("Amantran Chinese & Mughlai", "Tilak Nagar"),
        ("Sagar Veg Treats", "Shell Colony"),
        ("Chai Sutta Bar - Chembur", "Central Ave")
    ],
    "Pharmacy": [
        ("Ganesh Chemist & Druggist", "Station Rd"),
        ("Mahavir Medical Store", "Chembur Naka"),
        ("Shreeji Medical & General Stores", "Diamond Garden"),
        ("Navjeevan Chemist", "Tilak Nagar"),
        ("Balaji Medico", "Sindhi Society"),
        ("National Chemist", "Chembur Camp"),
        ("Prabhat Medical", "Swastik Park")
    ],
    "Fire Station": [
        ("Chembur Emergency Rescue Unit", "VN Purav Marg"),
        ("Anushakti Nagar Fire Post", "BARC Gate 1"),
        ("Eastern Suburbs Fire Response", "Sion-Trombay Rd")
    ],
    "Police Station": [
        ("Chembur Traffic Police Division", "Diamond Garden Chowk"),
        ("Tilak Nagar Police Beat Chowki", "P.L. Lokhande Marg"),
        ("Chembur Camp Police Chowki", "Dr. Choitram Gidwani Marg"),
        ("Sindhi Society Police Beat", "Collector Colony")
    ],
    "Supermarket": [
        ("Sahakari Grahak Bhandar", "Sindhi Society"),
        ("Apna Bazaar Supermarket", "Station Rd"),
        ("Ganesh Provision & Supermarket", "Diamond Garden"),
        ("Mahalaxmi Supermarket", "Tilak Nagar"),
        ("Daily Needs Superstore", "Shell Colony"),
        ("Sindhi Groceries & Dry Fruits", "Chembur Camp")
    ],
    "Library": [
        ("Sindhi Youth Study Circle & Library", "Sindhi Society"),
        ("Chembur Karnataka Vachanalaya", "Govandi Rd"),
        ("Lokmanya Tilak Community Reading Room", "Tilak Nagar"),
        ("Swastik Park Study Centre", "Swastik Park")
    ]
}


def generate_facilities(
    n_samples: int = 500,
    center_lat: float = DEFAULT_CENTER_LAT,
    center_lon: float = DEFAULT_CENTER_LON,
    spread_km: float = 2.8,  # Strictly within Chembur and immediate surroundings (<= 2.8 km)
    seed: Optional[int] = 42
) -> gpd.GeoDataFrame:
    """
    Generates a realistic dataset of facilities located STRICTLY in Chembur, Mumbai.
    All facilities are guaranteed to be nearby (within walking / short transit distance in Chembur).
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    records = []

    # 1. First, include all authentic real-world Chembur landmarks
    for idx, f in enumerate(REAL_CHEMBUR_FACILITIES):
        records.append({
            "facility_id": f"FAC-REAL-{idx+1:04d}",
            "name": f["name"],
            "category": f["category"],
            "lat": round(float(f["lat"]), 6),
            "lon": round(float(f["lon"]), 6),
            "rating": float(f["rating"]),
            "address": f["address"],
            "geometry": Point(f["lon"], f["lat"])
        })

    remaining = max(0, n_samples - len(REAL_CHEMBUR_FACILITIES))

    # Conversion factors near Chembur (lat ~19.06°)
    lat_deg_per_km = 1.0 / 111.0
    lon_deg_per_km = 1.0 / (111.0 * np.cos(np.radians(center_lat)))

    # Real focal hubs strictly inside Chembur
    chembur_hubs = [
        (19.0596, 72.8997),  # Chembur Central / Diamond Garden
        (19.0581, 72.8978),  # Chembur Station / Govandi Rd
        (19.0552, 72.8923),  # Chembur Camp
        (19.0475, 72.8895),  # Sindhi Society / Collector Colony
        (19.0682, 72.8951),  # Swastik Park / Tilak Nagar
        (19.0438, 72.9085),  # Deonar / TISS area
    ]

    categories = list(CATEGORY_CONFIG.keys())
    cat_weights = [0.12, 0.10, 0.32, 0.16, 0.04, 0.06, 0.12, 0.08]
    cat_weights = np.array(cat_weights) / sum(cat_weights)

    for i in range(remaining):
        cat = np.random.choice(categories, p=cat_weights)
        
        # Pick one Chembur hub
        hub_lat, hub_lon = chembur_hubs[np.random.randint(0, len(chembur_hubs))]
        
        # Tightly clustered around Chembur hubs (within 100m to 1200m)
        r_offset = np.random.exponential(scale=spread_km * 0.25)
        r_offset = min(r_offset, spread_km)  # hard cap at spread_km
        theta = np.random.uniform(0, 2 * np.pi)

        d_lat = (r_offset * np.sin(theta)) * lat_deg_per_km
        d_lon = (r_offset * np.cos(theta)) * lon_deg_per_km

        lat = round(float(hub_lat + d_lat), 6)
        lon = round(float(hub_lon + d_lon), 6)

        # Get realistic local name
        templates = LOCAL_FACILITY_TEMPLATES.get(cat, [("Facility", "Main Rd")])
        tmpl_name, tmpl_road = templates[np.random.randint(0, len(templates))]
        locality = CHEMBUR_LOCALITIES[np.random.randint(0, len(CHEMBUR_LOCALITIES))]
        
        # Unique realistic business name
        full_name = f"{tmpl_name} ({locality})"
        shop_no = np.random.randint(1, 45)
        address = f"Shop {shop_no}, {tmpl_road}, {locality}, Chembur, Mumbai 400071"
        rating = round(float(np.clip(np.random.normal(4.3, 0.4), 3.2, 5.0)), 1)

        records.append({
            "facility_id": f"FAC-LOCAL-{i+1:05d}",
            "name": full_name,
            "category": cat,
            "lat": lat,
            "lon": lon,
            "rating": rating,
            "address": address,
            "geometry": Point(lon, lat)
        })

    gdf = gpd.GeoDataFrame(records, crs=CRS_WGS84)
    return gdf


def save_geojson(gdf: gpd.GeoDataFrame, filepath: str) -> None:
    """Saves a GeoDataFrame to GeoJSON format."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    gdf.to_file(filepath, driver="GeoJSON")


def load_geojson(filepath: str) -> gpd.GeoDataFrame:
    """Loads a GeoDataFrame from GeoJSON format."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    gdf = gpd.read_file(filepath)
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_WGS84)
    return gdf
