#!/usr/bin/env python3
"""
Etape 3 : generer des itineraires de trafic coherents pour Marrakech.
"""

from pathlib import Path
import os
import subprocess
import sys


BASE_DIR = Path(__file__).resolve().parent
SUMO_APP_DIR = BASE_DIR.parents[1]
sys.path.insert(0, str(SUMO_APP_DIR))

from sumo_paths import find_sumo_tool, sumo_tools_path

NET_FILE = BASE_DIR / "marrakech.net.xml"
TRIPS_FILE = BASE_DIR / "marrakech_rebuilt.trips.xml"
ROUTES_FILE = BASE_DIR / "marrakech_rebuilt.rou.xml"
TOOLS_DIR = sumo_tools_path()
RANDOM_TRIPS = TOOLS_DIR / "randomTrips.py" if TOOLS_DIR else Path("randomTrips.py")
DUAROUTER = find_sumo_tool("duarouter")

SIM_BEGIN = 0
SIM_END = 7200
VEHICLE_PERIOD = 6


def generer_itineraires():
    print("=" * 60)
    print("  Generation des itineraires Marrakech")
    print("=" * 60)

    if not NET_FILE.exists():
        print(f"Reseau manquant : {NET_FILE}")
        sys.exit(1)
    if not RANDOM_TRIPS.exists():
        print(f"randomTrips.py introuvable : {RANDOM_TRIPS}")
        sys.exit(1)
    if not DUAROUTER.exists():
        print(f"duarouter introuvable : {DUAROUTER}")
        sys.exit(1)

    cmd_trips = [
        sys.executable,
        str(RANDOM_TRIPS),
        "--net-file", str(NET_FILE),
        "--output-trip-file", str(TRIPS_FILE),
        "--begin", str(SIM_BEGIN),
        "--end", str(SIM_END),
        "--period", str(VEHICLE_PERIOD),
        "--vehicle-class", "passenger",
        "--seed", "42",
        "--fringe-factor", "8",
        "--min-distance", "1200",
        "--validate",
    ]

    print("[1/2] Generation des trips...")
    proc = subprocess.run(cmd_trips, check=False)
    if proc.returncode != 0 or not TRIPS_FILE.exists():
        print("Echec de generation des trips")
        sys.exit(1)

    cmd_routes = [
        str(DUAROUTER),
        "--net-file", str(NET_FILE),
        "--route-files", str(TRIPS_FILE),
        "--output-file", str(ROUTES_FILE),
        "--ignore-errors",
        "--repair",
        "--seed", "42",
    ]

    print("[2/2] Calcul des routes...")
    proc2 = subprocess.run(cmd_routes, check=False)
    if proc2.returncode != 0 or not ROUTES_FILE.exists():
        print("Echec de duarouter")
        sys.exit(1)

    print(f"Routes creees : {ROUTES_FILE}")


if __name__ == "__main__":
    generer_itineraires()
