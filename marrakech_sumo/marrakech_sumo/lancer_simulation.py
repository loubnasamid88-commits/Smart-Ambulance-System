#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SUMO_APP_DIR = BASE_DIR.parents[1]
sys.path.insert(0, str(SUMO_APP_DIR))

from sumo_paths import find_sumo_tool, sumo_binary, sumo_tools_path


NETCONVERT = find_sumo_tool("netconvert")
DUAROUTER = find_sumo_tool("duarouter")
SUMO = sumo_binary(gui=False)
SUMO_GUI = sumo_binary(gui=True)

BBOX = {
    "south": 31.6200,
    "west": -8.0050,
    "north": 31.6350,
    "east": -7.9850,
}

FILES = {
    "osm": BASE_DIR / "marrakech.osm",
    "net": BASE_DIR / "marrakech.net.xml",
    "trips": BASE_DIR / "marrakech.trips.xml",
    "routes": BASE_DIR / "marrakech.rou.xml",
    "config": BASE_DIR / "marrakech.sumocfg",
}

SIM_DURATION = 3600
VEHICLE_PERIOD = 5


def banner(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def require_file(path, label):
    path = Path(path)
    if not path.exists():
        print(f"{label} introuvable : {path}")
        sys.exit(1)
    print(f"OK {label}: {path}")


def check_sumo():
    banner("Verification de SUMO")
    require_file(NETCONVERT, "netconvert")
    require_file(DUAROUTER, "duarouter")
    require_file(SUMO, "sumo")


def telecharger_osm():
    banner("Etape 1 - Telechargement OSM")

    query = f"""
[out:xml][timeout:240];
(
  way["highway"]
     ({BBOX['south']},{BBOX['west']},{BBOX['north']},{BBOX['east']});
);
(._;>;);
out body;
"""

    data = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request("https://overpass-api.de/api/interpreter", data=data)

    with urllib.request.urlopen(req) as response:
        FILES["osm"].write_bytes(response.read())

    print(f"OSM telecharge : {FILES['osm']}")


def convertir_reseau():
    banner("Etape 2 - Conversion reseau")

    cmd = [
        str(NETCONVERT),
        "--osm-files", str(FILES["osm"]),
        "--output-file", str(FILES["net"]),
        "--geometry.remove",
        "--roundabouts.guess",
        "--ramps.guess",
        "--junctions.join",
        "--remove-edges.isolated",
        "--proj.utm",
        "--tls.guess",
        "--default.speed", "50",
    ]

    subprocess.run(cmd, check=True)
    require_file(FILES["net"], "reseau SUMO")


def generer_itineraires():
    banner("Etape 3 - Generation trafic")

    tools_dir = sumo_tools_path()
    random_trips = tools_dir / "randomTrips.py" if tools_dir else Path("randomTrips.py")
    require_file(random_trips, "randomTrips.py")

    subprocess.run([
        sys.executable,
        str(random_trips),
        "--net-file", str(FILES["net"]),
        "--output-trip-file", str(FILES["trips"]),
        "--begin", "0",
        "--end", str(SIM_DURATION),
        "--period", str(VEHICLE_PERIOD),
        "--vehicle-class", "passenger",
    ], check=True)

    subprocess.run([
        str(DUAROUTER),
        "--net-file", str(FILES["net"]),
        "--route-files", str(FILES["trips"]),
        "--output-file", str(FILES["routes"]),
    ], check=True)

    require_file(FILES["routes"], "routes SUMO")


def creer_config():
    banner("Creation fichier .sumocfg")

    config_content = f"""<configuration>
    <input>
        <net-file value="{FILES['net'].name}"/>
        <route-files value="{FILES['routes'].name}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{SIM_DURATION}"/>
    </time>
</configuration>
"""

    FILES["config"].write_text(config_content, encoding="utf-8")
    print(f"Config creee : {FILES['config']}")


def lancer_simulation(gui=False):
    banner("Etape 4 - Lancement simulation")
    exe = SUMO_GUI if gui else SUMO
    require_file(exe, "simulateur SUMO")
    subprocess.run([str(exe), "-c", str(FILES["config"])], cwd=BASE_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--skip-dl", action="store_true")
    parser.add_argument("--sim-only", action="store_true")
    args = parser.parse_args()

    check_sumo()

    if not args.sim_only:
        if not args.skip_dl:
            telecharger_osm()
        convertir_reseau()
        generer_itineraires()
        creer_config()

    lancer_simulation(gui=args.gui)


if __name__ == "__main__":
    main()
