#!/usr/bin/env python3
"""
Etape 2 : convertir le fichier OSM en reseau SUMO plus fiable.
Le script garde uniquement le reseau routier utile aux voitures/ambulances
et retire les composantes isolees pour ameliorer le routage.
"""

from pathlib import Path
import os
import subprocess
import sys


BASE_DIR = Path(__file__).resolve().parent
SUMO_APP_DIR = BASE_DIR.parents[1]
sys.path.insert(0, str(SUMO_APP_DIR))

from sumo_paths import find_sumo_tool

INPUT_OSM = BASE_DIR / "marrakech.osm"
OUTPUT_NET = BASE_DIR / "marrakech.net.xml"
NETCONVERT = find_sumo_tool("netconvert")


NETCONVERT_PARAMS = [
    str(NETCONVERT),
    "--osm-files", str(INPUT_OSM),
    "--output-file", str(OUTPUT_NET),
    "--keep-edges.by-vclass", "passenger,emergency,bus,delivery,taxi",
    "--remove-edges.isolated",
    "--keep-edges.components", "1",
    "--keep-edges.min-speed", "1",
    "--geometry.remove",
    "--geometry.min-dist", "2.5",
    "--roundabouts.guess",
    "--ramps.guess",
    "--junctions.join",
    "--junctions.join-dist", "12",
    "--junctions.corner-detail", "5",
    "--proj.utm",
    "--tls.guess",
    "--tls.join",
    "--tls.default-type", "actuated",
    "--default.speed", "50",
    "--default.lanenumber", "1",
    "--output.street-names",
    "--output.original-names",
    "--verbose",
]


def convertir():
    print("=" * 60)
    print("  Conversion OSM -> reseau SUMO Marrakech")
    print("=" * 60)

    if not INPUT_OSM.exists():
        print(f"Fichier manquant : {INPUT_OSM}")
        print("Lancez d'abord 01_telecharger_osm.py")
        sys.exit(1)

    if not NETCONVERT.exists():
        print(f"netconvert introuvable : {NETCONVERT}")
        sys.exit(1)

    version = subprocess.run(
        [str(NETCONVERT), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if version.stdout:
        print(version.stdout.splitlines()[0])

    proc = subprocess.run(NETCONVERT_PARAMS, check=False)
    if proc.returncode != 0 or not OUTPUT_NET.exists():
        print(f"Echec de conversion reseau (code {proc.returncode})")
        sys.exit(1)

    size_mb = OUTPUT_NET.stat().st_size / 1024 / 1024
    print(f"Reseau cree : {OUTPUT_NET} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    convertir()
