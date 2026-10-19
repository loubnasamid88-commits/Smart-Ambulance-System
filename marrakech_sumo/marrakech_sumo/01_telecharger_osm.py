#!/usr/bin/env python3
"""
Etape 1 : telecharger une zone reduite de Marrakech autour des hopitaux.
"""

import os
import urllib.parse
import urllib.request


# Zone reduite utile au projet: hopitaux + marge pour l'ambulance.
BBOX = {
    "south": 31.6145,
    "west": -7.9990,
    "north": 31.6391,
    "east": -7.9680,
}

OUTPUT_FILE = "marrakech.osm"


def telecharger_osm():
    print("=" * 60)
    print("  Telechargement OSM Marrakech reduit")
    print("=" * 60)

    query = f"""
[out:xml][timeout:180];
(
  way["highway"~"motorway|trunk|primary|secondary|tertiary|
                  unclassified|residential|service|living_street|
                  pedestrian|track|road|motorway_link|trunk_link|
                  primary_link|secondary_link|tertiary_link"]
     ({BBOX['south']},{BBOX['west']},{BBOX['north']},{BBOX['east']});
);
(._;>;);
out body;
"""

    encoded_query = urllib.parse.urlencode({"data": query})
    url = f"https://overpass-api.de/api/interpreter?{encoded_query}"

    print(
        f"Zone : {BBOX['south']}N - {BBOX['north']}N | "
        f"{BBOX['west']}E - {BBOX['east']}E"
    )
    print("Telechargement depuis : overpass-api.de ...")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SUMO-Marrakech/1.0"})
        with urllib.request.urlopen(req, timeout=180) as response:
            data = response.read()

        with open(OUTPUT_FILE, "wb") as f:
            f.write(data)

        size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
        print(f"Fichier sauvegarde : {OUTPUT_FILE} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"Erreur telechargement : {e}")
        print("Alternative : telechargez manuellement sur https://www.openstreetmap.org/export")
        print(f"Bbox : {BBOX['west']},{BBOX['south']},{BBOX['east']},{BBOX['north']}")
        return False


if __name__ == "__main__":
    telecharger_osm()
