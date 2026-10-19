import os
import random
import threading
import time
from datetime import datetime
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO
from concurrent.futures import ThreadPoolExecutor
import math
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from vision_ai.vision_worker import VisionWorker
from sumo_paths import BASE_DIR, ensure_traci_on_path, sumo_binary, sumo_config
# ============================
# SUMO CONFIG
# ============================
SUMO_CONFIG      = str(sumo_config())
SUMO_BINARY      = str(sumo_binary(gui=True))

import sys
ensure_traci_on_path()

try:
    import traci
    TRACI_AVAILABLE = True
    print("✅ TraCI disponible")
except ImportError:
    TRACI_AVAILABLE = False
    print("⚠ TraCI non disponible — mode fallback")

# ============================
# FLASK + SOCKETIO
# ============================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
executor = ThreadPoolExecutor(max_workers=10)

# ============================
# ÉTAT GLOBAL SUMO
# ============================
simulation_running = False
sumo_thread        = None
sumo_lock          = threading.Lock()
sumo_sim_time      = 0
sumo_vehicle_count = 0
_amb_route_counter = 0
vision_worker = None
# Accidents SUMO matérialisés par des véhicules immobilisés
_sumo_accident_scenes = {}
_sumo_accident_pois = set()
_sumo_scene_route_counter = 0
_sumo_hospital_pois = set()
_sumo_hospital_polygons = set()
_sumo_route_polygons = {}

# ============================
# DONNÉES DASHBOARD
# ============================
SIMULATION_STEPS    = 20
SIMULATION_INTERVAL = 0.3

ambulances = [
    {
        "id": i + 1,
        "lat": 31.6295 + random.uniform(-0.008, 0.008),
        "lng": -7.9810 + random.uniform(-0.008, 0.008),
        "status": "available",
        "eta": None,
        "history": [],
        "missions": 0,
        "distance_travelled": 0.0,
        "times": [],
    }
    for i in range(6)
]

hospitals = [
    {"id": 1, "name": "Hôpital Mohammed VI", "lat": 31.6278, "lng": -7.9890, "capacity": 50, "current_load": random.randint(10, 40)},
    {"id": 2, "name": "Clinique Gueliz",      "lat": 31.6311, "lng": -7.9865, "capacity": 40, "current_load": random.randint(10, 35)},
    {"id": 3, "name": "Hôpital Ibn Tofail",   "lat": 31.6225, "lng": -7.9780, "capacity": 60, "current_load": random.randint(20, 50)},
]

accidents  = []
evaluation_history = []
statistics = {
    "total_interventions": 0,
    "completed": 0,
    "total_distance": 0.0,
    "total_time": 0.0,
    "interventions_per_hour": {}
}

# ============================
# UTILS GÉO
# ============================
def haversine(lat1, lon1, lat2, lon2):
    R    = 6371
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a    = (math.sin(dLat / 2) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
            * math.sin(dLon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def interpolate_path(start, end, steps=None):
    n = steps or SIMULATION_STEPS
    return [
        {
            "lat": start["lat"] + (end["lat"] - start["lat"]) * i / n,
            "lng": start["lng"] + (end["lng"] - start["lng"]) * i / n,
        }
        for i in range(1, n + 1)
    ]

# ============================
# TRACI — HELPERS
# ============================

def _edge_allows_passenger(edge_id):
    """
    Vérifie qu'au moins une lane de l'edge accepte les véhicules
    de type 'passenger'. Évite l'erreur 'not allowed to depart'.
    """
    try:
        nb_lanes = traci.edge.getLaneNumber(edge_id)
        for i in range(nb_lanes):
            allowed = traci.lane.getAllowed(f"{edge_id}_{i}")
            # allowed vide = tout est autorisé
            if not allowed or "passenger" in allowed or "emergency" in allowed:
                return True
        return False
    except Exception:
        return False


def _get_nearest_normal_edge(lat, lng):
    """
    Convertit GPS → edge SUMO valide :
      - Non interne (pas de ':')
      - Accepte les véhicules passenger
    """
    if not TRACI_AVAILABLE or not simulation_running:
        return None
    try:
        with sumo_lock:
            x, y = traci.simulation.convertGeo(lng, lat, fromGeo=True)

            # Tentative directe
            try:
                edge_id, _, _ = traci.simulation.convertRoad(x, y, isGeo=False)
                if (edge_id
                        and not edge_id.startswith(':')
                        and _edge_allows_passenger(edge_id)):
                    return edge_id
            except Exception:
                pass

            # Fallback : scanner tous les edges valides
            all_edges = traci.edge.getIDList()
            nearest   = None
            min_dist  = float("inf")

            for edge_id in all_edges:
                if edge_id.startswith(':'):
                    continue
                if not _edge_allows_passenger(edge_id):
                    continue
                try:
                    nb_lanes = traci.edge.getLaneNumber(edge_id)
                    for i in range(nb_lanes):
                        shape = traci.lane.getShape(f"{edge_id}_{i}")
                        if not shape:
                            continue
                        mid    = shape[len(shape) // 2]
                        ex, ey = mid[0], mid[1]
                        dist   = math.sqrt((x - ex) ** 2 + (y - ey) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            nearest  = edge_id
                except Exception:
                    continue

            if nearest:
                print(f"  📍 Edge valide : {nearest} (dist={min_dist:.1f}m)")
            else:
                print(f"  ⚠ Aucun edge valide pour ({lat:.4f}, {lng:.4f})")
            return nearest

    except Exception as e:
        print(f"  ⚠ _get_nearest_normal_edge : {e}")
        return None


def _inject_ambulance_sumo(amb_id_str, from_edge, to_edge, color=(0, 100, 255, 255)):
    """
    Injecte une ambulance dans SUMO avec le plus court chemin réel.
    color RGBA :
      🔵 (0, 100, 255, 255) = bleu  → phase 1 : vers l'accident
      🟢 (0, 200,  80, 255) = vert  → phase 2 : vers l'hôpital
    """
    global _amb_route_counter

    if not TRACI_AVAILABLE or not simulation_running:
        return False
    if not from_edge or not to_edge:
        print(f"  ⚠ Edge manquant : from={from_edge} to={to_edge}")
        return False
    if from_edge.startswith(':') or to_edge.startswith(':'):
        print(f"  ⚠ Edge interne → fallback")
        return False
    if from_edge == to_edge:
        print(f"  ⚠ from==to ({from_edge}) → fallback")
        return False

    try:
        with sumo_lock:
            _amb_route_counter += 1
            route_id = f"route_amb_{_amb_route_counter}"

            path = traci.simulation.findRoute(from_edge, to_edge)
            if not path.edges:
                print(f"  ⚠ Pas de chemin {from_edge} → {to_edge}")
                return False

            _show_sumo_route(amb_id_str, path.edges, color)
            traci.route.add(route_id, path.edges)
            traci.vehicle.add(
                vehID       = amb_id_str,
                routeID     = route_id,
                typeID      = "DEFAULT_VEHTYPE",
                depart      = "now",
                departLane  = "best",
                departSpeed = "max",
            )
            traci.vehicle.setMaxSpeed(amb_id_str, 22.0)
            traci.vehicle.setColor(amb_id_str, color)  # couleur selon phase
            traci.vehicle.setSpeedMode(amb_id_str, 7)  # passe les feux rouges

            if color[1] > color[2]:
                label = "🟢 VERT→HÔPITAL"
            else:
                label = "🔵 BLEU→ACCIDENT"
            print(f"  ✅ {amb_id_str} [{label}] → {len(path.edges)} edges ({path.length:.0f}m)")
            return True

    except Exception as e:
        print(f"  ❌ Injection échouée : {e}")
        return False


def _show_sumo_route(route_key, edges, color):
    """
    Dessine dans SUMO le chemin prévu de l'ambulance pour le rendre visible.
    """
    if not TRACI_AVAILABLE or not simulation_running or not edges:
        return

    polygon_id = f"route_vis_{route_key}"
    try:
        if polygon_id in _sumo_route_polygons:
            return

        shape = []
        for edge_id in edges:
            if edge_id.startswith(":"):
                continue
            lane_id = f"{edge_id}_0"
            lane_shape = traci.lane.getShape(lane_id)
            if not lane_shape:
                continue
            if shape and shape[-1] == lane_shape[0]:
                shape.extend(lane_shape[1:])
            else:
                shape.extend(lane_shape)

        if len(shape) < 2:
            return

        is_hospital_phase = color[1] > color[2]
        line_width = 18.0 if is_hospital_phase else 12.0
        marker_size = 13.0 if is_hospital_phase else 9.0
        step = 3 if is_hospital_phase else 6

        traci.polygon.add(
            polygonID=polygon_id,
            shape=shape,
            color=color,
            fill=False,
            polygonType="ambulance_route",
            layer=160,
            lineWidth=line_width,
        )

        marker_ids = []
        for idx, (mx, my) in enumerate(shape[::step]):
            marker_id = f"{polygon_id}_dot_{idx}"
            traci.poi.add(
                poiID=marker_id,
                x=mx,
                y=my,
                color=color,
                poiType="ambulance_route_marker",
                layer=161,
                imgFile="",
                width=marker_size,
                height=marker_size,
                angle=0.0,
            )
            marker_ids.append(marker_id)

        _sumo_route_polygons[polygon_id] = marker_ids
    except Exception as e:
        print(f"  ⚠ Tracé route SUMO : {e}")


def _clear_sumo_route(route_key):
    """Supprime le tracé de route affiché dans SUMO."""
    if not TRACI_AVAILABLE or not simulation_running:
        return

    polygon_id = f"route_vis_{route_key}"
    if polygon_id not in _sumo_route_polygons:
        return

    try:
        with sumo_lock:
            traci.polygon.remove(polygon_id)
            for marker_id in _sumo_route_polygons.get(polygon_id, []):
                traci.poi.remove(marker_id)
    except Exception:
        pass
    finally:
        _sumo_route_polygons.pop(polygon_id, None)


def _set_vehicle_color_sumo(veh_id, color):
    """Change la couleur d'un véhicule SUMO pendant qu'il roule."""
    if not TRACI_AVAILABLE or not simulation_running:
        return
    try:
        with sumo_lock:
            if veh_id in traci.vehicle.getIDList():
                traci.vehicle.setColor(veh_id, color)
    except Exception:
        pass


def _locate_accident_on_network(lat, lng):
    """
    Trouve l'edge, la lane et la position les plus proches pour placer
    une vraie scène d'accident dans SUMO.
    """
    if not TRACI_AVAILABLE or not simulation_running:
        return None

    try:
        with sumo_lock:
            x, y = traci.simulation.convertGeo(lng, lat, fromGeo=True)
            try:
                edge_id, pos, lane_index = traci.simulation.convertRoad(x, y, isGeo=False)
                if edge_id and not edge_id.startswith(':') and _edge_allows_passenger(edge_id):
                    lane_count = max(traci.edge.getLaneNumber(edge_id), 1)
                    lane_index = max(0, min(int(lane_index), lane_count - 1))
                    lane_id = f"{edge_id}_{lane_index}"
                    lane_length = traci.lane.getLength(lane_id)
                    pos = max(8.0, min(float(pos), max(lane_length - 8.0, 8.0)))
                    return edge_id, lane_index, pos
            except Exception:
                pass
    except Exception:
        pass

    edge_id = _get_nearest_normal_edge(lat, lng)
    if not edge_id:
        return None

    try:
        with sumo_lock:
            lane_index = 0
            lane_id = f"{edge_id}_{lane_index}"
            lane_length = traci.lane.getLength(lane_id)
            pos = max(8.0, lane_length / 2)
            return edge_id, lane_index, pos
    except Exception:
        return None


def _spawn_accident_scene(accident):
    """
    Crée un vrai accident visible dans SUMO avec deux véhicules immobilisés
    sur la chaussée, à la position de l'accident.
    """
    global _sumo_scene_route_counter

    if not TRACI_AVAILABLE or not simulation_running:
        return False
    if accident["id"] in _sumo_accident_scenes:
        return True

    location = _locate_accident_on_network(accident["lat"], accident["lng"])
    if not location:
        print(f"  ⚠ Impossible de placer l'accident #{accident['id']} dans SUMO")
        return False

    edge_id, base_lane, base_pos = location
    veh_ids = []
    poi_id = f"accident_{accident['id']}_alert"

    try:
        with sumo_lock:
            lane_count = max(traci.edge.getLaneNumber(edge_id), 1)
            lane_choices = [base_lane]
            if lane_count > 1:
                lane_choices.append((base_lane + 1) % lane_count)
            else:
                lane_choices.append(base_lane)

            vehicle_specs = [
                (lane_choices[0], base_pos, (220, 20, 60, 255)),
                (lane_choices[1], base_pos + 4.0, (255, 140, 0, 255)),
            ]

            for idx, (lane_index, stop_pos, color) in enumerate(vehicle_specs, start=1):
                lane_id = f"{edge_id}_{lane_index}"
                lane_length = traci.lane.getLength(lane_id)
                safe_pos = max(8.0, min(stop_pos, max(lane_length - 6.0, 8.0)))

                _sumo_scene_route_counter += 1
                route_id = f"accident_scene_route_{_sumo_scene_route_counter}"
                veh_id = f"accident_{accident['id']}_veh_{idx}"

                traci.route.add(route_id, [edge_id])
                traci.vehicle.add(
                    vehID=veh_id,
                    routeID=route_id,
                    typeID="DEFAULT_VEHTYPE",
                    depart="now",
                    departLane=str(lane_index),
                    departPos=str(safe_pos),
                    departSpeed="0",
                )
                traci.vehicle.setColor(veh_id, color)
                traci.vehicle.moveTo(veh_id, lane_id, safe_pos)
                traci.vehicle.setStop(
                    veh_id,
                    edge_id,
                    pos=safe_pos,
                    laneIndex=lane_index,
                    duration=3600 * 24,
                )
                traci.vehicle.setSpeed(veh_id, 0.0)
                veh_ids.append(veh_id)

            x, y = traci.simulation.convertGeo(accident["lng"], accident["lat"], fromGeo=True)
            traci.poi.add(
                poiID=poi_id,
                x=x + 22.0,
                y=y + 22.0,
                color=(255, 0, 0, 255),
                poiType="accident_alert",
                layer=220,
                imgFile="",
                width=28.0,
                height=28.0,
                angle=0.0,
            )
            traci.poi.setParameter(poi_id, "name", f"ACCIDENT #{accident['id']}")
            _sumo_accident_pois.add(poi_id)
            _sumo_accident_scenes[accident["id"]] = veh_ids
            print(
                f"  🚗 Accident réel #{accident['id']} créé dans SUMO "
                f"sur {edge_id} avec {len(veh_ids)} véhicule(s) immobilisé(s)"
            )
            return True
    except Exception as e:
        print(f"  ⚠ Scène accident SUMO : {e}")
        for veh_id in veh_ids:
            try:
                if veh_id in traci.vehicle.getIDList():
                    traci.vehicle.remove(veh_id)
            except Exception:
                pass
        try:
            if poi_id in _sumo_accident_pois:
                traci.poi.remove(poi_id)
                _sumo_accident_pois.discard(poi_id)
        except Exception:
            pass
        return False


def _sync_waiting_accidents_to_sumo():
    """
    Rejoue dans SUMO tous les accidents encore actifs.
    Permet de garder Flask et SUMO synchronisés.
    """
    if not TRACI_AVAILABLE or not simulation_running:
        return 0

    synced = 0
    for accident in accidents:
        if accident["status"] == "completed":
            continue
        if accident["id"] in _sumo_accident_scenes:
            continue
        if _spawn_accident_scene(accident):
            synced += 1
    return synced


def _add_hospital_poi(hospital):
    """Ajoute un hôpital comme bâtiment visible dans SUMO."""
    if not TRACI_AVAILABLE or not simulation_running:
        return False

    poi_id = f"hospital_{hospital['id']}"
    building_id = f"{poi_id}_building"
    annex_id = f"{poi_id}_annex"
    entrance_id = f"{poi_id}_entrance"
    cross_v_id = f"{poi_id}_cross_v"
    cross_h_id = f"{poi_id}_cross_h"

    if (
        poi_id in _sumo_hospital_pois
        and building_id in _sumo_hospital_polygons
        and annex_id in _sumo_hospital_polygons
        and entrance_id in _sumo_hospital_polygons
        and cross_v_id in _sumo_hospital_polygons
        and cross_h_id in _sumo_hospital_polygons
    ):
        return True

    try:
        with sumo_lock:
            x, y = traci.simulation.convertGeo(
                hospital["lng"], hospital["lat"], fromGeo=True
            )
            # Silhouette plus réaliste : corps principal avec ailes latérales.
            building_shape = [
                (x - 34.0, y - 18.0),
                (x - 14.0, y - 18.0),
                (x - 14.0, y - 28.0),
                (x + 14.0, y - 28.0),
                (x + 14.0, y - 18.0),
                (x + 34.0, y - 18.0),
                (x + 34.0, y + 18.0),
                (x + 14.0, y + 18.0),
                (x + 14.0, y + 30.0),
                (x - 14.0, y + 30.0),
                (x - 14.0, y + 18.0),
                (x - 34.0, y + 18.0),
            ]
            annex_shape = [
                (x - 18.0, y - 8.0),
                (x + 18.0, y - 8.0),
                (x + 18.0, y + 8.0),
                (x - 18.0, y + 8.0),
            ]
            entrance_shape = [
                (x - 9.0, y + 30.0),
                (x + 9.0, y + 30.0),
                (x + 9.0, y + 42.0),
                (x - 9.0, y + 42.0),
            ]
            cross_vertical = [
                (x - 5.0, y - 12.0),
                (x + 5.0, y - 12.0),
                (x + 5.0, y + 12.0),
                (x - 5.0, y + 12.0),
            ]
            cross_horizontal = [
                (x - 12.0, y - 5.0),
                (x + 12.0, y - 5.0),
                (x + 12.0, y + 5.0),
                (x - 12.0, y + 5.0),
            ]

            traci.polygon.add(
                polygonID=building_id,
                shape=building_shape,
                color=(232, 238, 242, 255),
                fill=True,
                polygonType="hospital_building",
                layer=170,
                lineWidth=2.5,
            )
            traci.polygon.add(
                polygonID=annex_id,
                shape=annex_shape,
                color=(205, 216, 224, 255),
                fill=True,
                polygonType="hospital_annex",
                layer=169,
                lineWidth=1.5,
            )
            traci.polygon.add(
                polygonID=entrance_id,
                shape=entrance_shape,
                color=(148, 175, 194, 255),
                fill=True,
                polygonType="hospital_entrance",
                layer=172,
                lineWidth=1.5,
            )
            traci.polygon.add(
                polygonID=cross_v_id,
                shape=cross_vertical,
                color=(200, 0, 0, 255),
                fill=True,
                polygonType="hospital_cross",
                layer=171,
                lineWidth=1.0,
            )
            traci.polygon.add(
                polygonID=cross_h_id,
                shape=cross_horizontal,
                color=(200, 0, 0, 255),
                fill=True,
                polygonType="hospital_cross",
                layer=171,
                lineWidth=1.0,
            )
            traci.poi.add(
                poiID=poi_id,
                x=x,
                y=y + 48.0,
                color=(0, 200, 80, 255),
                poiType="hospital",
                layer=180,
                imgFile="",
                width=12.0,
                height=12.0,
                angle=0.0,
            )
            traci.poi.setParameter(poi_id, "name", hospital["name"])
            _sumo_hospital_pois.add(poi_id)
            _sumo_hospital_polygons.update({
                building_id,
                annex_id,
                entrance_id,
                cross_v_id,
                cross_h_id,
            })
            print(f"  🏥 Hôpital visible dans SUMO : {hospital['name']}")
            return True
    except Exception as e:
        print(f"  ⚠ POI hôpital : {e}")
        return False


def _sync_hospitals_to_sumo():
    """Synchronise les hôpitaux de l'application vers SUMO."""
    if not TRACI_AVAILABLE or not simulation_running:
        return 0

    synced = 0
    for hospital in hospitals:
        if _add_hospital_poi(hospital):
            synced += 1
    return synced


def _remove_accident_scene(accident_id):
    """Supprime la scène d'accident SUMO une fois la mission terminée."""
    if not TRACI_AVAILABLE or not simulation_running:
        return
    veh_ids = _sumo_accident_scenes.pop(accident_id, [])
    poi_id = f"accident_{accident_id}_alert"
    if not veh_ids:
        try:
            with sumo_lock:
                if poi_id in _sumo_accident_pois:
                    traci.poi.remove(poi_id)
                    _sumo_accident_pois.discard(poi_id)
        except Exception:
            pass
        return
    try:
        with sumo_lock:
            current_ids = set(traci.vehicle.getIDList())
            for veh_id in veh_ids:
                if veh_id in current_ids:
                    traci.vehicle.remove(veh_id)
            if poi_id in _sumo_accident_pois:
                traci.poi.remove(poi_id)
                _sumo_accident_pois.discard(poi_id)
    except Exception:
        pass


def _get_vehicle_position_sumo(veh_id):
    """Récupère la position GPS réelle depuis SUMO."""
    if not TRACI_AVAILABLE or not simulation_running:
        return None
    try:
        with sumo_lock:
            if veh_id not in traci.vehicle.getIDList():
                return None
            x, y     = traci.vehicle.getPosition(veh_id)
            lon, lat = traci.simulation.convertGeo(x, y, fromGeo=False)
            speed    = traci.vehicle.getSpeed(veh_id)
            return {"lat": lat, "lng": lon, "speed": speed}
    except Exception:
        return None


def _remove_vehicle_sumo(veh_id):
    """Supprime un véhicule de SUMO proprement."""
    if not TRACI_AVAILABLE or not simulation_running:
        return
    try:
        with sumo_lock:
            if veh_id in traci.vehicle.getIDList():
                traci.vehicle.remove(veh_id)
    except Exception:
        pass
    finally:
        _clear_sumo_route(veh_id)

# ============================
# ============================
# ML — PRÉDICTION ETA (amélioré)
# ============================
from ml_eta_module import build_and_train_model, predict_eta
build_and_train_model(model_type="auto", n_samples=3000)
# predict_eta(distance, severity_str, hospital_load) → fonctionne sans autre changement

# ============================
# DISPATCH LOGIC
# ============================
def score_ambulance(amb, accident, hospital):
    response_distance = haversine(amb["lat"], amb["lng"], accident["lat"], accident["lng"])
    hospital_distance = haversine(accident["lat"], accident["lng"], hospital["lat"], hospital["lng"])
    total_distance    = response_distance + hospital_distance
    load_ratio        = hospital["current_load"] / hospital["capacity"]
    nearest_hospital_distance = min(
        haversine(accident["lat"], accident["lng"], h["lat"], h["lng"])
        for h in hospitals
    )

    response_eta = predict_eta(response_distance, accident["severity"], hospital["current_load"])
    transfer_eta = predict_eta(hospital_distance, accident["severity"], hospital["current_load"])
    total_eta    = response_eta + transfer_eta

    severity_priority = {"low": 1.0, "medium": 1.2, "high": 1.5}[accident["severity"]]
    overload_penalty  = 12 if load_ratio >= 0.9 else 0
    mission_penalty   = amb["missions"] * 0.15
    hospital_detour_penalty = max(0.0, hospital_distance - nearest_hospital_distance) * 8.0
    distance_priority = 1.25 if accident["severity"] == "low" else 1.0

    return round(
        response_eta * 0.45 * severity_priority +
        total_eta * 0.18 +
        response_distance * 0.35 * distance_priority +
        hospital_distance * 0.95 * distance_priority +
        total_distance * 0.08 +
        load_ratio * 0.9 +
        overload_penalty +
        hospital_detour_penalty +
        mission_penalty,
        3
    )

def best_ambulance(accident):
    with sumo_lock:   # 🔥 AJOUT IMPORTANT
        available = [
            a for a in ambulances
            if a["status"] == "available" and not a.get("locked", False)
        ]

        if not available:
            return None, None

        best_amb = None
        best_hosp = None
        best_score = float("inf")

        for amb in available:
            for hosp in hospitals:
                s = score_ambulance(amb, accident, hosp)
                if s < best_score:
                    best_score = s
                    best_amb = amb
                    best_hosp = hosp

        # 🔥 LOCK IMMÉDIAT
        if best_amb:
            best_amb["locked"] = True

        return best_amb, best_hosp
def suggest_ambulance(accident):
    with sumo_lock:
        available = [
            a for a in ambulances
            if a["status"] == "available" and not a.get("locked", False)
        ]

        if not available:
            return None, None

        best_amb = None
        best_hosp = None
        best_score = float("inf")

        for amb in available:
            for hosp in hospitals:
                s = score_ambulance(amb, accident, hosp)
                if s < best_score:
                    best_score = s
                    best_amb = amb
                    best_hosp = hosp

        return best_amb, best_hosp


#j ajoute a 18:58
def auto_dispatch(accident):
    """
    Fonction centrale: déclenche automatiquement une mission
    dès qu'un accident est détecté.
    """

    amb, hosp = best_ambulance(accident)

    if not amb or not hosp:
        accident["status"] = "waiting"
        return False

    with sumo_lock:
        if accident["status"] not in ["waiting", "detected"]:
            return False

        accident["status"] = "dispatched"
        amb["status"] = "dispatched"
        amb["locked"] = False  # sécurité

    dist = haversine(
        amb["lat"], amb["lng"],
        accident["lat"], accident["lng"]
    )

    amb["eta"] = predict_eta(
        dist,
        accident["severity"],
        hosp["current_load"]
    )

    record_evaluation(accident, amb, hosp, source="auto")
    statistics["total_interventions"] += 1

    executor.submit(simulate_mission, amb, accident, hosp)

    socketio.emit("ambulance_update", amb)
    socketio.emit("accident_update", accident)

    print("🚑 AUTO DISPATCH EXECUTED")

    return True



def baseline_ambulance(accident):
    available = [
        a for a in ambulances if a["status"] == "available"
    ]
    if not available:
        return None, None

    ambulance = min(
        available,
        key=lambda a: haversine(a["lat"], a["lng"], accident["lat"], accident["lng"])
    )

    # Baseline manuel plus réaliste :
    # l'opérateur choisit souvent un hôpital de référence plutôt qu'un choix optimisé.
    if accident["severity"] == "high":
        hospital = max(hospitals, key=lambda h: h["capacity"])
    else:
        hospital = next((h for h in hospitals if h["id"] == 1), hospitals[0])
    return ambulance, hospital


def performance_score(total_eta, total_distance, load_ratio, strategy="smart"):
    coordination_penalty = 8 if strategy == "baseline" else 0
    return round(
        total_eta * 0.50
        + total_distance * 1.30
        + load_ratio * 15
        + coordination_penalty,
        2
    )


def estimate_strategy_metrics(amb, accident, hospital, strategy="smart"):
    response_distance = haversine(amb["lat"], amb["lng"], accident["lat"], accident["lng"])
    hospital_distance = haversine(accident["lat"], accident["lng"], hospital["lat"], hospital["lng"])
    total_distance = response_distance + hospital_distance
    response_eta = predict_eta(response_distance, accident["severity"], hospital["current_load"])
    transfer_eta = predict_eta(hospital_distance, accident["severity"], hospital["current_load"])
    manual_delay = 3.5 if strategy == "baseline" else 0.0
    coordination_factor = 1.12 if strategy == "baseline" else 1.0
    response_eta = round(response_eta * coordination_factor + manual_delay, 2)
    transfer_eta = round(transfer_eta * coordination_factor + (1.0 if strategy == "baseline" else 0.0), 2)
    total_eta = round(response_eta + transfer_eta, 2)
    load_ratio = hospital["current_load"] / hospital["capacity"]
    score = performance_score(total_eta, total_distance, load_ratio, strategy=strategy)
    return {
        "response_distance": round(response_distance, 2),
        "hospital_distance": round(hospital_distance, 2),
        "total_distance": round(total_distance, 2),
        "response_eta": round(response_eta, 2),
        "transfer_eta": round(transfer_eta, 2),
        "total_eta": total_eta,
        "score": score,
        "hospital_load_ratio": round(load_ratio, 2),
    }


def record_evaluation(accident, smart_amb, smart_hospital, source="dispatch"):
    baseline_amb, baseline_hospital = baseline_ambulance(accident)
    if not baseline_amb or not baseline_hospital or not smart_amb or not smart_hospital:
        return

    without_metrics = estimate_strategy_metrics(baseline_amb, accident, baseline_hospital, strategy="baseline")
    with_metrics = estimate_strategy_metrics(smart_amb, accident, smart_hospital, strategy="smart")

    def improvement(old, new):
        if old <= 0:
            return 0.0
        return round(((old - new) / old) * 100, 2)

    evaluation_history.append({
        "accident_id": accident["id"],
        "severity": accident["severity"],
        "source": source,
        "without_system": {
            "ambulance_id": baseline_amb["id"],
            "hospital_name": baseline_hospital["name"],
            **without_metrics,
        },
        "with_system": {
            "ambulance_id": smart_amb["id"],
            "hospital_name": smart_hospital["name"],
            **with_metrics,
        },
        "improvement": {
            "time_pct": improvement(without_metrics["total_eta"], with_metrics["total_eta"]),
            "distance_pct": improvement(without_metrics["total_distance"], with_metrics["total_distance"]),
            "score_pct": improvement(without_metrics["score"], with_metrics["score"]),
            "time_saved": round(without_metrics["total_eta"] - with_metrics["total_eta"], 2),
            "distance_saved": round(without_metrics["total_distance"] - with_metrics["total_distance"], 2),
            "score_saved": round(without_metrics["score"] - with_metrics["score"], 2),
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })


def build_evaluation_summary():
    if not evaluation_history:
        return {
            "cases": 0,
            "without_system": {"avg_time": 0, "avg_distance": 0, "avg_score": 0},
            "with_system": {"avg_time": 0, "avg_distance": 0, "avg_score": 0},
            "improvement": {"time_pct": 0, "distance_pct": 0, "score_pct": 0},
            "totals": {"time_saved": 0, "distance_saved": 0, "score_saved": 0},
            "latest": None,
            "history": [],
        }

    cases = len(evaluation_history)
    avg_without_time = round(sum(x["without_system"]["total_eta"] for x in evaluation_history) / cases, 2)
    avg_with_time = round(sum(x["with_system"]["total_eta"] for x in evaluation_history) / cases, 2)
    avg_without_distance = round(sum(x["without_system"]["total_distance"] for x in evaluation_history) / cases, 2)
    avg_with_distance = round(sum(x["with_system"]["total_distance"] for x in evaluation_history) / cases, 2)
    avg_without_score = round(sum(x["without_system"]["score"] for x in evaluation_history) / cases, 2)
    avg_with_score = round(sum(x["with_system"]["score"] for x in evaluation_history) / cases, 2)
    avg_time_gain = round(sum(x["improvement"]["time_pct"] for x in evaluation_history) / cases, 2)
    avg_distance_gain = round(sum(x["improvement"]["distance_pct"] for x in evaluation_history) / cases, 2)
    avg_score_gain = round(sum(x["improvement"]["score_pct"] for x in evaluation_history) / cases, 2)
    total_time_saved = round(sum(x["improvement"]["time_saved"] for x in evaluation_history), 2)
    total_distance_saved = round(sum(x["improvement"]["distance_saved"] for x in evaluation_history), 2)
    total_score_saved = round(sum(x["improvement"]["score_saved"] for x in evaluation_history), 2)
    latest = evaluation_history[-1]

    return {
        "cases": cases,
        "without_system": {
            "avg_time": avg_without_time,
            "avg_distance": avg_without_distance,
            "avg_score": avg_without_score,
        },
        "with_system": {
            "avg_time": avg_with_time,
            "avg_distance": avg_with_distance,
            "avg_score": avg_with_score,
        },
        "improvement": {
            "time_pct": avg_time_gain,
            "distance_pct": avg_distance_gain,
            "score_pct": avg_score_gain,
        },
        "totals": {
            "time_saved": total_time_saved,
            "distance_saved": total_distance_saved,
            "score_saved": total_score_saved,
        },
        "latest": {
            "accident_id": latest["accident_id"],
            "severity": latest["severity"],
            "source": latest.get("source", "dispatch"),
            "without_ambulance": latest["without_system"]["ambulance_id"],
            "with_ambulance": latest["with_system"]["ambulance_id"],
            "without_hospital": latest["without_system"]["hospital_name"],
            "with_hospital": latest["with_system"]["hospital_name"],
            "time_saved": latest["improvement"]["time_saved"],
            "distance_saved": latest["improvement"]["distance_saved"],
            "score_saved": latest["improvement"]["score_saved"],
        },
        "history": evaluation_history[-10:],
    }

# ============================
# SIMULATION AMBULANCE
# ============================
def simulate_mission(ambulance, accident, hospital):
    amb_id_str  = f"amb_{ambulance['id']}_{int(time.time())}"
    dist_to_acc = haversine(ambulance["lat"], ambulance["lng"],
                            accident["lat"],  accident["lng"])
    dist_to_hop = haversine(accident["lat"],  accident["lng"],
                            hospital["lat"],  hospital["lng"])
    total_dist  = dist_to_acc + dist_to_hop

    ambulance["status"] = "moving"
    ambulance["eta"]    = predict_eta(dist_to_acc, accident["severity"],
                                      hospital["current_load"])
    socketio.emit("ambulance_update", ambulance)

    # ── PHASE 1 : Ambulance → Accident ───────────────────────────────────
    sumo_used = False
    if TRACI_AVAILABLE and simulation_running:
        from_edge = _get_nearest_normal_edge(ambulance["lat"], ambulance["lng"])
        to_edge   = _get_nearest_normal_edge(accident["lat"],  accident["lng"])
        sumo_used = _inject_ambulance_sumo(amb_id_str, from_edge, to_edge, color=(0, 100, 255, 255))  # 🔵 BLEU → accident

    if sumo_used:
        _track_sumo_vehicle(ambulance, amb_id_str, accident)
    else:
        _track_interpolated(ambulance, ambulance.copy(), accident)

    # Accident atteint → supprimer la scène d'accident et passer au statut arrived
    accident["status"] = "arrived"
    _remove_accident_scene(accident["id"])
    socketio.emit("accident_update", accident)

    # ── PHASE 2 : Accident → Hôpital ─────────────────────────────────────
    ambulance["lat"] = accident["lat"]
    ambulance["lng"] = accident["lng"]

    if sumo_used:
        _remove_vehicle_sumo(amb_id_str)
        amb_id_str2 = f"{amb_id_str}_h"
        from_edge2  = _get_nearest_normal_edge(accident["lat"],  accident["lng"])
        to_edge2    = _get_nearest_normal_edge(hospital["lat"],  hospital["lng"])
        sumo_phase2 = _inject_ambulance_sumo(amb_id_str2, from_edge2, to_edge2, color=(0, 220, 80, 255))  # 🟢 VERT → hôpital
        if sumo_phase2:
            _track_sumo_vehicle(ambulance, amb_id_str2, hospital)
        else:
            _track_interpolated(ambulance, accident.copy(), hospital)
        _remove_vehicle_sumo(amb_id_str2)
    else:
        _track_interpolated(ambulance, accident.copy(), hospital)

    # ── Finalisation ──────────────────────────────────────────────────────
    duration                         = (SIMULATION_STEPS * 2 * SIMULATION_INTERVAL) / 60
    ambulance["status"]              = "available"
    ambulance["eta"]                 = None
    ambulance["missions"]           += 1
    ambulance["distance_travelled"] += total_dist
    ambulance["times"].append(duration)
    hospital["current_load"]        += 1
    statistics["completed"]         += 1
    statistics["total_distance"]    += total_dist
    statistics["total_time"]        += duration
    accident["status"]               = "completed"

    socketio.emit("ambulance_update", ambulance)
    socketio.emit("accident_update",  accident)
    print(f"  ✅ Mission terminée : Ambulance #{ambulance['id']}")


def _track_sumo_vehicle(ambulance, veh_id, target):
    """Suit la position réelle SUMO et pousse via socket."""
    max_wait = 180
    elapsed  = 0
    interval = 0.4
    while elapsed < max_wait:
        pos = _get_vehicle_position_sumo(veh_id)
        if pos is None:
            break
        ambulance["lat"] = pos["lat"]
        ambulance["lng"] = pos["lng"]
        ambulance["history"].append({"lat": pos["lat"], "lng": pos["lng"]})
        socketio.emit("ambulance_update", ambulance)
        time.sleep(interval)
        elapsed += interval
    ambulance["lat"] = target["lat"]
    ambulance["lng"] = target["lng"]
    ambulance["history"].append({"lat": target["lat"], "lng": target["lng"]})
    socketio.emit("ambulance_update", ambulance)


def _track_interpolated(ambulance, start, end):
    """Fallback linéaire si SUMO non disponible."""
    for pos in interpolate_path(start, end):
        ambulance["lat"] = pos["lat"]
        ambulance["lng"] = pos["lng"]
        ambulance["history"].append(pos)
        socketio.emit("ambulance_update", ambulance)
        time.sleep(SIMULATION_INTERVAL)

# ============================
# BOUCLE SUMO
# ============================
def sumo_loop():
    global simulation_running, sumo_sim_time, sumo_vehicle_count

    try:
        print(f"🚀 Démarrage SUMO : {SUMO_BINARY}")
        traci.start([
            SUMO_BINARY, "-c", SUMO_CONFIG,
            "--start",       "true",
            "--quit-on-end", "false",
            "--no-warnings", "true",
        ])
        simulation_running = True
        print("✅ SUMO connecté via TraCI")
        socketio.emit("sumo_status", {"running": True})
        hospitals_synced = _sync_hospitals_to_sumo()
        if hospitals_synced:
            print(f"✅ {hospitals_synced} hôpital(s) synchronisé(s) vers SUMO")
        synced = _sync_waiting_accidents_to_sumo()
        if synced:
            print(f"✅ {synced} accident(s) resynchronisé(s) vers SUMO")

        while simulation_running:
            with sumo_lock:
                traci.simulationStep()
                sumo_sim_time      = int(traci.simulation.getTime())
                sumo_vehicle_count = traci.vehicle.getIDCount()

            if sumo_sim_time % 5 == 0:
                socketio.emit("sumo_status", {
                    "running":  True,
                    "sim_time": sumo_sim_time,
                    "vehicles": sumo_vehicle_count,
                })
            time.sleep(0.05)

    except Exception as e:
        print(f"❌ SUMO erreur : {e}")
    finally:
        simulation_running = False
        sumo_sim_time      = 0
        sumo_vehicle_count = 0
        _sumo_accident_scenes.clear()
        _sumo_accident_pois.clear()
        _sumo_hospital_pois.clear()
        _sumo_hospital_polygons.clear()
        _sumo_route_polygons.clear()
        try:
            traci.close()
        except Exception:
            pass
        print("🛑 SUMO fermé")
        socketio.emit("sumo_status", {"running": False})

# ============================
# IOT — POSITIONS AMBULANCES LIBRES
# ============================
def iot_background():
    while True:
        for amb in ambulances:
            if amb["status"] == "available":
                amb["lat"] += (random.random() - 0.5) * 0.0004
                amb["lng"] += (random.random() - 0.5) * 0.0004
            socketio.emit("ambulance_update", amb)
        socketio.sleep(1.5)

# ============================
# ROUTES FLASK
# ============================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/data")
def get_data():
    completed   = statistics["completed"]
    avg_dist    = round(statistics["total_distance"] / completed, 2) if completed else 0
    avg_time    = round(statistics["total_time"]     / completed, 2) if completed else 0
    most_active = max(ambulances, key=lambda a: a["missions"])

    suggestions = {}
    for acc in accidents:
        if acc["status"] == "waiting":
            amb, hosp = suggest_ambulance(acc)
            if amb:
                suggestions[acc["id"]] = f"Ambulance #{amb['id']} → {hosp['name']}"

    performance = []
    for amb in ambulances:
        avg_t      = round(sum(amb["times"]) / len(amb["times"]), 2) if amb["times"] else 0
        efficiency = round((amb["missions"] / max(statistics["total_interventions"], 1)) * 100, 2)
        performance.append({
            "id":         amb["id"],
            "avg_time":   avg_t,
            "missions":   amb["missions"],
            "efficiency": efficiency,
        })

    evaluation = build_evaluation_summary()

    return jsonify({
        "ambulances":  ambulances,
        "hospitals":   hospitals,
        "accidents":   accidents,
        "statistics":  statistics,
        "analysis": {
            "average_distance": avg_dist,
            "average_time":     avg_time,
            "most_active":      most_active["id"],
        },
        "suggestions": suggestions,
        "performance": performance,
        "evaluation":  evaluation,
    })


@app.route("/performance_evaluation")
def performance_evaluation():
    return jsonify(build_evaluation_summary())

@app.route("/generate_road_accident")
def generate_road_accident():

    # 📡 Support YOLO + manuel
    source = request.args.get("source", "manual")   # manual | yolo
    event  = request.args.get("event", "accident")  # accident | congestion | etc.

    # ⚠️ sécurité parsing confidence
    try:
        confidence = float(request.args.get("confidence", 0))
    except:
        confidence = 0.0

    # 🎯 génération position (simulation SUMO map)
    lat = round(31.610 + random.random() * 0.030, 6)
    lng = round(-8.010 + random.random() * 0.040, 6)

    # 🚨 logique de gravité intelligente (YOLO influence)
    if event == "accident":
        if confidence >= 0.8:
            severity = "high"
        elif confidence >= 0.5:
            severity = "medium"
        else:
            severity = random.choice(["low", "medium"])
    else:
        severity = random.choice(["low", "medium"])

    # 🎯 objet accident
    accident = {
        "id": len(accidents) + 1,
        "lat": lat,
        "lng": lng,
        "severity": severity,
        "status": "waiting",

        # 👁 meta YOLO / system
        "source": source,
        "event_type": event,
        "confidence": confidence
    }

    # 📦 stockage
    # 📦 stockage
    accidents.append(accident)

    socketio.emit("new_accident", accident)
    auto_dispatch(accident)
    print(f"💥 Event reçu [{source.upper()}]: {event} #{accident['id']} (conf={confidence})")

# 🚫 PAS DE SUMO ICI (YOLO ONLY POLICY)
    accident["sumo_synced"] = False

    return jsonify(accident)


import time
import uuid
import random
from flask import request, jsonify


last_yolo_event_time = 0
YOLO_COOLDOWN = 2.5


@app.route("/yolo_event", methods=["POST"])
def yolo_event():
    global accidents, last_yolo_event_time

    data = request.get_json() or {}

    # =========================
    # PARSING SAFE
    # =========================
    try:
        confidence = float(data.get("confidence", 0))
    except:
        confidence = 0.0

    bbox = data.get("bbox", [])

    # =========================
    # FILTER LOW CONFIDENCE
    # =========================
    if confidence < 0.4:
        return jsonify({"status": "ignored_low_confidence"}), 200

    # =========================
    # COOLDOWN YOLO
    # =========================
    now = time.time()
    if now - last_yolo_event_time < YOLO_COOLDOWN:
        return jsonify({"status": "ignored_cooldown"}), 200
    last_yolo_event_time = now

    # =========================
    # SIMULATED POSITION
    # =========================
    lat = round(31.610 + random.random() * 0.030, 6)
    lng = round(-8.010 + random.random() * 0.040, 6)

    # =========================
    # SEVERITY
    # =========================
    if confidence >= 0.8:
        severity = "high"
    elif confidence >= 0.5:
        severity = "medium"
    else:
        severity = "low"

    # =========================
    # CREATE ACCIDENT
    # =========================
    with sumo_lock:
        accident_id = str(uuid.uuid4())

        accident = {
            "id": accident_id,
            "lat": lat,
            "lng": lng,
            "severity": severity,
            "status": "detected",
            "source": "yolo",
            "confidence": confidence,
            "bbox": bbox,
            "sumo_synced": False
        }

        accidents.append(accident)

    # =========================
    # PUSH FRONTEND
    # =========================
    socketio.emit("new_accident", accident)

    print(f"💥 YOLO accident {accident_id} conf={confidence}")

    # =========================
    # SUMO SCENE
    # =========================
    try:
        if TRACI_AVAILABLE and simulation_running:
            accident["sumo_synced"] = _spawn_accident_scene(accident)
    except Exception as e:
        print("SUMO error:", e)

    # =========================
    # 🚨 AUTO DISPATCH (FIX COMPLET)
    # =========================
    amb, hosp = best_ambulance(accident)

    if not amb or not hosp:
        accident["status"] = "waiting"
        return jsonify(accident)

    # verrouillage anti double dispatch
    with sumo_lock:
        if accident["status"] != "detected":
            return jsonify(accident)

        accident["status"] = "dispatched"
        amb["status"] = "dispatched"

    # =========================
    # ETA ML
    # =========================
    dist = haversine(
        amb["lat"], amb["lng"],
        accident["lat"], accident["lng"]
    )

    amb["eta"] = predict_eta(
        dist,
        accident["severity"],
        hosp["current_load"]
    )

    record_evaluation(accident, amb, hosp, source="yolo")
    statistics["total_interventions"] += 1

    # =========================
    # START MISSION ASYNC
    # =========================
    executor.submit(simulate_mission, amb, accident, hosp)

    # =========================
    # SOCKET UPDATE
    # =========================
    socketio.emit("ambulance_update", amb)
    socketio.emit("accident_update", accident)

    print("🚑 AUTO DISPATCH DONE")

    return jsonify(accident)

@app.route("/dispatch", methods=["POST"])
def dispatch():
    pending = [a for a in accidents if a["status"] == "waiting"]
    if not pending:
        return jsonify({"error": "Aucun accident en attente"}), 400

    dispatched = 0
    for accident in pending:
        amb, hosp = best_ambulance(accident)
        if not amb:
            continue
        record_evaluation(accident, amb, hosp)
        dist               = haversine(amb["lat"], amb["lng"], accident["lat"], accident["lng"])
        amb["eta"]         = predict_eta(dist, accident["severity"], hosp["current_load"])
        amb["status"]      = "dispatched"
        accident["status"] = "dispatched"
        statistics["total_interventions"] += 1

        executor.submit(simulate_mission, amb, accident, hosp)
        dispatched += 1

        socketio.emit("ambulance_update", amb)
        socketio.emit("accident_update",  accident)

    return jsonify({"status": f"Dispatch lancé — {dispatched} ambulance(s)"})


@app.route("/start_sumo")
def start_sumo_route():
    global sumo_thread, simulation_running

    if not TRACI_AVAILABLE:
        return jsonify({"status": "❌ TraCI non disponible — vérifiez SUMO_HOME"}), 500
    if not Path(SUMO_BINARY).exists():
        return jsonify({"status": f"❌ SUMO introuvable : {SUMO_BINARY}"}), 500
    if not Path(SUMO_CONFIG).exists():
        return jsonify({"status": f"❌ Fichier config introuvable : {SUMO_CONFIG}"}), 500
    if simulation_running:
        return jsonify({"status": "⚠ SUMO déjà en cours"})

    sumo_thread = threading.Thread(target=sumo_loop, daemon=True)
    sumo_thread.start()

    for _ in range(40):
        if simulation_running:
            break
        time.sleep(0.2)

    if simulation_running:
        synced = len([acc for acc in accidents if acc["id"] in _sumo_accident_scenes])
        return jsonify({
            "status": "✅ SUMO démarré",
            "config": SUMO_CONFIG,
            "synced_accidents": synced,
        })
    global vision_worker
    if vision_worker:
        vision_worker.set_traci_ready(True)

    return jsonify({"status": "❌ SUMO n'a pas démarré (vérifiez le .sumocfg)"}), 500

@app.route("/start_vision")
def start_vision():
    global vision_worker

    if vision_worker is None:
        vision_worker = VisionWorker(
           model_path=str(BASE_DIR / "vision_ai" / "best.pt"),
           backend_url="http://localhost:5000"
)
        vision_worker.start()
        return jsonify({"status": "👁 YOLO started"})

    return jsonify({"status": "⚠ YOLO already running"})

@app.route("/stop_vision")
def stop_vision():
    global vision_worker

    if vision_worker:
        vision_worker.stop()
        vision_worker = None

    return jsonify({"status": "🛑 YOLO stopped"})
@app.route("/stop")
def stop():
    global simulation_running
    simulation_running = False
    return jsonify({"status": "🛑 Arrêt SUMO en cours..."})


@app.route("/sumo_status")
def sumo_status():
    return jsonify({
        "running":  simulation_running,
        "sim_time": sumo_sim_time,
        "vehicles": sumo_vehicle_count,
    })


@app.route("/reset")
def reset():
    global accidents, evaluation_history
    # Supprimer toutes les scènes d'accidents dans SUMO
    for acc in accidents:
        _remove_accident_scene(acc["id"])
    accidents = []
    for amb in ambulances:
        amb.update({
            "status": "available", "eta": None, "history": [],
            "missions": 0, "distance_travelled": 0.0, "times": [],
        })
    for h in hospitals:
        h["current_load"] = random.randint(5, h["capacity"] // 2)
    statistics.update({
        "total_interventions": 0, "completed": 0,
        "total_distance": 0.0,   "total_time": 0.0,
        "interventions_per_hour": {}
    })
    evaluation_history = []
    socketio.emit("reset")
    return jsonify({"status": "♻ Système réinitialisé"})

def handle_event(event):
    print("💥 EVENT YOLO REÇU :", event)

# ============================
# MAIN
# ============================
if __name__ == "__main__":

    # 🚀 Démarrage automatique YOLO
    try:
        MODEL_PATH = str(BASE_DIR / "vision_ai" / "best.pt")

        vision_worker = VisionWorker(
         model_path=MODEL_PATH,
         callback=handle_event,
         backend_url="http://127.0.0.1:5000"
)
       
        vision_worker.start()
        print("👁 YOLO démarré automatiquement")
    except Exception as e:
        print(f"❌ Erreur démarrage YOLO : {e}")


    
    # 🚑 Background IoT
    socketio.start_background_task(iot_background)

    # 🌐 Flask
    socketio.run(app, debug=False, host="0.0.0.0", port=5000)
