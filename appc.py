import os
import random
import threading
import time
from datetime import datetime
from pathlib import Path
from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO
from concurrent.futures import ThreadPoolExecutor
import math
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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

# Accidents SUMO matérialisés par des véhicules immobilisés
_sumo_accident_scenes = {}
_sumo_scene_route_counter = 0

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

            label = "🔵 BLEU→ACCIDENT" if color[2] > 150 else "🟢 VERT→HÔPITAL"
            print(f"  ✅ {amb_id_str} [{label}] → {len(path.edges)} edges ({path.length:.0f}m)")
            return True

    except Exception as e:
        print(f"  ❌ Injection échouée : {e}")
        return False


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


def _remove_accident_scene(accident_id):
    """Supprime la scène d'accident SUMO une fois la mission terminée."""
    if not TRACI_AVAILABLE or not simulation_running:
        return
    veh_ids = _sumo_accident_scenes.pop(accident_id, [])
    if not veh_ids:
        return
    try:
        with sumo_lock:
            current_ids = set(traci.vehicle.getIDList())
            for veh_id in veh_ids:
                if veh_id in current_ids:
                    traci.vehicle.remove(veh_id)
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

# ============================
# ML — PRÉDICTION ETA
# ============================
def _build_ml_model():
    rows = []
    for _ in range(1200):
        distance  = random.uniform(1, 12)
        severity  = random.choice([0, 1, 2])
        hosp_load = random.randint(5, 90)
        hour      = random.randint(0, 23)
        traffic   = 1.6 if (8 <= hour <= 10 or 17 <= hour <= 19) else 1.0
        eta       = round(distance * 5.5 * traffic * (1 + severity * 0.25) * (1 + hosp_load / 120), 2)
        rows.append([distance, severity, hosp_load, hour, eta])
    df  = pd.DataFrame(rows, columns=["distance", "severity", "hospital_load", "hour", "eta"])
    mdl = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
    mdl.fit(df[["distance", "severity", "hospital_load", "hour"]], df["eta"])
    return mdl

rf_model = _build_ml_model()
print("✅ Modèle ML entraîné")

def predict_eta(distance, severity_str, hospital_load):
    hour = datetime.now().hour
    sev  = {"low": 0, "medium": 1, "high": 2}.get(severity_str, 1)
    return round(float(rf_model.predict([[distance, sev, hospital_load, hour]])[0]), 2)

# ============================
# DISPATCH LOGIC
# ============================
def score_ambulance(amb, accident, hospital):
    dist        = haversine(amb["lat"], amb["lng"], accident["lat"], accident["lng"])
    sev_w       = {"low": 1, "medium": 2, "high": 3}[accident["severity"]]
    load_ratio  = hospital["current_load"] / hospital["capacity"]
    hist_factor = 1 / (amb["missions"] + 1)
    return dist * 0.4 + sev_w * 0.3 + load_ratio * 0.2 + hist_factor * 0.1

def best_ambulance(accident):
    available  = [a for a in ambulances if a["status"] == "available"]
    if not available:
        return None, None
    best_amb   = None
    best_hosp  = None
    best_score = float("inf")
    for amb in available:
        for hosp in hospitals:
            s = score_ambulance(amb, accident, hosp)
            if s < best_score:
                best_score = s
                best_amb   = amb
                best_hosp  = hosp
    return best_amb, best_hosp

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
        sumo_phase2 = _inject_ambulance_sumo(amb_id_str2, from_edge2, to_edge2, color=(0, 100, 255, 255))  # 🔵 BLEU → hôpital (même couleur)
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
            amb, hosp = best_ambulance(acc)
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
    })


@app.route("/generate_road_accident")
def generate_road_accident():
    accident = {
        "id":       len(accidents) + 1,
        "lat":      round(31.610 + random.random() * 0.030, 6),
        "lng":      round(-8.010 + random.random() * 0.040, 6),
        "severity": random.choice(["low", "medium", "high"]),
        "status":   "waiting",
    }
    accidents.append(accident)
    socketio.emit("new_accident", accident)
    print(f"💥 Accident #{accident['id']} généré ({accident['severity']})")

    # ── Créer immédiatement une vraie scène d'accident dans SUMO ──
    accident["sumo_synced"] = _spawn_accident_scene(accident)

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
    return jsonify({"status": "❌ SUMO n'a pas démarré (vérifiez le .sumocfg)"}), 500


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
    global accidents
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
    socketio.emit("reset")
    return jsonify({"status": "♻ Système réinitialisé"})


# ============================
# MAIN
# ============================
if __name__ == "__main__":
    socketio.start_background_task(iot_background)
    socketio.run(app, debug=False, host="0.0.0.0", port=5000)
