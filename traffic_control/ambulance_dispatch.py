import traci
import random

from traffic_control.smart_lights import set_green_corridor
from traffic_control.route_finder import calculate_route
from traffic_control.route_finder import get_random_edge


def dispatch_ambulance(accident_edge):

    ambulance_id = f"ambulance_{random.randint(1,9999)}"

    try:

        # Point départ ambulance
        start_edge = get_random_edge()

        # Calcul route dynamique
        route_edges = calculate_route(
            start_edge,
            accident_edge
        )

        route_id = f"route_{ambulance_id}"

        # Ajouter route
        traci.route.add(
            route_id,
            route_edges
        )

        # Ajouter ambulance
        traci.vehicle.add(
            vehID=ambulance_id,
            routeID=route_id,
            typeID="ambulance"
        )

        # Couleur rouge
        traci.vehicle.setColor(
            ambulance_id,
            (255, 0, 0)
        )

        # Priorité
        traci.vehicle.setSpeedMode(
            ambulance_id,
            0
        )

        print(f"[AMBULANCE] {ambulance_id} dispatched")

        # Feux intelligents
        set_green_corridor(ambulance_id)

    except Exception as e:
        print(e)