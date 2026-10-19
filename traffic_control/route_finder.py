import traci
import random


def get_random_edge():

    edges = traci.edge.getIDList()

    valid_edges = []

    for edge in edges:

        # Ignorer edges internes
        if not edge.startswith(":"):

            valid_edges.append(edge)

    return random.choice(valid_edges)


def calculate_route(start_edge, destination_edge):

    route = traci.simulation.findRoute(
        start_edge,
        destination_edge
    )

    return route.edges