import traci


def set_green_corridor(ambulance_id):

    """
    Active corridor vert
    """

    tls_list = traci.trafficlight.getIDList()

    for tls_id in tls_list:

        try:
            traci.trafficlight.setRedYellowGreenState(
                tls_id,
                "GGGG"
            )

            print(f"[SMART LIGHT] Green light activated: {tls_id}")

        except Exception as e:
            print(e)