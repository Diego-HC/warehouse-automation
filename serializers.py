from typing import Dict, Any, Tuple
from mesa import Agent
from model import Warehouse, Robot, Pallet


def serialize_pallet(pallet: Pallet) -> Dict[str, Any]:
    return {
        "unique_id": pallet.unique_id,
        "pos": list(pallet.pos),
        "product": ["water", "food", "medicine"][pallet.product]
    }


def serialize_robot(robot: Robot) -> Dict[str, Any]:
    return {
        "unique_id": robot.unique_id,
        "pos": list(robot.pos),
        "speed": robot.speed,
        "pallets": [serialize_pallet(pallet) for pallet in robot.pallets]
    }


def serialize_generic_agent(agent: Agent) -> Dict[str, Any]:
    resp = {
        "unique_id": agent.unique_id,
        "pos": list(agent.pos)
    }
    try:
        # noinspection PyUnresolvedReferences
        resp["direction"] = ["up", "down", "left", "right"][agent.direction]
    except AttributeError:
        pass
    return resp


def map_to_json(model: Warehouse) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    serialized_agents = [
        "Robot", "Storage", "Spawner", "Despawner",
        "Palletizer", "ConveyorBelt", "ChargingStation"]
    details = {}
    counts = {}
    for agent in model.schedule.agents:
        class_name = agent.__class__.__name__
        if class_name in serialized_agents:
            key = class_name.lower()
            try:
                details[key].append(serialize_generic_agent(agent))
            except KeyError:
                details[key] = [serialize_generic_agent(agent)]

    for key in details:
        counts[key + "_count"] = len(details[key])
    return counts, details


def current_frame_to_json(model: Warehouse) -> Dict[str, Any]:
    robots = []
    other_pallets = []
    for agent in model.schedule.agents:
        if isinstance(agent, Robot):
            robots.append(serialize_robot(agent))
        elif isinstance(agent, Pallet) and not agent.in_robot:
            other_pallets.append(serialize_pallet(agent))
    return {
        "robots": robots,
        "other_pallets": other_pallets
    }
