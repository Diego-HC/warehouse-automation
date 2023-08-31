from typing import Dict, Any
from model import Warehouse, Robot, Pallet


def serialize_pallet(pallet: Pallet) -> Dict[Any, Any]:
    return {
        "unique_id": pallet.unique_id,
        "pos": [coord for coord in pallet.pos],
        "product": ["water", "food", "medicine"][pallet.product]
    }


def serialize_robot(robot: Robot) -> Dict[Any, Any]:
    return {
        "unique_id": robot.unique_id,
        "pos": [coord for coord in robot.pos],
        "speed": robot.speed,
        "pallets": [serialize_pallet(pallet) for pallet in robot.pallets]
    }


def current_frame_to_json(model: Warehouse) -> Dict[Any, Any]:
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
