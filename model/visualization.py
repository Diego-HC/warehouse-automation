from typing import Dict, Any

from mesa import Agent
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid

from .warehouse import Warehouse
from .items import Item, Pallet, Storage
from .robot import Robot
from .stationary import ConveyorBelt, ChargingStation, Spawner, Despawner, Palletizer


def portrayal(agent: Agent) -> Dict[str, Any]:
    if isinstance(agent, Item):
        product_name = ["W", "F", "M"][agent.product]
        return {"Shape": "circle", "Filled": "false", "Color": "Red", "Layer": 2, "r": 0.3,
                "text": f"{product_name}", "text_color": "white"}
    if isinstance(agent, Pallet):
        product_name = ["W", "F", "M"][agent.product]
        return {"Shape": "rect", "Filled": "false", "Color": "Red", "Layer": 2, "w": 0.6, "h": 0.6,
                "text": f"{product_name}", "text_color": "white"}
    if isinstance(agent, Storage):
        return {"Shape": "rect", "Filled": "true", "Color": "brown", "Layer": 1,
                "w": 0.9, "h": 0.9}
    if isinstance(agent, Robot):
        return {"Shape": "circle", "Filled": "false", "Color": "Cyan", "Layer": 1, "r": 0.9,
                "text": f"{agent.battery}", "text_color": "black"}
    if isinstance(agent, ConveyorBelt):
        return {"Shape": "rect", "Filled": "true", "Color": "black", "Layer": 0,
                "w": 0.9, "h": 0.9}
    if isinstance(agent, ChargingStation):
        return {"Shape": "rect", "Filled": "true", "Color": "green", "Layer": 0,
                "w": 0.9, "h": 0.9}
    if isinstance(agent, Spawner):
        return {"Shape": "rect", "Filled": "true", "Color": "blue", "Layer": 0,
                "w": 0.9, "h": 0.9}
    if isinstance(agent, Despawner):
        return {"Shape": "rect", "Filled": "true", "Color": "red", "Layer": 0,
                "w": 0.9, "h": 0.9}
    if isinstance(agent, Palletizer):
        return {"Shape": "rect", "Filled": "true", "Color": "gray", "Layer": 0,
                "w": 0.9, "h": 0.9}


grid = CanvasGrid(portrayal, 15, 15, 400, 400)

server = ModularServer(
    Warehouse,
    [grid],
    "Warehouse Automation",
    {
        "width": 15,
        "height": 15,
        "num_robots": 5,
        "num_spawners": 3,
        "num_despawners": 3,
    }
)
