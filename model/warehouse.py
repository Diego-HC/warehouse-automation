from __future__ import annotations

from typing import Optional, Any, List, Tuple

from mesa.model import Model
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector

import numpy as np

from .enums import Dir, Product, Msg
from .items import Pallet, Task, Storage
from .robot import Robot
from .stationary import ChargingStation, Spawner, Despawner, ConveyorBelt, Palletizer

STORAGE_POSITIONS = [
    (5, 3), (6, 3), (7, 3),
    (5, 7), (6, 7), (7, 7),
    (5, 11), (6, 11), (7, 11),
    (9, 5), (10, 5), (11, 5),
    (9, 9), (10, 9), (11, 9)
]

STATION_POSITIONS = [
    (10, 2), (12, 2), (14, 2)
]

SPAWNER_POSITIONS = [
    (3, 5), (3, 7), (3, 9)
]

DESPAWNER_POSITIONS = [
    (1, 5), (1, 7), (1, 9)
]

CONVEYOR_BELT_POSITIONS = [
    (12, 13), (13, 13), (14, 13)
]

PALLETIZER_POSITIONS = [
    (11, 13)
]

ITEM_SPAWNER_POSITIONS = [
    (14, 13)
]


FULL_SIMULATION = False


class Warehouse(Model):
    def __init__(
        self,
        width: int,
        height: int,
        num_robots: int,
        num_spawners: int,
        num_despawners: int,
    ) -> None:
        super().__init__()

        self.width = width
        self.height = height
        self.num_robots = num_robots
        self.num_spawners = num_spawners
        self.num_despawners = num_despawners

        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = SimultaneousActivation(self)
        self.open_spaces = []
        for x in range(self.width):
            for y in range(self.height):
                self.open_spaces.append((x, y))

        self.storage: List[Storage] = []
        self.charging_stations: List[ChargingStation] = []

        self.robots: List[Robot] = []
        self.pallets: List[Pallet] = []
        self.tasks: List[Task] = []

        self.datacollector = DataCollector(
            model_reporters={
                "Battery": lambda m: np.mean([robot.battery for robot in m.robots]),
                "Tasks": lambda m: len(
                    [robot for robot in m.robots if robot.current_task is not None]
                ),
                "Pallets": lambda m: len(
                    [robot for robot in m.robots if len(robot.pallets) > 0]
                ),
                "Charging Stations": lambda m: len(
                    [robot for robot in m.robots if len(robot.charging_stations) > 0]
                ),
            },
            # agent_reporters={
            #     "Battery": lambda a: a.battery,
            #     "State": lambda a: a.state,
            #     "Tasks": lambda a: len(a.tasks),
            #     "Pallets": lambda a: len(a.pallets),
            #     "Charging Stations": lambda a: len(a.charging_stations)
            # }
        )

        self.running = True

        self.generate_static_warehouse()

    def filter_grid(self) -> None:
        for agents, pos in self.grid.coord_iter():
            if agents:
                self.open_spaces.remove(pos)

    def generate_static_warehouse(self) -> None:
        # Create storage in specific locations
        for pos in STORAGE_POSITIONS:
            s = Storage(self.next_id(), self, Dir.DOWN, pos)
            self.schedule.add(s)
            self.grid.place_agent(s, pos)
            self.storage.append(s)

        # Create charging stations in specific locations
        for pos in STATION_POSITIONS:
            c = ChargingStation(self.next_id(), self)
            self.schedule.add(c)
            self.grid.place_agent(c, pos)
            self.charging_stations.append(c)

        # Create spawners in specific locations
        for i, pos in enumerate(SPAWNER_POSITIONS):
            s = Spawner(self.next_id(), self, [Product.all()[i]])
            self.schedule.add(s)
            self.grid.place_agent(s, pos)

        # Create despawners in specific locations
        for i, pos in enumerate(DESPAWNER_POSITIONS):
            d = Despawner(self.next_id(), self, [Product.all()[i]])
            self.schedule.add(d)
            self.grid.place_agent(d, pos)

        if FULL_SIMULATION:
            for pos in CONVEYOR_BELT_POSITIONS:
                c = ConveyorBelt(self.next_id(), self, pos, Dir.LEFT)
                self.schedule.add(c)
                self.grid.place_agent(c, pos)

            for pos in PALLETIZER_POSITIONS:
                p = Palletizer(self.next_id(), self, pos, 4, 1, Dir.LEFT)
                self.schedule.add(p)
                self.grid.place_agent(p, pos)

            # Create spawners in specific locations
            for pos in ITEM_SPAWNER_POSITIONS:
                s = Spawner(self.next_id(), self, Product.all(), spawns_items=True)
                self.schedule.add(s)
                self.grid.place_agent(s, pos)

        self.filter_grid()

        # Create robots in specific locations
        self.robots = []
        for _ in range(self.num_robots):
            r = Robot(
                self.next_id(),
                self,
                storage=self.storage,
                charging_stations=self.charging_stations,
            )
            self.robots.append(r)
            self.schedule.add(r)

            idx = np.random.randint(0, len(self.open_spaces))
            self.grid.place_agent(r, self.open_spaces[idx])
            self.open_spaces.pop(idx)

    def step(self) -> None:
        self.datacollector.collect(self)
        self.schedule.step()

    def broadcast_message(self, subject: Msg, data: Any) -> None:
        for robot in self.robots:
            robot.send_message(subject, data)

    def create_task(
        self,
        product: Product,
        start: Optional[Tuple[int, int]] = None,
        destination: Optional[Tuple[int, int]] = None,
        id_: Optional[str] = "",
    ) -> None:
        task = Task(product, start, destination, id_)
        self.broadcast_message(Msg.NEW_TASK, task)

    def get_pallets(self) -> List[Pallet]:
        pallets = []
        pos_set = set()

        for agents, pos in self.grid.coord_iter():
            # noinspection PyTypeChecker
            pallets.extend([agent for agent in agents if isinstance(agent, Pallet)])

            if pos in pos_set:
                print(f"-----Error, multiple pallets on {pos}-----")

        return pallets


# if __name__ == "__main__":
#     STEPS = 1000
#
#     WIDTH = 15
#     HEIGHT = 15
#
#     NUM_ROBOTS = 3
#     NUM_SPAWNERS = 3
#     NUM_DESPAWNERS = 3
#
#     def batch_run(iterations: int, steps: int):
#         for _ in range(iterations):
#             w = Warehouse(WIDTH, HEIGHT, NUM_ROBOTS, NUM_SPAWNERS, NUM_DESPAWNERS)
#
#             for _ in range(steps):
#                 w.step()
#
#     if __name__ == "__main__":
#         # batch_run(100, 100)
#
#         m = Warehouse(WIDTH, HEIGHT, NUM_ROBOTS, NUM_SPAWNERS, NUM_DESPAWNERS)
#         pallets = []
#         tasks = []
#
#         for i in range(STEPS):
#             print(f"\n-----Step {i}-----\n")
#             print(f"Pallets: {pallets}")
#             print(f"Tasks: {tasks}\n")
#             m.step()
#
#             pallets = m.get_pallets()
#             for agent in m.schedule.agents:
#                 if isinstance(agent, Robot):
#                     tasks = agent.tasks
#                     break
