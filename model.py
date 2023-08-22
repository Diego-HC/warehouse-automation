from __future__ import annotations

from enum import IntEnum
from typing import Optional, Union, Any, Dict, List, Tuple

from mesa.model import Model
from mesa.agent import Agent
# from mesa.space import MultiGrid
# from mesa.time import SimultaneousActivation
# from mesa.datacollection import DataCollector

import numpy as np

class MSG(IntEnum):
    """Robot Messages"""
    NEW_TASK = 0
    TOOK_TASK = 1
    NEW_PALLET = 2
    TOOK_PALLET = 3
    CHARGING_STATION = 4

class RS(IntEnum):
    """Robot State"""
    IDLE = 0
    MOVING_TO_PALLET = 1
    MOVING_TO_DESTINATION = 2
    CHARGING = 3


class Product(IntEnum):
    WATER = 0
    FOOD = 1
    MEDICINE = 2


class Item:
    def __init__(self, name: str, product: Product) -> None:
        self.name = name
        self.product = product


class Pallet(Agent):
    def __init__(self, unique_id: int, model: Model, product: Product, weight: int = 1) -> None:
        super().__init__(unique_id, model)

        self.weight = weight
        self.product = product

        self.in_robot = False


class ChargingStation(Agent):
    def __init__(self, unique_id: int, model: Model) -> None:
        super().__init__(unique_id, model)


class Task:
    def __init__(
            self, unique_id: int, model: Model, product: Product,
            destination: (int, int) = None, start=None) -> None:
        super().__init__(unique_id, model)

        self.product = product
        # If start is None, the robot will pick up the pallet from the storage
        self.start = start
        # If destination is None, the robot will drop off the pallet in the storage
        self.destination = destination


class Robot(Agent):
    def __init__(
            self, unique_id: int, model: Model, speed: int = 1,
            weight_capacity: int = 10, charging_rate: int = 15) -> None:

        """
        :param unique_id:
        :param model:
        :param speed:
        :param weight_capacity:
        :param charging_rate:
        """

        super().__init__(unique_id, model)

        self.speed = speed
        self.weightCapacity = weight_capacity

        self.state: RS = RS.IDLE
        self.next_pos = None

        self.tasks: [Task] = []
        self.current_task: Optional[Task] = None
        self.path = []
        self.pallets: [Pallet] = []
        self.available_pallets: [Pallet] = []
        self.available_storage: [(int, int)] = []

        self.battery = 100
        self.charging_rate = charging_rate
        self.charging_stations: [ChargingStation] = []

        self.messages = []

    def step(self) -> None:
        if self.battery <= 0:
            return
        
        self.read_messages()
        self.find_charging_stations()

        if self.state == RS.IDLE:
            self.find_best_task()
            if self.current_task is not None:
                self.state = RS.MOVING_TO_PALLET
                if self.current_task.start is None:
                    # Find closest pallet 
                    self.path = self.find_closest_pallet(self.current_task.product)
                else:
                    self.path = self.find_path(self.current_task.start)

        elif self.state == RS.MOVING_TO_PALLET:
            if self.next_pos is None:
                self.load_pallet()
                self.state = RS.MOVING_TO_DESTINATION
                if self.current_task.destination is None:
                    # Find the closest storage space
                    self.path = self.find_closest_storage()
                else:
                    self.path = self.find_path(self.current_task.destination)

        elif self.state == RS.MOVING_TO_DESTINATION:
            if self.next_pos is None:
                self.unload_pallet()
                self.state = RS.IDLE
                self.current_task = None
                self.path = []
                return

        elif self.state == RS.CHARGING:
            if self.battery >= 90:
                self.state = RS.IDLE
                return

        # TODO: Consider using the list as a stack instead of a queue
        self.next_pos = self.path.pop(0)

    def advance(self) -> None:
        if self.battery <= 0:
            return
        
        charging_station = self.model.grid.get_cell_list_contents([self.pos])
        if any(isinstance(station, ChargingStation) for station in charging_station):
            self.recharge()
        else:
            if self.battery > 0:
                self.battery -= 1

        if self.next_pos is not None:
            for pallet in self.pallets:
                self.model.grid.move_agent(pallet, self.next_pos)
            
            self.model.grid.move_agent(self, self.next_pos)
            self.next_pos = None 

    def send_message(self, subject: str, data: Any) -> None:
        self.messages.append((subject, data))

    def broadcast_message(self, subject: str, data: Any) -> None:
        for robot in self.model.schedule.agents:
            if robot != self and isinstance(robot, Robot):
                robot.send_message(subject, data) 

    def read_messages(self) -> None:
        while len(self.messages) > 0:
            message = self.messages.pop(0)
            if message[0] == MSG.CHARGING_STATION and message[1] not in self.charging_stations:
                self.charging_stations.append(message[1])
            elif message[0] == MSG.NEW_TASK:
                self.tasks.append(message[1])
            elif message[0] == MSG.TOOK_TASK:
                self.tasks.remove(message[1])
                # ! This may not work
                if self.current_task == message[1]:
                    self.current_task = None
            elif message[0] == MSG.NEW_PALLET:
                self.available_pallets.append(message[1])
            elif message[0] == MSG.TOOK_PALLET:
                self.available_pallets.remove(message[1])

    def find_best_task(self) -> None:
        if len(self.tasks) > 0:
            # TODO: Implement a better task selection algorithm
            self.current_task = self.tasks[0]
            self.tasks.remove(self.current_task)
            self.send_message(MSG.TOOK_TASK, self.current_task)

    def find_charging_stations(self) -> None:
        charging_stations = self.model.grid.get_neighbors(self.pos, moore=False, radius=2, include_center=False)

        for station in charging_stations:
            if isinstance(station, ChargingStation) and station not in self.charging_stations:
                self.charging_stations.append(station)
                self.broadcast_message("charging_station", station.pos)

    @staticmethod
    def lowest_cost_tile(non_visited_positions, costs) -> Optional[(int, int)]:
        if not non_visited_positions:
            return None
        return min(non_visited_positions, key=lambda position: costs[position])

    def find_paths(self, positions) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """
        Calculates the shortest path from the current position to the indicated
        position. Returns a list of positions representing the path.
        """

        # noinspection PyPEP8Naming
        Obstacle = Union[Pallet, ChargingStation, Robot]

        # Djisktra in python
        non_visited_positions = [
            pos for agent, pos in self.model.grid.coord_iter()
            if not isinstance(agent, Obstacle) and pos not in self.available_storage]

        positions_copy = positions.copy()
        costs = {pos: np.inf for pos in non_visited_positions}
        costs[self.pos] = 0
        current_pos = self.pos

        while len(non_visited_positions) > 0:
            # print(current_pos)
            neighbors = self.model.grid.get_neighbors(
                current_pos, moore=False, include_center=False)
            for neighbor in neighbors:
                if isinstance(neighbor, Obstacle) or neighbor.pos in self.available_storage:
                    continue
                if costs[neighbor.pos] > costs[current_pos] + 1:
                    costs[neighbor.pos] = costs[current_pos] + 1
            non_visited_positions.remove(current_pos)

            if current_pos in positions_copy:
                positions_copy.remove(current_pos)

                if len(positions_copy) == 0:
                    break
            current_pos = self.lowest_cost_tile(non_visited_positions, costs)
            if current_pos is None:
                break

        paths = {}

        for pos in positions:
            path = []
            current_pos = pos
            while current_pos != self.pos:
                path.append(current_pos)
                neighbors = self.model.grid.get_neighbors(
                    current_pos, moore=False, include_center=False)
                for neighbor in neighbors:
                    if costs[neighbor.pos] == costs[current_pos] - 1:
                        current_pos = neighbor.pos
                        break
            paths[pos] = path

        return paths

    def find_path(self, position) -> List[Tuple[int, int]]:
        return self.find_paths([position])[position]

    def find_closest_pallet(self, product) -> [Tuple[int, int]]:
        paths = self.find_paths([pallet.pos for pallet in self.available_pallets if pallet.product == product])

        return min(paths, key=len)

    def find_closest_storage(self) -> [Tuple[int, int]]:
        paths = self.find_paths([storage for storage in self.available_storage])

        return min(paths, key=len)

    def load_pallet(self) -> None:
        pallets = self.model.grid.get_cell_list_contents([self.pos])
        for pallet in pallets:
            if isinstance(pallet, Pallet) and not pallet.in_robot:
                self.pallets.append(pallet)
                pallet.in_robot = True
                break
        
    def unload_pallet(self) -> None:
        pallet = self.pallets.pop()
        pallet.in_robot = False

    def recharge(self) -> None:
        self.battery = min(100, self.battery + self.charging_rate)
