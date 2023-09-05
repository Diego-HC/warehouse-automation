from __future__ import annotations

from typing import Optional, Any, Dict, List, Tuple, TYPE_CHECKING

from mesa.agent import Agent
import numpy as np

from .enums import Product, Msg, RS
from .items import Pallet, Storage, Task
from .stationary import ChargingStation


if TYPE_CHECKING:
    from .warehouse import Warehouse


class Robot(Agent):
    """
    A robot agent that can move around the warehouse, pick up and deliver pallets, and charge its battery.

    Attributes:
        unique_id : int
            A unique identifier for the robot.
        model : Model
            The model that the robot belongs to.
        speed : int
            The speed of the robot in grid cells per step.
        weight_capacity : int
            The maximum weight that the robot can carry.
        charging_rate : int
            The rate at which the robot's battery charges per step.

    Properties:
        current_task (Optional[Task]): The current task that the robot is working on.

    Methods:
        step(): Advances the robot by one step in its current state.
        advance(): Advances the robot by one step in its current position.
        send_message(subject: str, data: Any): Sends a message to another robot.
        broadcast_message(subject: str, data: Any): Broadcasts a message to all other robots.
        read_messages(): Reads and processes all messages in the robot's message queue.
        find_best_task(): Finds the best task for the robot to work on.
        find_charging_stations(): Finds all charging stations around the robot.
    """

    def __init__(
        self,
        unique_id: int,
        model: Warehouse,
        storage: List[Storage],
        speed: int = 1,
        weight_capacity: int = 10,
        charging_rate: int = 15,
        charging_stations: List[ChargingStation] = None,
    ) -> None:
        """
        Hola
        """

        super().__init__(unique_id, model)
        self.model: Warehouse = model

        self.speed = speed
        self.weight_capacity = weight_capacity

        self.state: RS = RS.IDLE
        self.next_pos = None

        self.tasks: List[Task] = []
        self._current_task: Optional[Task] = None
        self.path = []
        self.next_path = []
        self.pallets: List[Pallet] = []
        self.available_pallets: List[Pallet] = []
        self.storage: List[Storage] = storage

        self.battery = 100
        self.charging_rate = charging_rate
        self.charging_stations: List[ChargingStation] = charging_stations if charging_stations is not None else []

        self.messages = []
        self.costs = {}
        self.path_cache = {}

    @property
    def current_task(self) -> Optional[Task]:
        return self._current_task

    @current_task.setter
    def current_task(self, task: Optional[Task]) -> None:
        self._current_task = task
        if task is not None:
            task.robot = self

    def step(self) -> None:
        if self.battery <= 0:
            self.print_robot_data("Battery depleted")
            return

        self.print_robot_data(
            f"{self.pos}, {self.state}, {self.available_pallets}"
        )
        self.read_messages()
        self.find_charging_stations()

        charging_station_path = self.find_closest_charging_station()

        if self.state == RS.MOVING_TO_STATION and not self.path:
            self.state = RS.CHARGING
            return

        elif (
            charging_station_path is not None
            and self.battery - len(charging_station_path) <= 30
            and self.state == RS.IDLE
        ):
            self.print_robot_data("---Moving to station---")
            self.state = RS.MOVING_TO_STATION
            # print(charging_station_path)
            self.path = charging_station_path

        elif self.state == RS.IDLE:
            self.find_best_task()
            if self.current_task is not None:
                self.print_robot_data(f"took task {self.current_task}")
                self.state = RS.MOVING_TO_PALLET
                if self.current_task.start is None:
                    # Find closest pallet
                    self.path, pallet = self.find_closest_pallet(
                        self.current_task.product
                    )
                    pallet.robot = self
                    self.broadcast_message(Msg.TOOK_PALLET, pallet)
                else:
                    self.path = self.find_path(self.current_task.start)
                    pallet = self.model.grid.get_cell_list_contents(
                        [self.current_task.start]
                    )[0]
                    pallet.robot = self

                    temp_pos = self.pos
                    self.pos = self.current_task.start
                    self.next_path = self.find_closest_storage()
                    self.pos = temp_pos
                    self.print_robot_data(f"{self.next_path}")
                    self.get_storage(self.next_path[0]).is_available = False
            else:
                return

        elif self.state == RS.MOVING_TO_PALLET and not self.path:
            if self.get_storage(self.pos) is not None:
                # The robot arrived at a storage agent
                self.load_pallet()
                self.next_pos = self.get_storage(self.pos).entry_pos
                self.state = RS.MOVING_TO_ENTRY_POINT
                self.get_storage(self.pos).is_available = True
                return
            else:
                # The robot arrived at a spawner agent
                self.load_pallet()
                self.state = RS.MOVING_TO_DESTINATION
                if self.current_task.destination is None:
                    self.path = self.next_path
                    self.next_path = []
                else:
                    self.path = self.find_path(self.current_task.destination)

        elif self.state == RS.MOVING_TO_ENTRY_POINT:
            self.state = RS.MOVING_TO_DESTINATION
            self.path = self.find_path(self.current_task.destination)

        elif self.state == RS.MOVING_TO_DESTINATION and not self.path:
            self.state = RS.IDLE
            self.path = []

            if self.current_task.destination is None:
                self.print_robot_data(f"unloading pallet")
                self.broadcast_message(Msg.NEW_PALLET, self.pallets[-1])
                self.unload_pallet()
                self.next_pos = self.get_storage(self.pos).entry_pos
                self.current_task = None
                return

            self.unload_pallet()

            self.current_task = None
            return

        elif self.state == RS.CHARGING:
            if self.battery >= 90:
                self.state = RS.IDLE
            return

        if self.path:
            self.next_pos = self.path.pop()

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
                # if pallet.pos is None:
                #     self.pallets.remove(pallet)
                # else:
                self.model.grid.move_agent(pallet, self.next_pos)

            self.model.grid.move_agent(self, self.next_pos)
            self.next_pos = None
            self.path_cache = {}

    def send_message(self, subject: Msg, data: Any) -> None:
        self.messages.append((subject, data))

    def broadcast_message(self, subject: Msg, data: Any) -> None:
        for robot in self.model.schedule.agents:
            if isinstance(robot, Robot):
                robot.send_message(subject, data)

    def read_messages(self) -> None:
        while len(self.messages) > 0:
            message = self.messages.pop(0)
            if (
                message[0] == Msg.CHARGING_STATION
                and message[1] not in self.charging_stations
            ):
                self.charging_stations.append(message[1])

            elif message[0] == Msg.NEW_TASK:
                self.tasks.append(message[1])

            elif message[0] == Msg.NEW_PALLET:
                self.print_robot_data(
                    f"adding pallet {message[1]} to available pallets list ({self.available_pallets})"
                )
                self.available_pallets.append(message[1])
            elif message[0] == Msg.TOOK_PALLET:
                self.print_robot_data(
                    f"removing pallet {message[1]} from available pallets list ({self.available_pallets})"
                )
                self.available_pallets.remove(message[1])

    def get_storage(self, pos: Tuple[int, int]) -> Optional[Storage]:
        for storage in self.storage:
            if storage.pos == pos:
                return storage

    def filter_tasks(self) -> None:
        self.tasks = [task for task in self.tasks if task.robot is None]

    def has_available_pallet(self, product: Product):
        return (
            True
            if [
                pallet for pallet in self.available_pallets if pallet.product == product
            ]
            else False
        )

    def find_best_task(self) -> None:
        self.filter_tasks()

        if len(self.tasks) > 0:
            # Find the closest task
            if any(storage.is_available for storage in self.storage):
                paths_storage = {
                    task: self.find_path(task.start)
                    for task in self.tasks
                    if task.start is not None  # and task.robot is None
                }
            else:
                self.print_robot_data(f"Skipping storage tasks")
                paths_storage = {}

            paths_dropoff = {
                task: self.find_path(task.destination)
                for task in self.tasks
                if task.destination is not None
                and self.has_available_pallet(task.product)  # and task.robot is None
            }

            if len(paths_storage) == 0 and len(paths_dropoff) == 0:
                return
            if len(paths_storage) == 0:
                self.current_task = min(
                    paths_dropoff, key=lambda task: len(paths_dropoff[task])
                )
                return
            if len(paths_dropoff) == 0:
                self.current_task = min(
                    paths_storage, key=lambda task: len(paths_storage[task])
                )
                return

            min_storage = min(paths_storage, key=lambda task: len(paths_storage[task]))
            min_dropoff = min(paths_dropoff, key=lambda task: len(paths_dropoff[task]))

            if len(paths_storage[min_storage]) < len(paths_dropoff[min_dropoff]):
                self.current_task = min_storage
            else:
                self.current_task = min_dropoff

    def find_charging_stations(self) -> None:
        charging_stations = self.model.grid.get_neighbors(
            self.pos, moore=False, radius=2, include_center=False
        )

        for station in charging_stations:
            if (
                isinstance(station, ChargingStation)
                and station not in self.charging_stations
            ):
                self.charging_stations.append(station)
                self.broadcast_message(Msg.CHARGING_STATION, station.pos)

    def find_closest_charging_station(self) -> Optional[List[Tuple[int, int]]]:
        if len(self.charging_stations) > 0:
            paths = self.find_paths([station.pos for station in self.charging_stations])
            charging_station_pos = min(paths, key=lambda pos: len(paths[pos]))
            return paths[charging_station_pos]
        return None

    def find_path(self, position) -> List[Tuple[int, int]]:
        if self.path_cache != {}:
            return self.path_cache[position]

        return self.find_paths([position])[position]

    def find_closest_pallet(self, product) -> Tuple[List[Tuple[int, int]], Pallet]:
        paths = {
            pallet: self.find_path(pallet.pos)
            for pallet in self.available_pallets
            if pallet.product == product and pallet.robot is None
        }
        pallet = min(paths, key=lambda pallet: len(paths[pallet]))
        return paths[pallet], pallet

    def find_closest_storage(self) -> List[Tuple[int, int]]:
        storage_destinations = [storage.entry_pos for storage in self.storage if storage.is_available]
        paths = self.find_paths(storage_destinations)

        storage_entry_pos = min(paths, key=lambda pos: len(paths[pos]))
        path = paths[storage_entry_pos]
        path.insert(
            0,
            [
                storage.pos
                for storage in self.storage
                if storage_entry_pos == storage.entry_pos
            ][0],
        )

        return path

    def load_pallet(self) -> None:
        pallets = self.model.grid.get_cell_list_contents([self.pos])
        for pallet in pallets:
            if isinstance(pallet, Pallet) and not pallet.in_robot:
                self.print_robot_data(f"loaded pallet {pallet.unique_id} at {self.pos}")
                self.pallets.append(pallet)
                pallet.in_robot = True
                break
        else:
            self.print_robot_data(f"could not load pallet")

    def unload_pallet(self) -> None:
        pallet = self.pallets.pop()
        self.print_robot_data(f"unloaded pallet {pallet.unique_id} at {self.pos}")
        pallet.in_robot = False
        pallet.robot = None

    def recharge(self) -> None:
        self.battery = min(100, self.battery + self.charging_rate)

    def __repr__(self) -> str:
        return f"<Robot id={self.unique_id}, state={self.state}, battery={self.battery}, pos={self.pos}>"

    def print_robot_data(self, data: str) -> None:
        print(f"Robot {self.unique_id} -> {data}")

    @staticmethod
    def lowest_cost_tile(non_visited_positions, costs) -> Optional[Tuple[int, int]]:
        if not non_visited_positions:
            return None
        return min(non_visited_positions, key=lambda position: costs[position])

    @staticmethod
    def is_obstacle(agents: List[Agent]) -> bool:
        return any([any([
            isinstance(agent, Storage),
            isinstance(agent, ChargingStation),
            # isinstance(agent, Robot),
            isinstance(agent, Storage),
        ]) for agent in agents])

    # def is_storage(self, pos: Tuple[int, int]) -> bool:
    #     for storage in self.available_storage:
    #         print(storage)
    #         if storage.pos == pos:
    #             return True
    #
    #     for pallet in self.available_pallets:
    #         if pallet.pos == pos:
    #             return True

    def find_paths(self, positions) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """
        Calculates the shortest path from the current position to the indicated
        position. Returns a list of positions representing the path.
        """

        # TODO: Update function to account for storage, since robots can only enter from one direction

        if self.path_cache != {}:
            print("Using cache")
            return self.path_cache
        if len(positions) == 0:
            return {}

        # Djisktra in python
        non_visited_positions = [
            pos
            for agents, pos in self.model.grid.coord_iter()
            if not self.is_obstacle(agents)
            # and not self.is_storage(pos)
            or pos in positions
        ]
        non_visited_positions.append(self.pos)

        non_obstacle_positions = non_visited_positions.copy()
        # positions_copy = positions.copy()
        costs = {pos: np.inf for pos in non_visited_positions}
        costs[self.pos] = 0
        current_pos = self.pos

        while len(non_visited_positions) > 0:
            # print(current_pos)
            neighborhood = self.model.grid.get_neighborhood(
                current_pos, moore=False, include_center=False
            )

            for neighbor_pos in neighborhood:
                if neighbor_pos in non_obstacle_positions:
                    costs[neighbor_pos] = min(
                        costs[neighbor_pos], costs[current_pos] + 1
                    )

            non_visited_positions.remove(current_pos)

            current_pos = self.lowest_cost_tile(non_visited_positions, costs)
            if current_pos is None:
                break

        paths = {}

        for pos in positions:
            path = []
            current_pos = pos
            while current_pos != self.pos:
                # print(current_pos)
                path.append(current_pos)

                neighborhood = self.model.grid.get_neighborhood(
                    current_pos, moore=False, include_center=False
                )
                for neighbor_pos in neighborhood:
                    if (
                        neighbor_pos in non_obstacle_positions
                        and costs[neighbor_pos] == costs[current_pos] - 1
                    ):
                        current_pos = neighbor_pos
                        break
            paths[pos] = path

        return paths
