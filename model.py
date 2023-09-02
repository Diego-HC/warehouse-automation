from __future__ import annotations

from enum import IntEnum
from typing import Optional, Union, Any, Dict, List, Tuple

from mesa.model import Model
from mesa.agent import Agent
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector

import numpy as np


class Msg(IntEnum):
    """Robot Messages"""

    NEW_TASK = 0
    TOOK_TASK = 1
    NEW_PALLET = 2
    TOOK_PALLET = 3
    CHARGING_STATION = 4
    AVAILABLE_STORAGE = 5
    UNAVAILABLE_STORAGE = 6


class RS(IntEnum):
    """Robot State"""

    IDLE = 0
    MOVING_TO_PALLET = 1
    MOVING_TO_DESTINATION = 2
    MOVING_TO_STATION = 3
    CHARGING = 4


class Dir(IntEnum):
    """Robot Direction"""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Product(IntEnum):
    WATER = 0
    FOOD = 1
    MEDICINE = 2


def calc_pos(pos: Tuple[int, int], direction: Dir):
    x, y = pos
    if direction == Dir.UP:
        return (x, y + 1)
        # return pos + (0, 1)
    elif direction == Dir.DOWN:
        return (x, y - 1)
    elif direction == Dir.LEFT:
        return (x - 1, y)
    elif direction == Dir.RIGHT:
        return (x + 1, y)


class Item(Agent):
    def __init__(self, unique_id: int, model: Model, product: Product) -> None:
        super().__init__(unique_id, model)

        self.product = product

    def step(self) -> None:
        pass

    def advance(self) -> None:
        pass

    def __repr__(self) -> str:
        product_name = ["Water", "Food", "Medicine"][self.product]
        return f"<Product id={self.unique_id} name={product_name}>"


class Pallet(Agent):
    def __init__(
        self, unique_id: int, model: Model, product: Product, weight: int = 1
    ) -> None:
        super().__init__(unique_id, model)

        self.weight = weight
        self.product = product

        self.in_robot = False
        self.robot: Optional[Robot] = None

    def step(self) -> None:
        pass

    def advance(self) -> None:
        pass

    def __repr__(self) -> str:
        product_name = ["Water", "Food", "Medicine"][self.product]
        return f"<Pallet id={self.unique_id} name={product_name}, pos={self.pos}>"


class Storage(Agent):
    def __init__(
        self, unique_id: int, model: Model, open_direction: Dir, pos: Tuple[int, int]
    ) -> None:
        super().__init__(unique_id, model)
        self.open_direction = open_direction
        self.pos = pos
        self.entry_pos = calc_pos(pos, self.open_direction)

    def step(self) -> None:
        pass

    def advance(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"<Storage id={self.unique_id}, pos={self.pos}"


class ChargingStation(Agent):
    def __init__(self, unique_id: int, model: Model) -> None:
        super().__init__(unique_id, model)

    def step(self) -> None:
        pass

    def advance(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"<ChargingStation id={self.unique_id} pos={self.pos}>"


class Task:
    """
    A class representing a task to be completed by a robot in the warehouse.

    Attributes:
    -----------
    product : Product
        The product associated with the task.
    start : Tuple[int, int], optional
        The starting location of the robot. If None, the robot will pick up the pallet from the storage.
    destination : Tuple[int, int], optional
        The destination location of the robot. If None, the robot will drop off the pallet in the storage.
    robot : Robot, optional
        The robot assigned to complete the task.
    """

    def __init__(
        self,
        product: Product,
        start: Optional[Tuple[int, int]] = None,
        destination: Optional[Tuple[int, int]] = None,
        id: Optional[str] = "",
    ) -> None:
        self.product = product
        self.start = start
        self.destination: Optional[Tuple[int, int]] = destination
        self.robot: Optional[Robot] = None
        self.id = id

    def __repr__(self) -> str:
        return f"<Task product={self.product}, start={self.start}, destination={self.destination}> robot={self.robot}, id={self.id}"


class Robot(Agent):
    """
    A robot agent that can move around the warehouse, pick up and deliver pallets, and charge its battery.

    Attributes:
        unique_id : int
            A unique identifier for the robot.
        model : Model
            The model that the robot belongs to.
        available_storage : List [Tuple[int, int]]
            A list of coordinates representing available storage spaces.
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
        charging_stations: List[ChargingStation] = [],
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
        self.available_storage: List[Storage] = storage

        self.battery = 100
        self.charging_rate = charging_rate
        self.charging_stations: List[ChargingStation] = charging_stations

        self.messages = []
        self.costs = {}

        self.path_cache = {}

        print(f"{id(self.available_storage)}")

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
            f"{self.pos}, {self.state}, {self.available_storage}, {self.available_pallets}"
        )
        self.read_messages()
        self.find_charging_stations()

        charging_station_path = self.find_closest_charging_station()

        if self.state == RS.MOVING_TO_STATION:
            if not self.path:
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
                    self.broadcast_message(
                        Msg.UNAVAILABLE_STORAGE, self.get_storage(self.next_path[0])
                    )
            else:
                return

        elif self.state == RS.MOVING_TO_PALLET:
            if not self.path:
                self.load_pallet()

                self.state = RS.MOVING_TO_DESTINATION
                if self.current_task.destination is None:
                    self.path = self.next_path
                    self.next_path = []
                else:
                    self.path = self.find_path(self.current_task.destination)
                    self.broadcast_message(
                        Msg.AVAILABLE_STORAGE, self.get_storage(self.pos)
                    )

        elif self.state == RS.MOVING_TO_DESTINATION:
            if not self.path:
                self.state = RS.IDLE
                self.path = []

                if self.current_task.destination is None:
                    self.print_robot_data(f"unloading pallet")
                    self.broadcast_message(Msg.NEW_PALLET, self.pallets[-1])

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
                    f"adding pallet {message[1]} to available pallets list ({self.available_storage})"
                )
                self.available_pallets.append(message[1])
            elif message[0] == Msg.TOOK_PALLET:
                self.print_robot_data(
                    f"removing pallet {message[1]} from available pallets list ({self.available_storage})"
                )
                self.available_pallets.remove(message[1])

            elif message[0] == Msg.AVAILABLE_STORAGE:
                self.print_robot_data(
                    f"Adding {message[1]} to available storage {self.available_storage}"
                )
                self.available_storage.append(message[1])
            elif message[0] == Msg.UNAVAILABLE_STORAGE:
                self.print_robot_data(
                    f"Removing {message[1]} from available storage {self.available_storage}"
                )
                self.available_storage.remove(message[1])

    def get_storage(self, pos: Tuple[int, int]) -> Storage:
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
            if self.available_storage:
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
        storage_destinations = []
        for storage in self.available_storage:
            storage_destinations.append(storage.entry_pos)

        paths = self.find_paths(storage_destinations)

        storage_pos = min(paths, key=lambda pos: len(paths[pos]))

        path = paths[storage_pos]
        path.insert(
            0,
            [
                storage.pos
                for storage in self.available_storage
                if storage_pos
                in self.model.grid.get_neighborhood(storage.pos, moore=False)
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
    def is_obstacle(agent) -> bool:
        if isinstance(agent, Pallet):
            return True
        if isinstance(agent, ChargingStation):
            return True
        if isinstance(agent, Robot):
            return True

        return False

    def is_storage(self, pos: Tuple[int, int]) -> bool:
        for storage in self.available_storage:
            print(storage)
            if storage.pos == pos:
                return True

        for pallet in self.available_pallets:
            if pallet.pos == pos:
                return True

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
            for agent, pos in self.model.grid.coord_iter()
            if not self.is_obstacle(agent)
            and not self.is_storage(pos)
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


class ConveyorBelt(Agent):
    """
    A class representing a conveyor belt in a warehouse automation system.

    Attributes:
    -----------
    unique_id : int
        A unique identifier for the conveyor belt.
    model : Model
        The model that the conveyor belt belongs to.
    direction : Dir
        The direction that the conveyor belt moves in.
    next_pos : tuple
        The next position that the conveyor belt moves the objects to.
    """

    def __init__(self, unique_id: int, model: Warehouse, direction: Dir) -> None:
        super().__init__(unique_id, model)
        self.model = model

        self.direction = direction
        self.next_pos = None

        if self.direction == Dir.UP:
            self.next_pos = self.pos + (0, 1)
        elif self.direction == Dir.DOWN:
            self.next_pos = self.pos - (0, 1)
        elif self.direction == Dir.LEFT:
            self.next_pos = self.pos - (1, 0)
        elif self.direction == Dir.RIGHT:
            self.next_pos = self.pos + (1, 0)

    def step(self) -> None:
        pass

    def advance(self) -> None:
        items = self.model.grid.get_cell_list_contents([self.pos])
        if len(items) > 0:
            item = items[0]
            if isinstance(item, Item) or isinstance(item, Pallet):
                self.model.grid.move_agent(item, self.next_pos)

    def __repr__(self) -> str:
        direction_name = ["Up", "Down", "Left", "Right"][self.direction]
        return f"<ConveyorBelt id={self.unique_id}, pos={self.pos} direction={direction_name}>"


class Spawner(Agent):
    """
    A class representing a spawner agent that generates new items or pallets.

    Attributes:
    -----------
    unique_id : int
        A unique identifier for the agent.
    model : Model
        A reference to the model instance the agent belongs to.
    products : List[Product]
        A list of products that the spawner can generate.
    spawns_items : bool
        A flag indicating whether the spawner generates items or pallets.
    queue : List[Product]
        A queue of products waiting to be spawned.
    spawn_rate : int
        The rate at which the spawner generates new products.
    last_spawn : int
        The number of steps since the last product was spawned.
    """

    def __init__(
        self,
        unique_id: int,
        model: Warehouse,
        products: List[Product] = [],
        spawns_items: bool = False,
    ) -> None:
        super().__init__(unique_id, model)
        self.model = model

        if products is not None:
            self.products = products
        else:
            self.products = [Product.WATER, Product.FOOD, Product.MEDICINE]

        self.spawns_items = spawns_items

        self.queue: List[Product] = []
        self.spawn_rate = 2

        self.last_spawn = 0
        self.task_id = 0

    def step(self) -> None:
        if self.last_spawn > self.spawn_rate and np.random.rand() < 0.5:
            self.last_spawn = 0
            self.queue.append(np.random.choice(self.products))

        self.last_spawn += 1

    def advance(self) -> None:
        # Check if the position is empty
        if [
            product
            for product in self.model.grid.get_cell_list_contents([self.pos])
            if isinstance(product, Item) or isinstance(product, Pallet)
        ] or not self.queue:
            return

        product = self.queue.pop(0)

        if self.spawns_items:
            new_object = Item(self.model.next_id(), self.model, product)
        else:
            new_object = Pallet(self.model.next_id(), self.model, product)

        print(f"Spawner {self.unique_id} spawned {new_object.unique_id} at {self.pos}")
        self.model.grid.place_agent(new_object, self.pos)
        self.model.schedule.add(new_object)

        # Create a new task
        self.model.create_task(product, start=self.pos, id=f"s {self.task_id}")
        self.task_id += 1


class Despawner(Agent):
    """
    A class representing an agent that removes pallets from the warehouse grid.

    Attributes:
    -----------
    unique_id : int
        A unique identifier for the agent.
    model : Model
        A reference to the model instance the agent belongs to.
    queue : List[Product]
        A list of products that the agent is waiting to despawn.
    request_rate : int
        The number of steps between each request for a new product to despawn.
    last_request : int
        The number of steps since the agent last requested a new product to despawn.
    """

    def __init__(
        self, unique_id: int, model: Warehouse, products: List[Product]
    ) -> None:
        super().__init__(unique_id, model)
        self.model = model

        self.queue: List[Product] = []

        self.request_rate = 5
        self.last_request = 0
        self.products = products
        self.active_request = None
        self.task_id = 0

    def step(self) -> None:
        if self.last_request > self.request_rate and np.random.rand() < 0.5:
            self.last_request = 0
            self.queue.append(np.random.choice(self.products))

        self.last_request += 1

    def advance(self) -> None:
        if len(self.queue) == 0:
            return

        pallets = [
            pallet
            for pallet in self.model.grid.get_cell_list_contents([self.pos])
            if isinstance(pallet, Pallet) and not pallet.in_robot
        ]

        if len(pallets) > 0 and pallets[0].product == self.active_request:
            item = pallets[0]
            self.model.grid.remove_agent(item)
            self.model.schedule.remove(item)
            self.active_request = None

        if self.active_request is None:
            self.active_request = self.queue.pop(0)
            self.model.create_task(
                self.active_request, destination=self.pos, id=f"d {self.task_id}"
            )


class Palletizer(Agent):
    """
    An agent that palletizes items.

    Attributes:
    -----------
    unique_id : int
        Unique identifier for the agent.
    model : Model
        The model that the agent belongs to.
    items_to_palletize : int
        The number of items to palletize before creating a new pallet.
    speed : int
        The operating speed.
    direction : Dir
        The direction that the agent puts the pallets in.
    input_pos : tuple
        The position where the agent receives items.
    output_pos : tuple
        The position where the agent places pallets.
    quantities : Dict[Product, int]
        A dictionary that stores the quantity of each product that the agent has received.
    """

    def __init__(
        self,
        unique_id: int,
        model: Warehouse,
        items_to_palletize: int,
        speed: int,
        direction: Dir,
    ) -> None:
        super().__init__(unique_id, model)
        self.model = model

        self.items_to_palletize = items_to_palletize
        self.speed = speed
        self.direction = direction
        self.input_pos = None
        self.output_pos = None
        self.quantities: Dict[Product, int] = {}

        if self.direction == Dir.UP:
            self.input_pos = self.pos + (0, 1)
            self.output_pos = self.pos - (0, 1)
        elif self.direction == Dir.DOWN:
            self.input_pos = self.pos - (0, 1)
            self.output_pos = self.pos + (0, 1)
        elif self.direction == Dir.LEFT:
            self.input_pos = self.pos - (1, 0)
            self.output_pos = self.pos + (1, 0)
        elif self.direction == Dir.RIGHT:
            self.input_pos = self.pos + (1, 0)
            self.output_pos = self.pos - (1, 0)

    def step(self) -> None:
        for item in self.model.grid.get_cell_list_contents([self.input_pos]):
            if isinstance(item, Item):
                if item.product not in self.quantities:
                    self.quantities[item.product] = 1
                self.quantities[item.product] += 1
                self.model.grid.remove_agent(item)
                self.model.schedule.remove(item)

        product_to_palletize = max(self.quantities, key=self.quantities.get)
        if self.quantities[product_to_palletize] < self.items_to_palletize:
            return

        self.quantities[product_to_palletize] -= self.items_to_palletize
        pallet = Pallet(self.model.next_id(), self.model, product_to_palletize)
        self.model.grid.place_agent(pallet, self.output_pos)
        self.model.schedule.add(pallet)


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

        # self.generate_warehouse()
        self.generate_static_warehouse()

    def filter_grid(self) -> None:
        for agents, pos in self.grid.coord_iter():
            if agents or pos in self.storage:
                self.open_spaces.remove(pos)

    def generate_static_warehouse(self) -> None:
        # Create storage in specific locations
        self.storage_positions = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]
        for pos in self.storage_positions:
            s = Storage(self.next_id(), self, Dir.RIGHT, pos)
            self.schedule.add(s)
            self.grid.place_agent(s, pos)
            self.storage.append(s)

        # Create charging stations in specific locations
        c = ChargingStation(self.next_id(), self)
        self.schedule.add(c)
        self.grid.place_agent(c, (10, 10))
        self.charging_stations.append(c)

        # Create spawners in specific locations
        s = Spawner(self.next_id(), self, [Product.WATER])
        self.schedule.add(s)
        self.grid.place_agent(s, (1, 10))

        # Create despawners in specific locations
        d = Despawner(self.next_id(), self, [Product.WATER])
        self.schedule.add(d)
        self.grid.place_agent(d, (2, 10))

        # Create robots in specific locations
        self.robots = []
        for _ in range(self.num_robots):
            r = Robot(
                self.next_id(),
                self,
                self.storage.copy(),
                charging_stations=self.charging_stations,
            )
            self.robots.append(r)
            self.schedule.add(r)

            idx = np.random.randint(0, len(self.open_spaces))
            self.grid.place_agent(r, self.open_spaces[idx])
            self.open_spaces.pop(idx)

    def generate_warehouse(self) -> None:
        # Create storage
        for x in range(self.width):
            for y in range(self.height):
                if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
                    self.storage.append((x, y))
                    # self.grid.place_agent(ChargingStation(
                    #     self.next_id(), self), (x, y))

        # Create robots
        for _ in range(self.num_robots):
            robot = Robot(self.next_id(), self, self.storage)
            self.robots.append(robot)
            self.schedule.add(robot)
            # self.grid.place_agent(robot, (1, 1))

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
        id: Optional[str] = "",
    ) -> None:
        task = Task(product, start, destination, id)
        self.broadcast_message(Msg.NEW_TASK, task)

    def get_pallets(self) -> List[Pallet]:
        pallets = []
        pos_set = set()

        for agents, pos in self.grid.coord_iter():
            pallets.extend([agent for agent in agents if isinstance(agent, Pallet)])

            if pos in pos_set:
                print(f"-----Error, multiple pallets on {pos}-----")

        return pallets


if __name__ == "__main__":
    STEPS = 1000

    WIDTH = 15
    HEIGHT = 15

    NUM_ROBOTS = 3
    NUM_SPAWNERS = 3
    NUM_DESPAWNERS = 3

    def batch_run(iterations: int, steps: int):
        for _ in range(iterations):
            w = Warehouse(WIDTH, HEIGHT, NUM_ROBOTS, NUM_SPAWNERS, NUM_DESPAWNERS)

            for _ in range(steps):
                w.step()

    if __name__ == "__main__":
        # batch_run(100, 100)

        m = Warehouse(WIDTH, HEIGHT, NUM_ROBOTS, NUM_SPAWNERS, NUM_DESPAWNERS)
        pallets = []
        tasks = []

        for i in range(STEPS):
            print(f"\n-----Step {i}-----\n")
            print(f"Pallets: {pallets}")
            print(f"Tasks: {tasks}\n")
            m.step()

            pallets = m.get_pallets()
            for agent in m.schedule.agents:
                if isinstance(agent, Robot):
                    tasks = agent.tasks
                    break
