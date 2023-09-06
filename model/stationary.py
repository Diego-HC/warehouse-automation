from __future__ import annotations

from typing import Tuple, Dict, List, TYPE_CHECKING

from mesa.model import Model
from mesa.agent import Agent

import numpy as np

from .enums import Dir, Product
from .items import Item, Pallet

if TYPE_CHECKING:
    from .warehouse import Warehouse


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

    def __init__(
        self, unique_id: int, model: Warehouse, pos: Tuple[int, int], direction: Dir
    ) -> None:
        super().__init__(unique_id, model)
        self.model = model

        self.direction = direction
        self.next_pos = None

        if self.direction == Dir.UP:
            self.next_pos = (pos[0], pos[1] + 1)
        elif self.direction == Dir.DOWN:
            self.next_pos = (pos[0], pos[1] - 1)
        elif self.direction == Dir.LEFT:
            self.next_pos = (pos[0] - 1, pos[1])
        elif self.direction == Dir.RIGHT:
            self.next_pos = (pos[0] + 1, pos[1])

    def step(self) -> None:
        pass

    def advance(self) -> None:
        items = [
            item
            for item in self.model.grid.get_cell_list_contents([self.pos])
            if isinstance(item, Item) or isinstance(item, Pallet)
        ]
        for item in items:
            if isinstance(item, Item) or isinstance(item, Pallet):
                self.model.grid.move_agent(item, self.next_pos)

    def __repr__(self) -> str:
        direction_name = ["Up", "Down", "Left", "Right"][self.direction]
        return f"<ConveyorBelt id={self.unique_id}, pos={self.pos} direction={direction_name}>"


class ChargingStation(Agent):
    def __init__(self, unique_id: int, model: Model) -> None:
        super().__init__(unique_id, model)

    def step(self) -> None:
        pass

    def advance(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"<ChargingStation id={self.unique_id} pos={self.pos}>"


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


    espon
    """

    def __init__(
        self,
        unique_id: int,
        model: Warehouse,
        products: List[Product] = None,
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

        # Create a new task if the spawner is a pallet spawner
        if not self.spawns_items:
            self.model.create_task(product, start=self.pos, id_=f"s {self.task_id}")
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
                self.active_request, destination=self.pos, id_=f"d {self.task_id}"
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
        pos: Tuple[int, int],
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
            self.input_pos = (pos[0], pos[1] - 1)
            self.output_pos = (pos[0], pos[1] + 1)
        elif self.direction == Dir.DOWN:
            self.input_pos = (pos[0], pos[1] + 1)
            self.output_pos = (pos[0], pos[1] - 1)
        elif self.direction == Dir.LEFT:
            self.input_pos = (pos[0] + 1, pos[1])
            self.output_pos = (pos[0] - 1, pos[1])
        elif self.direction == Dir.RIGHT:
            self.input_pos = (pos[0] - 1, pos[1])
            self.output_pos = (pos[0] + 1, pos[1])

    def step(self) -> None:
        for item in self.model.grid.get_cell_list_contents([self.input_pos]):
            if isinstance(item, Item):
                if item.product not in self.quantities:
                    self.quantities[item.product] = 1
                self.quantities[item.product] += 1
                self.model.grid.remove_agent(item)
                self.model.schedule.remove(item)

        try:
            product_to_palletize = max(self.quantities, key=self.quantities.get)
        except ValueError:
            return

        if self.quantities[product_to_palletize] < self.items_to_palletize or [
            pallet
            for pallet in self.model.grid.get_cell_list_contents([self.output_pos])
            if isinstance(pallet, Pallet)
        ]:
            return

        self.quantities[product_to_palletize] -= self.items_to_palletize
        pallet = Pallet(self.model.next_id(), self.model, product_to_palletize)
        self.model.grid.place_agent(pallet, self.output_pos)
        self.model.schedule.add(pallet)
        self.model.create_task(
            product_to_palletize, start=self.output_pos, id_=f"p {self.unique_id}"
        )

    def advance(self) -> None:
        pass
