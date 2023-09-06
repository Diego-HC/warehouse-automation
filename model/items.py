from __future__ import annotations

from typing import Optional, Union, Tuple, TYPE_CHECKING

from mesa.model import Model
from mesa.agent import Agent

from .enums import RS, Dir, Product, calc_pos

if TYPE_CHECKING:
    from .robot import Robot


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
        self.direction = open_direction
        self.pos = pos
        self.entry_pos = calc_pos(pos, self.direction)
        self.is_available = True

    def step(self) -> None:
        pass

    def advance(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"<Storage id={self.unique_id}, pos={self.pos}"


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
        id_: Optional[str] = "",
    ) -> None:
        self.product = product
        self.start = start
        self.destination: Optional[Tuple[int, int]] = destination
        self.robot: Optional[Robot] = None
        self.id = id_

    def is_best_for_robot(self, robot: Robot) -> bool:
        robots = [robot for robot in robot.model.robots if robot.state == RS.IDLE]

        if not robots:
            return False

        if self.start is not None:
            if not robot.has_available_storage():
                return False
            def min_key(r: Robot) -> int:
                return len(r.find_path(self.start))
            
            return robot == min(robots, key=min_key)
        else:
            if not robot.has_available_pallet(self.product):
                return False

            def min_key(r: Robot) -> Union[int, float]:
                return len(r.find_closest_pallet(self.product)[0])
            
            return robot == min(robots, key=min_key)

    def __repr__(self) -> str:
        return (f"<Task product={self.product}, start={self.start}, "
                f"destination={self.destination}> robot={self.robot}, id={self.id}")
