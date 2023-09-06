from __future__ import annotations

from typing import Tuple, List
from enum import IntEnum


class Msg(IntEnum):
    """Robot Messages"""

    NEW_TASK = 0
    TOOK_TASK = 1
    NEW_PALLET = 2
    TOOK_PALLET = 3
    CHARGING_STATION = 4
    AVAILABLE_STORAGE = 5
    UNAVAILABLE_STORAGE = 6
    DO_TASK = 7


class RS(IntEnum):
    """Robot State"""

    IDLE = 0
    MOVING_TO_PALLET = 1
    MOVING_TO_DESTINATION = 2
    MOVING_TO_STATION = 3
    CHARGING = 4
    MOVING_TO_ENTRY_POINT = 5
    WAITING_FOR_CHARGING_STATION = 6
    CALCULATING_PATH = 7


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

    @staticmethod
    def all() -> List[Product]:
        return [Product.WATER, Product.FOOD, Product.MEDICINE]


def calc_pos(pos: Tuple[int, int], direction: Dir):
    x, y = pos
    if direction == Dir.UP:
        return x, y + 1
        # return pos + (0, 1)
    elif direction == Dir.DOWN:
        return x, y - 1
    elif direction == Dir.LEFT:
        return x - 1, y
    elif direction == Dir.RIGHT:
        return x + 1, y
