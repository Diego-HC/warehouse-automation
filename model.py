from mesa.model import Model
from mesa.agent import Agent
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector

import numpy as np


class Item:
    def __init__(self, name: str, weight: int = 1) -> None:
        self.name = name
        self.weight = weight


class Pallet(Agent):
    def __init__(self, unique_id: int, model: Model, product: Item, weight: int = 1) -> None:
        super().__init__(unique_id, model)

        self.weight = weight
        self.product = product

        self.in_robot = False


class ChargingStation(Agent):
    def __init__(self, unique_id: int, model: Model) -> None:
        super().__init__(unique_id, model)

class Task:
    def __init__(self, unique_id: int, model: Model, product: Item, destination: (int, int) = None, start = None) -> None:
        super().__init__(unique_id, model)

        self.product = product
        # If start is None, the robot will pick up the pallet from the storage
        self.start = start
        # If destination is None, the robot will drop off the pallet in the storage
        self.destination = None


class Robot(Agent):
    def __init__(self, unique_id: int, model: Model, speed: int = 1, weightCapacity: int = 10, charging_rate: int = 15) -> None:
        super().__init__(unique_id, model)

        self.speed = speed
        self.weightCapacity = weightCapacity

        self.state = "idle"
        self.next_pos = None

        self.tasks: [Task] = []
        self.currentTask: Task = None
        self.path = []
        self.pallets = []
        self.available_pallets = []

        self.battery = 100
        self.charging_rate = charging_rate
        self.chargingStations = []

        self.messages = []

    def step(self):
        if self.battery <= 0:
            return
        
        self.read_messages()
        self.find_charging_stations()

        if self.state == "idle":
            self.find_best_task()
            if self.currentTask is not None:
                self.state = "moving to pallet"
                if self.currentTask.start is None:
                    # Find closest pallet 
                    self.path = self.find_closest_pallet(self.currentTask.product)
                else:
                    self.find_path(self.currentTask.start)
        elif self.state == "moving to pallet":
            if self.next_pos is None:
                self.load_pallet()
                self.state = "moving to destination"
            # TODO: Consider using the list as a stack instead of a queue
            self.next_pos = self.path.pop(0)
    
    def advance(self) -> None:
        if self.battery <= 0:
            return
        
        charging_station = self.model.grid.get_cell_list_contents([self.pos])
        for station in charging_station:
            if isinstance(station, ChargingStation):
                self.recharge()
                break
        else:
            if self.battery > 0:
                self.battery -= 1

        if self.next_pos is not None:
            for pallet in self.pallets:
                self.model.grid.move_agent(pallet, self.next_pos)
            
            self.model.grid.move_agent(self, self.next_pos)
            self.next_pos = None 

    def send_message(self, subject, data):
        self.messages.append((subject, data))

    def broadcast_message(self, subject, data):
        for robot in self.model.schedule.agents:
            if robot != self and isinstance(robot, Robot):
                robot.send_message(subject, data) 

    def read_messages(self):
        while len(self.messages) > 0:
            message = self.messages.pop(0)
            if message[0] == "charging_station" and message[1] not in self.chargingStations:
                self.chargingStations.append(message[1])
            if message[0] == "new task":
                self.tasks.append(message[1])
            if message[0] == "took task":
                self.tasks.remove(message[1])
                #! This may not work
                if self.currentTask == message[1]:
                    self.currentTask = None
            if message[0] == "new pallet":
                self.available_pallets.append(message[1])
            if message[0] == "took pallet":
                self.available_pallets.remove(message[1])

    def find_best_task(self):
        if len(self.tasks) > 0:
            # TODO: Implement a better task selection algorithm
            self.currentTask = self.tasks[0]
            self.tasks.remove(self.currentTask)
            self.send_message("took task", self.currentTask)

    def find_charging_stations(self):
        chargingStations = self.model.grid.get_neighbors(self.pos, moore=False, radius=2, include_center=False)

        for station in chargingStations:
            if isinstance(station, ChargingStation) and station not in self.chargingStations:
                self.chargingStations.append(station)
                self.broadcast_message("charging_station", station.pos)

    def find_path(self, destination):
        # TODO: Implement a pathfinding algorithm
        pass

    def find_paths(self, destinations):
        paths = []
        #* If pathfinding algorithm is dijkstra, make one call to find_path
        for destination in destinations:
            paths.append(self.find_path(destination))
        return paths

    def find_closest_pallet(self, product):
        paths = self.find_paths([pallet.pos for pallet in self.available_pallets if pallet.product == product])

        return min(paths, key=len)

    def load_pallet(self):
        pallets = self.model.grid.get_cell_list_contents([self.pos])
        for pallet in pallets:
            if isinstance(pallet, Pallet) and not pallet.in_robot:
                self.pallets.append(pallet)
                pallet.in_robot = True
                break
        
    def unload_pallet(self):
        pallet = self.pallets.pop()
        pallet.in_robot = False

    def recharge(self):
        self.battery = min(100, self.battery + self.charging_rate)
