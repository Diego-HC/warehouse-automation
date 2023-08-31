"""
Warehouse Automation System
===========================

Hola

Autores
-------

- Leonardo Corona Garza
- Diego Eduardo Hernández Cadena
- Andrea Guadalupe Badillo Ibarra
- Bella Elisabet Perales Meléndez y Alcocer
- Alexa Jimena Ramírez Ortiz
"""

from model import Warehouse, Robot

STEPS = 100

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