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

from model import *

STEPS = 1000

WIDTH = 15
HEIGHT = 15

NUM_ROBOTS = 3
NUM_SPAWNERS = 3
NUM_DESPAWNERS = 3


if __name__ == "__main__":
    m = Warehouse(WIDTH, HEIGHT, NUM_ROBOTS, NUM_SPAWNERS, NUM_DESPAWNERS)

    for i in range(STEPS):
        m.step()
        # print(m)
        # print()
