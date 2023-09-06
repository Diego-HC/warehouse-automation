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


from api import app
from model.visualization import server


TEST_VISUALIZATION = True


if __name__ == "__main__":
    if TEST_VISUALIZATION:
        server.launch()
    else:
        app.run(debug=True, port=8000)
