from typing import Any, Callable
from functools import wraps
from flask import Flask, Response, request, jsonify, abort
from model import Warehouse
from serializers import map_to_json, current_frame_to_json


WIDTH = 15
HEIGHT = 15
NUM_ROBOTS = 3
NUM_SPAWNERS = 3
NUM_DESPAWNERS = 3


app = Flask("warehouse-automation")


models = {}
uuid_counter = 0


def validate_id(func: Callable) -> Callable:
    @wraps(func)  # flask checks the metadata of the original function, idk why
    def wrapper(*args, **kwargs) -> Any:
        simulation_id = kwargs.pop('simulation_id')
        try:
            simulation_id = int(simulation_id)
        except ValueError:
            abort(400, "simulation_id must be an integer!")
        try:
            _ = models[simulation_id]
        except KeyError:
            abort(404, "Simulation not found!")
        return func(*args, simulation_id=simulation_id, **kwargs)
    return wrapper


@app.route("/", methods=["GET"])
def index() -> Response:
    """
    In case someone ends up in the root path, we'll redirect them to the API documentation.
    """
    return jsonify({"message": "Welcome to the Mango Technologies Warehouse Automation System!"})


@app.route("/simulation", methods=["POST", "GET"])
def simulations() -> Response:
    """
    POST: Starts a new simulation.
    GET: Returns a list of the last frames of all currently running simulations.
    """
    global uuid_counter

    if request.method == "POST":
        uuid_counter += 1
        robot_count = NUM_ROBOTS
        if "robot_count" in request.args:
            try:
                robot_count = int(request.args["robot_count"])
            except ValueError:
                abort(400, "robot_count must be an integer!")
        model = Warehouse(WIDTH, HEIGHT, robot_count, NUM_SPAWNERS, NUM_DESPAWNERS)
        models[uuid_counter] = model

        counts, details = map_to_json(model)

        resp = {
            "message": "Simulation started!",
            "id": uuid_counter,
            "parameters": {
                "width": WIDTH,
                "height": HEIGHT,
                "robot_count": robot_count
            },
            "agents": details
        }
        resp["parameters"].update(counts)
        return jsonify(resp)
    else:
        resp = []
        for simulation_id, model in models.items():
            resp.append(current_frame_to_json(model))
        return jsonify(resp)


@app.route("/simulation/<simulation_id>", methods=["GET", "DELETE"])
@validate_id
def simulation(simulation_id: int) -> Response:
    """
    GET: Returns the last frame of the simulation.
    DELETE: Deletes the simulation.
    """
    if request.method == "GET":
        model: Warehouse = models[simulation_id]

        # TODO: What happens if step has never been called in this simulation?
        return jsonify(current_frame_to_json(model))
    else:
        del models[simulation_id]
        return jsonify({"message": "Simulation deleted!"})


@app.route("/simulation/<simulation_id>/step", methods=["PUT"])
@validate_id
def step(simulation_id: int) -> Response:
    """
    Advances the simulation by the specified number of frames. If no number is specified, it defaults to 1.
    Returns a list of the new frames.
    """
    model: Warehouse = models[simulation_id]

    frames = 1
    if "frames" in request.args:
        try:
            frames = int(request.args["frames"])
        except ValueError:
            abort(400, "frames must be an integer!")

    resp = []
    for _ in range(frames):
        model.step()
        resp.append(current_frame_to_json(model))

    return jsonify(resp)


@app.route("/simulation/<simulation_id>/reset", methods=["PUT"])
@validate_id
def reset(simulation_id: int) -> Response:
    """
    Resets the simulation. In reality, the model object is just replaced with a new one.
    Only the robot count is kept.
    """
    model: Warehouse = models[simulation_id]

    robot_count = model.num_robots
    model = Warehouse(WIDTH, HEIGHT, robot_count, NUM_SPAWNERS, NUM_DESPAWNERS)
    models[simulation_id] = model

    return jsonify({"message": "Simulation reset!"})
