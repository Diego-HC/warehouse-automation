# Warehouse-Automation

![Mango Technologies](https://cdn.discordapp.com/attachments/765560541309304862/1146670912528191588/MangoTechnologies.png)

Imaginen una warehouse
Imaginen una warehouse automation

## API Reference

- [GET /](#get-)
- [POST /simulation](#post-simulation)
- [GET /simulation](#get-simulation)
- [GET /simulation/{ID}](#get-simulationid)
- [DELETE /simulation/{ID}](#delete-simulationid)
- [PUT /simulation/{ID}/step](#put-simulationidstep)
- [PUT /simulation/{ID}/reset](#put-simulationidreset)

### GET /

Regresa un JSON con un mensaje. Había que poner algo en /, se iba a ver muy solo.

### POST /simulation

Crear e inicializar una nueva simulación. Una simulación es un modelo en Mesa. Al solo haberse inicializado, solo estarán colocados los agentes iniciales, pero no será un frame como tal. Para obtener los primeros frames, véase [step](#put-simulationidstep).

#### Params

- `robot_count` - entero opcional, la cantidad de robots con la que se iniciará la simulación, default es 3

#### Respuesta

```json
{
  "id": 1,
  "message": "Simulation started!",
  "parameters": {
    "despawners": 3,
    "height": 15,
    "robot_count": 3,
    "spawners": 3,
    "width": 15
  }
}
```

### GET /simulation

Obtener una lista de objetos que representan el último frame de cada simulación que está corriendo en el momento. 

#### Respuesta

```json
[
  {
    "robots": [
      {
        "unique_id": 1,
        "pallets": [
          {
            "pos": [1, 9],
            "product": "water",
            "unique_id": 7
          },
          ...
        ],
        "pos": [1, 9],
        "speed": 1
      },
      {
        "unique_id": 2,
        "pallets": [],
        "pos": [10, 2],
        "speed": 1
      },
      ...
    ],
    "other_pallets": [
      {
        "pos": [1, 10],
        "product": "water",
        "unique_id": 8
      },
      ...
    ]
  },
  ...
]
```

### GET /simulation/{ID}

Obtener un objeto que representa el último frame de la simulación solicitada.

#### Respuesta

```json
{
  "robots": [
    {
      "unique_id": 1,
      "pallets": [
        {
          "pos": [1, 9],
          "product": "water",
          "unique_id": 7
        },
        ...
      ],
      "pos": [1, 9],
      "speed": 1
    },
    {
      "unique_id": 2,
      "pallets": [],
      "pos": [10, 2],
      "speed": 1
    },
    ...
  ],
  "other_pallets": [
    {
      "pos": [1, 10],
      "product": "water",
      "unique_id": 8
    },
    ...
  ]
}
```

#### Errores

- `400` si el ID de la simulación no es un entero
- `404` si no se encontró una simulación con el ID solicitado

### DELETE /simulation/{ID}

Elimina una simulación de la memoria.

#### Errores

- `400` si el ID de la simulación no es un entero
- `404` si no se encontró una simulación con el ID solicitado

### PUT /simulation/{ID}/step

Calcula y regresa los siguientes N frames de la simulación especificada.

#### Params

- `frames` - entero opcional, la cantidad de frames que se van a calcular, default es 1

#### Respuesta

```json
[
  {
    "robots": [
      {
        "unique_id": 1,
        "pallets": [
          {
            "pos": [1, 9],
            "product": "water",
            "unique_id": 7
          },
          ...
        ],
        "pos": [1, 9],
        "speed": 1
      },
      {
        "unique_id": 2,
        "pallets": [],
        "pos": [10, 2],
        "speed": 1
      },
      ...
    ],
    "other_pallets": [
      {
        "pos": [1, 10],
        "product": "water",
        "unique_id": 8
      },
      ...
    ]
  },
  ...
]
```

#### Errores

- `400` si el ID de la simulación no es un entero
- `404` si no se encontró una simulación con el ID solicitado

### PUT /simulation/{ID}/reset

Reestablece una simulación a como estaba recién inicializada. Realmente solo se reemplaza con un nuevo modelo con el mismo número de robots que el anterior.

#### Errores

- `400` si el ID de la simulación no es un entero
- `404` si no se encontró una simulación con el ID solicitado

![Chistoso](https://cdn.discordapp.com/attachments/765560541309304862/1146671135577096232/que.jpg)
