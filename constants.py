from dataclasses import dataclass
from enum import *
from typing import *

import networkx as nx
import numpy as np


class Room(Enum):
    Entrance = 1
    Exit = 2
    Normal = 3
    Hub = 4
    Reward = 5
    MainShop = 6
    Boss = 7
    BossFoyer = 8
    Secret = 9
    Connector = 10
    Shop = 11
    Shrine = 12
    Composite = 13


class Direction(Enum):
    North = 1
    East = 2
    South = 3
    West = 4


class Symmetry(Enum):
    AllRotations = 1
    NoRotationOrMirror = 2
    EastWestMirror = 3
    NorthSouthMirror = 4
    DiagonalMirror = 5
    AllMirror = 6
    AllRotationAndMirror = 7


class Tile(Enum):
    Writable = 1
    Ground = 2
    Wall = 3
    Pit = 4
    Door = 5
    Chest = 6
    Shrine = 7
    Exit = 8
    SecretWall = 9


class Graph(Enum):
    Cycle = 1
    Tree = 2


@dataclass
class DungeonTemplate:
    name: str
    graph: nx.DiGraph


@dataclass
class Composite:
    graph_type: Graph
    root: int
    graph: nx.DiGraph


@dataclass
class Edge:
    source: int
    target: int


@dataclass
class CompositeContainer:
    composites: List[Composite]
    edges: List[Edge]


@dataclass
class Door:
    y: int
    x: int
    direction: Direction


@dataclass
class DetailedDoor:
    y: int
    x: int
    direction: Direction
    source: int
    target: int


@dataclass
class RoomTemplate:
    id_: int
    doors: List[Door]
    data: np.ndarray


RoomTemplateDict = Dict[Room, List[RoomTemplate]]

#                     root
AvailableDoors = Dict[int, List[Door]]

#                root
UsedDoors = Dict[int, List[DetailedDoor]]


@dataclass
class AssembledComposite:
    root: int
    available_doors: AvailableDoors
    composite_array: np.ndarray
