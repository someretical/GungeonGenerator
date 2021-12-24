from dataclasses import dataclass
from enum import Enum
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
    All = 1
    No = 2
    EastWest = 3
    NorthSouth = 4


class Tile(Enum):
    Writable = 1
    Ground = 2
    Wall = 3
    Pit = 4
    Door = 5
    Chest = 6
    Shrine = 7
    Exit = 8


class Graph(Enum):
    Cycle = 1
    Tree = 2


@dataclass
class DungeonTemplate:
    name: str
    graph: nx.DiGraph


#                             root
CompositeTuple = Tuple[Graph, int, nx.DiGraph]


@dataclass
class CompositeContainer:
    composites: List[CompositeTuple]
    #                  y    x
    edges: List[Tuple[int, int]]


#                           y    x
DoorTupleList = List[Tuple[int, int, Direction]]


@dataclass
class RoomTemplate:
    id_: int
    doors: DoorTupleList
    data: np.ndarray


RoomTemplateDict = Dict[Room, List[RoomTemplate]]

#                           node             y    x
AvailableDoors = List[Tuple[int, Direction, int, int]]

#                          root
AssembledComposite = Tuple[int, AvailableDoors, np.ndarray]
