import copy
import glob
import math
import random
import xml.etree.ElementTree as Et
from pathlib import Path

import pygraphviz as pgv
import yaml

from constants import *

SHRINE_ID_PREFIX = 21000
SECRET_ID_PREFIX = 20000


def parse_raw_room_string(data: str) -> np.ndarray:
    tiles: List[List[Tile]] = []

    rows = data.split('\n')
    for y, row in enumerate(rows):
        row_data: List[Tile] = []

        for x, col in enumerate(row):
            if col == '+':
                row_data.append(Tile.Wall)
            elif col == '.':
                row_data.append(Tile.Ground)
            elif col == 'D':
                row_data.append(Tile.Door)
            elif col == '^':
                row_data.append(Tile.Pit)
            elif col == 'C':
                row_data.append(Tile.Chest)
            elif col == '$':
                row_data.append(Tile.Chest)
            elif col == 'R':
                row_data.append(Tile.Shrine)
            elif col == '|':
                row_data.append(Tile.Wall)
            elif col == 'X':
                row_data.append(Tile.Exit)

        tiles.append(row_data.copy())

    return np.array(tiles, dtype=Tile)


def find_room_doors(arr: np.ndarray) -> List[Door]:
    door_locations = np.where(arr == Tile.Door)
    door_coordinates = zip(door_locations[0], door_locations[1])
    doors = []
    for y, x in door_coordinates:
        try:
            if arr[y + 1, x] == Tile.Ground:
                doors.append(Door(y, x, Direction.North))
        except IndexError:
            pass

        try:
            if arr[y - 1, x] == Tile.Ground:
                doors.append(Door(y, x, Direction.South))
        except IndexError:
            pass

        try:
            if arr[y, x + 1] == Tile.Ground:
                doors.append(Door(y, x, Direction.West))
        except IndexError:
            pass

        try:
            if arr[y, x + -1] == Tile.Ground:
                doors.append(Door(y, x, Direction.East))
        except IndexError:
            pass

    return doors


def display_tile_array(data: np.ndarray, h_delimiter='', v_delimiter='\n') -> str:
    rows = []

    for row in data:
        tmp_row = []

        for col in row:
            if col == Tile.Chest:
                tmp_row.append('C')
            elif col == Tile.Door:
                tmp_row.append('D')
            elif col == Tile.Exit:
                tmp_row.append('X')
            elif col == Tile.Ground:
                tmp_row.append('.')
            elif col == Tile.Pit:
                tmp_row.append('^')
            elif col == Tile.Wall:
                tmp_row.append('+')
            elif col == Tile.Writable:
                tmp_row.append(' ')
            elif col == Tile.Shrine:
                tmp_row.append('R')
            elif col == Tile.SecretWall:
                tmp_row.append('S')

        rows.append(h_delimiter.join(tmp_row))

    return v_delimiter.join(rows)


def load_room_templates() -> RoomTemplateDict:
    rooms = {}
    counter = 0

    for room_type in Room:
        file_paths = glob.glob('rooms/' + room_type.name + '/*')
        rooms[room_type] = []

        for file_path in file_paths:
            with open(file_path, 'r') as stream:
                data = yaml.safe_load(stream)
                arr = parse_raw_room_string(data['layout'])
                doors = find_room_doors(arr)
                rooms[room_type].append(RoomTemplate(counter, doors, arr))
                counter += 1

                print(f'Loaded room template {file_path}')

                symmetry = Symmetry(data['rotatable'])
                if symmetry == Symmetry.AllRotations or symmetry == Symmetry.AllRotationAndMirror:
                    for i in range(1, 4):
                        tmp_rot_data = np.rot90(arr, k=i)
                        tmp_door_data = find_room_doors(tmp_rot_data)
                        rooms[room_type].append(RoomTemplate(counter, tmp_door_data, tmp_rot_data))
                        counter += 1

                    print(f'Rotated 4*90 {file_path}')
                elif symmetry == Symmetry.NorthSouthMirror or \
                        symmetry == Symmetry.EastWestMirror or \
                        symmetry == Symmetry.AllMirror:
                    if symmetry == Symmetry.NorthSouthMirror or symmetry == Symmetry.AllMirror:
                        tmp_rot_data = np.flipud(arr)
                        tmp_door_data = find_room_doors(tmp_rot_data)
                        rooms[room_type].append(RoomTemplate(counter, tmp_door_data, tmp_rot_data))
                        counter += 1

                        print(f'Flipped north/south {file_path}')

                    if symmetry == Symmetry.EastWestMirror or symmetry == Symmetry.AllMirror:
                        tmp_rot_data = np.fliplr(arr)
                        tmp_door_data = find_room_doors(tmp_rot_data)
                        rooms[room_type].append(RoomTemplate(counter, tmp_door_data, tmp_rot_data))
                        counter += 1

                        print(f'Flipped east/west {file_path}')
                elif symmetry == Symmetry.DiagonalMirror:
                    tmp_rot_data = np.rot90(arr, k=1)
                    tmp_door_data = find_room_doors(tmp_rot_data)
                    rooms[room_type].append(RoomTemplate(counter, tmp_door_data, tmp_rot_data))
                    counter += 1

                    print(f'Rotated 90 degrees ONCE {file_path}')

    return rooms


def load_dungeon_templates() -> List[DungeonTemplate]:
    # Templates are made with https://graphonline.ru/en/
    # The label of the node is treated as its room type.
    # The internal ID is treated the same way here.

    file_paths = glob.glob('graphs/*')
    templates = []

    for file_path in file_paths:
        tree = Et.parse(file_path)
        root = tree.getroot()
        xml_graph = root[0]
        dg = nx.DiGraph()

        for child in xml_graph:
            if child.tag == 'node':
                dg.add_node(int(child.attrib['id']), type=Room(int(child.attrib['mainText'])))
            elif child.tag == 'edge':
                dg.add_edge(int(child.attrib['source']), int(child.attrib['target']))

        templates.append(DungeonTemplate(Path(file_path).stem, dg))
        print(f'Loaded dungeon template \'{file_path}\'')

    return templates


def draw_graph(graph: nx.DiGraph, name='generator_graphs/complete.png') -> None:
    nodes = graph.nodes.items()
    edges = graph.edges()
    graph = pgv.AGraph(directed=True)
    graph.node_attr['shape'] = 'box'

    for k, v in nodes:
        graph.add_node(k, label=f'{k:02} {v["type"].name}')

    for source, target in edges:
        graph.add_edge(source, target)

    graph.draw(path=name, format='png', prog='dot')


def add_shrines(graph: nx.DiGraph) -> None:
    base_probability: float = 0.2
    max_shrines = 1
    eligible_rooms = [
        (k, v) for k, v in graph.nodes.items()
        if v['type'] == Room.Normal or
           v['type'] == Room.MainShop or
           v['type'] == Room.Shop or
           v['type'] == Room.Hub or
           v['type'] == Room.Reward or
           v['type'] == Room.Exit
    ]
    weighted_ids = [
        [k] * 3 if (v['type'] == Room.Shop or v['type'] == Room.Reward) else [k] * 2 if (
                v['type'] == Room.Hub or v['type'] == Room.Normal) else [k]
        for k, v in eligible_rooms
    ]
    flattened_ids = [id_ for sublist in weighted_ids for id_ in sublist]

    for try_ in range(max_shrines):
        if random.uniform(0, 1) <= base_probability:
            base_probability *= 0.5
            rnd_id = random.choice(flattened_ids)
            new_id = SHRINE_ID_PREFIX + try_
            flattened_ids = [id_ for id_ in flattened_ids if id_ != rnd_id]

            graph.add_node(new_id, type=Room.Shrine)
            graph.add_edge(rnd_id, new_id)

            print(f'Added shrine at {rnd_id:05} -> {new_id:05}')


def add_secret_rooms(graph: nx.DiGraph) -> None:
    base_probability: float = 0.5
    max_secret_rooms = 2
    eligible_rooms: List[int] = [
        k for k, v in graph.nodes.items()
        if v['type'] == Room.Normal or
           v['type'] == Room.MainShop or
           v['type'] == Room.Shop or
           v['type'] == Room.Shrine or
           v['type'] == Room.Hub or
           v['type'] == Room.Reward or
           v['type'] == Room.Exit
    ]
    node_degrees = [graph.degree[id_] for id_ in eligible_rooms]
    weighted_ids = [([eligible_rooms[index]] * 4 if degree == 1 else [eligible_rooms[index]]) for index, degree in
                    enumerate(node_degrees)]
    flattened_ids = [id_ for sublist in weighted_ids for id_ in sublist]

    for try_ in range(max_secret_rooms):
        if random.uniform(0, 1) <= base_probability:
            base_probability *= 0.2
            rnd_id = random.choice(flattened_ids)
            new_id = SECRET_ID_PREFIX + try_
            flattened_ids = [id_ for id_ in flattened_ids if id_ != rnd_id]

            graph.add_node(new_id, type=Room.Secret)
            graph.add_edge(rnd_id, new_id)

            print(f'Added secret room at {rnd_id:05} -> {new_id:05}')


def add_extra_rooms(graph: nx.DiGraph) -> None:
    add_shrines(graph)
    add_secret_rooms(graph)


def construct_composite(graph: nx.DiGraph) -> CompositeContainer:
    try:
        composites = []

        # Get cycles
        # noinspection PyTypeChecker
        cycles = list(nx.simple_cycles(graph))
        flattened_cycle_nodes = [id_ for sublist in cycles for id_ in sublist]
        copy_ = graph.copy()
        nodes = graph.nodes()
        for node in nodes:
            if node not in flattened_cycle_nodes:
                copy_.remove_node(node)

        for i, cycle in enumerate(cycles, start=1):
            new_composite = nx.DiGraph()
            for node in cycle:
                new_composite.add_node(node, type=graph.nodes[node]['type'])

            edges = copy_.edges(cycle)
            for source, target in edges:
                new_composite.add_edge(source, target)

            composites.append(Composite(Graph.Cycle, cycle[0], new_composite))
            draw_graph(new_composite, f'generator_graphs/cycle{i:02}.png')
            print('Cut out cycle as composite.')

        # Get trees
        flattened_tree_nodes = []
        boundary_nodes = list(nx.node_boundary(graph, flattened_cycle_nodes))
        boundary_nodes.append(0)  # 0 technically isn't a boundary node, but it still counts in this case.

        copy_ = graph.copy()
        for node in flattened_cycle_nodes:
            copy_.remove_node(node)

        for i, node in enumerate(boundary_nodes, start=1):
            tree_graph = nx.dfs_tree(copy_, node)
            for n in tree_graph.nodes():
                tree_graph.nodes[n]['type'] = graph.nodes[n]['type']
                flattened_tree_nodes.append(n)

            composites.append(Composite(Graph.Tree, node, tree_graph))
            draw_graph(tree_graph, f'generator_graphs/tree{i:02}.png')
            print('Cut out tree as composite.')

        edges = [
            Edge(source, target) for source, target in graph.edges() if
            (source in flattened_cycle_nodes and target in flattened_tree_nodes) or
            (source in flattened_tree_nodes and target in flattened_cycle_nodes)
        ]

        return CompositeContainer(composites, edges)

    except nx.NetworkXException:
        return CompositeContainer([Composite(Graph.Tree, 0, graph)], [])


def opp_dir(d: Direction) -> Direction:
    if d == Direction.North:
        return Direction.South
    elif d == Direction.South:
        return Direction.North
    elif d == Direction.East:
        return Direction.West
    else:
        return Direction.East


def resize_composite_array(y_offset: int, x_offset: int, arr: np.ndarray, a_doors: AvailableDoors,
                           u_doors: UsedDoors) -> np.ndarray:
    # ((pad up, pad down), (pad left, pad right))
    x_pad = (0, x_offset) if x_offset > -1 else (-x_offset, 0)
    y_pad = (0, y_offset) if y_offset > -1 else (-y_offset, 0)
    arr = np.pad(arr, (y_pad, x_pad), constant_values=Tile.Writable)

    if x_offset < 0 or y_offset < 0:
        for a_door_list in a_doors.values():
            for i, a_door in enumerate(a_door_list):
                a_door_list[i].y = a_door.y - y_offset
                a_door_list[i].x = a_door.x - x_offset

        for u_door_list in u_doors.values():
            for i, u_door in enumerate(u_door_list):
                u_door_list[i].y = u_door.y - y_offset
                u_door_list[i].x = u_door.x - x_offset

    return arr


def pick_first_random_room(room_templates: RoomTemplateDict, available_doors: AvailableDoors, type_: Room,
                           node: int) -> np.ndarray:
    template_type_list = room_templates[type_]
    rnd_index = random.randrange(0, len(template_type_list))
    chosen_template = template_type_list[rnd_index]
    # Uncomment when more room templates are added.
    # del template_type_list[rnd_index]

    modified = chosen_template.data.copy()
    # noinspection PyTypeChecker
    modified = np.where(modified == Tile.Door, Tile.Wall, modified)
    # The deep copy is here because the list comprehension copies by reference which is undesirable behaviour.
    available_doors[node] = [copy.deepcopy(door) for door in chosen_template.doors]

    return modified


def pick_subsequent_random_room(room_templates: RoomTemplateDict, type_: Room, sdir: Direction) -> RoomTemplate:
    # Make sure to only include rooms that have a valid doorway connection.
    template_type_list = [
        template for template in room_templates[type_]
        if any(opp_dir(sdir) == door.direction for door in template.doors)
    ]
    
    if len(template_type_list) == 0:
        print(f'Failed to find suitable template for room type {type_}. Source door direction: {sdir}')

    rnd_index = random.randrange(0, len(template_type_list))
    chosen_template = copy.deepcopy(template_type_list[rnd_index])
    # Uncomment when more room templates are added.
    # del template_type_list[rnd_index]

    # noinspection PyTypeChecker
    chosen_template.data = np.where(chosen_template.data == Tile.Door, Tile.Wall, chosen_template.data)

    return chosen_template


def pick_source_door(edge: Edge, available_doors: AvailableDoors, used_doors: UsedDoors) -> DetailedDoor:
    # Prefer source doors that are far away from any used doors.

    if len(used_doors[edge.source]) == 0:
        rnd_index = random.randrange(0, len(available_doors[edge.source]))
        door = available_doors[edge.source][rnd_index]
        detailed_door = DetailedDoor(door.y, door.x, door.direction, edge.source, edge.target)

        used_doors[edge.source].append(detailed_door)
        del available_doors[edge.source][rnd_index]
        return detailed_door
    else:
        avg_distances = []

        for a_door in available_doors[edge.source]:
            distances = []
            for u_door in used_doors[edge.source]:
                distances.append(math.dist([a_door.y, a_door.x], [u_door.y, u_door.x]))

            avg_distances.append((a_door.y, a_door.x, a_door.direction, sum(distances) / len(distances)))

        avg_distances.sort(key=lambda elem: elem[3], reverse=True)

        door = Door(avg_distances[0][0], avg_distances[0][1], avg_distances[0][2])
        available_doors[edge.source] = [d for d in available_doors[edge.source] if (d.y != door.y and d.x != door.x)]

        detailed_door = DetailedDoor(door.y, door.x, door.direction, edge.source, edge.target)
        used_doors[edge.source].append(detailed_door)
        return detailed_door


def pick_target_door(sdir: Direction, available_doors: List[Door]) -> Door:
    valid_doors = [door for door in available_doors if door.direction == opp_dir(sdir)]
    rnd_index = random.randrange(0, len(valid_doors))

    return valid_doors[rnd_index]


def arrange_tree_composite(room_templates: RoomTemplateDict, composite: Composite) -> AssembledComposite:
    available_doors: AvailableDoors = {}
    used_doors: UsedDoors = {}
    graph_nodes = list(composite.graph.nodes())

    composite_arr = pick_first_random_room(room_templates, available_doors,
                                           composite.graph.nodes[graph_nodes[0]]['type'],
                                           graph_nodes[0])
    used_doors[graph_nodes[0]] = []
    print(f'Attached room for TREE node {graph_nodes[0]:02}')

    for i, node in enumerate(graph_nodes):
        if i == 0:
            continue

        prev_node = list(composite.graph.in_edges(node))[0][0]
        s_door = pick_source_door(Edge(prev_node, node), available_doors, used_doors)
        # VERY IMPORTANT:
        # The doors of the new room cannot be appended to available_doors and used_doors just yet.
        # This is because the x and y offsets relative to composite_arr are not known at this stage!
        new_room = pick_subsequent_random_room(room_templates, composite.graph.nodes[node]['type'], s_door.direction)
        t_door = pick_target_door(s_door.direction, new_room.doors)

        # Calculate the coordinates of the new room relative to the current composite_arr.
        # The coordinates of the new room can be specified by finding the top left point
        # since the width and height are known.
        # The bottom right point is also needed for bounds checking.
        # This way, it can be determined if the existing composite_arr needs to be expanded.
        # Assume the distance between the 2 doors is 4 tiles.

        # My mental map of this program kind of crashed at this point.
        # Had to resort to plotting debug values on a spreadsheet with square tiles to visualise everything HEEELLLP

        # Relative target door.
        # Coordinates can be negative.
        tr_door = Door(
            s_door.y if (
                    s_door.direction == Direction.East or s_door.direction == Direction.West
            ) else (
                s_door.y + 5 if s_door.direction == Direction.South else s_door.y - 5
            ),
            s_door.x if (
                    s_door.direction == Direction.North or s_door.direction == Direction.South
            ) else (
                s_door.x + 5 if s_door.direction == Direction.East else s_door.x - 5
            ),
            opp_dir(s_door.direction)
        )

        # Relative new room top left and bottom right.
        # Coordinates can be negative.
        new_room_height = new_room.data.shape[0]
        new_room_width = new_room.data.shape[1]
        r_tl = [tr_door.y - t_door.y, tr_door.x - t_door.x]
        r_br = [tr_door.y + (new_room_height - t_door.y) - 1, tr_door.x + (new_room_width - t_door.x) - 1]

        # Expand the composite array if necessary.
        original_height = composite_arr.shape[0]
        original_width = composite_arr.shape[1]
        if r_tl[1] < 0:
            composite_arr = resize_composite_array(0, r_tl[1], composite_arr, available_doors, used_doors)

        if r_tl[0] < 0:
            composite_arr = resize_composite_array(r_tl[0], 0, composite_arr, available_doors, used_doors)

        if r_br[1] > original_width - 1:
            composite_arr = resize_composite_array(0, r_br[1] - (original_width - 1), composite_arr,
                                                   available_doors,
                                                   used_doors)

        if r_br[0] > original_height:
            composite_arr = resize_composite_array(r_br[0] - (original_height - 1), 0, composite_arr,
                                                   available_doors,
                                                   used_doors)

        # Adjusted relative top left and top right of new room.
        # No negative coordinates anymore.
        a_tl = [max(0, r_tl[0]), max(0, r_tl[1])]
        a_br = [a_tl[0] + new_room_height - 1, a_tl[1] + new_room_width - 1]

        # Add the new room onto the composite officially.
        for ay, by in enumerate(range(a_tl[0], a_br[0] + 1)):
            for ax, bx in enumerate(range(a_tl[1], a_br[1] + 1)):
                if new_room.data[ay, ax] != Tile.Writable:
                    composite_arr[by, bx] = new_room.data[ay, ax]

        # Place in the connection tiles.
        # Adjusted relative coordinates of doors of the new room can be found by adding their x and y components
        # to the x and y components of a_tl.
        # Adjusted relative target door...
        at_door = [a_tl[0] + t_door.y, a_tl[1] + t_door.x]
        # Adjusted relative source door...
        # The pick_source_door function will have moved the source door from available_doors to the end of used_doors.
        # Both of those lists will have had their values adjusted if composite_arr was resized.
        as_door = [used_doors[prev_node][-1].y, used_doors[prev_node][-1].x]

        # Add the door coordinates to available_doors and used_doors
        used_doors[node] = [DetailedDoor(at_door[0], at_door[1], t_door.direction, prev_node, node)]
        available_doors[node] = [
            d for d in (
                Door(door.y + a_tl[0], door.x + a_tl[1], door.direction) for door in new_room.doors
            ) if not (d.y == at_door[0] and d.x == at_door[1])
        ]

        # This only works as long as the connection is straight!
        min_y = min(at_door[0], as_door[0])
        max_y = max(at_door[0], as_door[0])
        min_x = min(at_door[1], as_door[1])
        max_x = max(at_door[1], as_door[1])
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if (y == min_y and x == min_x) or (y == max_y and x == max_x):
                    if composite.graph.nodes[node]['type'] == Room.Secret:
                        if any((door.x == x and door.y == y) for door in used_doors[prev_node]):
                            composite_arr[y, x] = Tile.SecretWall
                        else:
                            composite_arr[y, x] = Tile.Ground
                    else:
                        composite_arr[y, x] = Tile.Door
                else:
                    composite_arr[y, x] = Tile.Ground
                    if composite_arr[y - 1, x - 1] == Tile.Writable:
                        composite_arr[y - 1, x - 1] = Tile.Wall

                    if composite_arr[y - 1, x] == Tile.Writable:
                        composite_arr[y - 1, x] = Tile.Wall

                    if composite_arr[y - 1, x + 1] == Tile.Writable:
                        composite_arr[y - 1, x + 1] = Tile.Wall

                    if composite_arr[y, x - 1] == Tile.Writable:
                        composite_arr[y, x - 1] = Tile.Wall

                    if composite_arr[y, x + 1] == Tile.Writable:
                        composite_arr[y, x + 1] = Tile.Wall

                    if composite_arr[y + 1, x - 1] == Tile.Writable:
                        composite_arr[y + 1, x - 1] = Tile.Wall

                    if composite_arr[y + 1, x] == Tile.Writable:
                        composite_arr[y + 1, x] = Tile.Wall

                    if composite_arr[y + 1, x + 1] == Tile.Writable:
                        composite_arr[y + 1, x + 1] = Tile.Wall

        print(f'Attached room for TREE node {node:02}')

#         print(f'''
# Composite array:
# {display_tile_array(composite_arr, "  ")}
# New room {new_room_height: 5} high, {new_room_width: 5} wide, {composite.graph.nodes[node]["type"]}
# Source door (relative before resizing) {s_door.y: 5}y {s_door.x: 5}x
# Target door (relative before resizing) {tr_door.y: 5}y {tr_door.x: 5}x
# Relative new room before resizing:     top left {r_tl[0]: 5}y {r_tl[1]: 5}x, bottom right {r_br[0]: 5}y {r_br[1]: 5}x
# Relative new room after resizing:      top left {a_tl[0]: 5}y {a_tl[1]: 5}x, bottom right {a_br[0]: 5}y {a_br[1]: 5}x
# Size of composite_arr:                 {composite_arr.shape[0]: 5} high, {composite_arr.shape[1]: 5} wide
#         ''')

    return AssembledComposite(graph_nodes[0], available_doors, composite_arr)


def arrange_cycle_composite(room_templates: RoomTemplateDict, composite: Composite) -> AssembledComposite:
    # Organise the nodes so that the loop approaches the centre from both ends.
    zipped_nodes = list(zip(list(composite.graph.nodes()), list(composite.graph.nodes())[::-1]))
    zipped_nodes = zipped_nodes[:len(zipped_nodes) // 2 + (len(zipped_nodes) % 2 > 0)]
    sorted_nodes = []

    for n1, n2 in zipped_nodes:
        sorted_nodes.append(n1)

        if n1 != n2:
            sorted_nodes.append(n2)

    # available_doors: AvailableDoors = {}
    #
    # # Pick first room
    # node_id = sorted_nodes[0]
    # template_type_list = room_templates[graph.nodes[node_id]['type']]
    # rnd_index = random.randrange(0, len(template_type_list))
    # chosen_template = template_type_list[rnd_index]
    #
    # # Uncomment when more room templates are added.
    # # del template_type_list[rnd_index]
    #
    # composite_arr = chosen_template.data.copy()
    # # Replace all doors with walls. The door locations are already stored.
    # composite_arr = np.where(composite_arr == Tile.Door, Tile.Wall, composite_arr)
    # available_doors.extend([(node_id, direction, y, x) for y, x, direction in chosen_template.doors])
    #
    # print(sorted_nodes)
    #
    # # Start adding rooms to the cycle.
    # for i, node_id in enumerate(sorted_nodes):
    #     if i == 0:
    #         # First room has already been added.
    #         continue
    #     elif i == len(sorted_nodes) - 1:
    #         # Last room in a cycle requires special treatment so the cycle actually joins up.
    #         # Namely, path finding :(
    #         pass
    #     elif i + 1 > len(sorted_nodes) // 2 + (len(sorted_nodes) % 2 > 0):
    #         # i > len(sorted_notes) / 2 (round UP)
    #         # Past halfway in constructing the cycle, new rooms should be added
    #         # in a way that brings the cycle back together.
    #         pass
    #     else:
    #         pass
    #         # template_type_list = room_templates[graph.nodes[node_id]['type']]
    #         # rnd_index = random.randrange(0, len(template_type_list))
    #         # chosen_template = template_type_list[rnd_index]
    #         # # del template_type_list[rnd_index]
    #         # data = chosen_template.data.copy()
    #         # data = np.where(data == Tile.Door, Tile.Wall, data)
    #
    #         # # Find previous node that was added. This is slightly more tricky
    #         # # since the construction commences from either end of the cycle.
    #         # previous_node =
    #         list(graph.in_edges(node_id))[0][0] if i % 2 == 0 else list(graph.out_edges(node_id))[0][1]

    # print(display_tile_array(composite_arr))


def arrange_composites(room_templates: RoomTemplateDict, composite_container: CompositeContainer) -> np.ndarray:
    assembled_composites: List[AssembledComposite] = []
    templates_copy = copy.deepcopy(room_templates)

    for composite in composite_container.composites:
        assembled_composites.append(
            (arrange_tree_composite if composite.graph_type == Graph.Tree else arrange_cycle_composite)
            (templates_copy, composite)
        )

        if assembled_composites[-1]:
            with open('output/output.txt', 'a') as output:
                output.write('\n' + display_tile_array(assembled_composites[-1].composite_array, '  ') + '\n')

