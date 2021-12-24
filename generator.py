import copy
import glob
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


def find_room_doors(arr: np.ndarray) -> DoorTupleList:
    door_locations = np.where(arr == Tile.Door)
    door_coordinates = zip(door_locations[0], door_locations[1])
    door_tuples = []
    for y, x in door_coordinates:
        try:
            if arr[y + 1, x] == Tile.Ground:
                door_tuples.append((y, x, Direction.South))
        except IndexError:
            pass

        try:
            if arr[y - 1, x] == Tile.Ground:
                door_tuples.append((y, x, Direction.North))
        except IndexError:
            pass

        try:
            if arr[y, x + 1] == Tile.Ground:
                door_tuples.append((y, x, Direction.West))
        except IndexError:
            pass

        try:
            if arr[y, x + -1] == Tile.Ground:
                door_tuples.append((y, x, Direction.East))
        except IndexError:
            pass

    return door_tuples


def display_tile_array(data: np.ndarray) -> str:
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

        rows.append(''.join(tmp_row))

    return '\n'.join(rows)


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

                rotational_symmetry = Symmetry(data['rotatable'])
                if rotational_symmetry == Symmetry.All:
                    for i in range(1, 4):
                        tmp_rot_data = np.rot90(arr, k=i)
                        tmp_door_data = find_room_doors(tmp_rot_data)
                        rooms[room_type].append(RoomTemplate(counter, tmp_door_data, tmp_rot_data))
                        counter += 1

                    print(f'Rotated {file_path}')
                elif rotational_symmetry == Symmetry.NorthSouth:
                    tmp_rot_data = np.flipud(arr)
                    tmp_door_data = find_room_doors(tmp_rot_data)
                    rooms[room_type].append(RoomTemplate(counter, tmp_door_data, tmp_rot_data))
                    counter += 1

                    print(f'Flipped north/south {file_path}')
                elif rotational_symmetry == Symmetry.EastWest:
                    tmp_rot_data = np.fliplr(arr)
                    tmp_door_data = find_room_doors(tmp_rot_data)
                    rooms[room_type].append(RoomTemplate(counter, tmp_door_data, tmp_rot_data))
                    counter += 1

                    print(f'Flipped east/west {file_path}')

    return rooms


def load_dungeon_templates() -> List[DungeonTemplate]:
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

            composites.append((Graph.Cycle, cycle[0], new_composite))
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

            composites.append((Graph.Tree, node, tree_graph))
            draw_graph(tree_graph, f'generator_graphs/tree{i:02}.png')
            print('Cut out tree as composite.')

        edges = [
            (source, target) for source, target in graph.edges() if
            (source in flattened_cycle_nodes and target in flattened_tree_nodes) or
            (source in flattened_tree_nodes and target in flattened_cycle_nodes)
        ]

        return CompositeContainer(composites, edges)

    except nx.NetworkXException:
        return CompositeContainer([(Graph.Tree, 0, graph)], [])


def opp_dir(d: Direction) -> Direction:
    if d == Direction.North:
        return Direction.South
    elif d == Direction.South:
        return Direction.North
    elif d == Direction.East:
        return Direction.West
    else:
        return Direction.East


def resize_composite_array(x_offset: int, y_offset: int, arr: np.ndarray, doors: AvailableDoors) -> Tuple[np.ndarray, AvailableDoors]:
    # ((pad up, pad down), (pad left, pad right))
    x_pad = (0, x_offset) if x_offset > -1 else (-x_offset, 0)
    y_pad = (0, y_offset) if y_offset > -1 else (-y_offset, 0)
    arr = np.pad(arr, (y_pad, x_pad), constant_values=Tile.Writable)

    if x_offset < 0 or y_offset < 0:
        for i, (root, dir_, y, x) in enumerate(doors):
            doors[i] = (root, dir_, y - y_offset, x - x_offset)

    return arr, doors


def arrange_tree_composite(room_templates: RoomTemplateDict, composite_tuple: CompositeTuple) -> AssembledComposite:
    pass


def arrange_cycle_composite(room_templates: RoomTemplateDict, composite_tuple: CompositeTuple) -> AssembledComposite:
    (graph_type, root, graph) = composite_tuple

    # Organise the nodes so that the loop approaches the centre from both ends.
    zipped_nodes = list(zip(list(graph.nodes()), list(graph.nodes())[::-1]))
    zipped_nodes = zipped_nodes[:len(zipped_nodes) // 2 + (len(zipped_nodes) % 2 > 0)]
    sorted_nodes = []

    for n1, n2 in zipped_nodes:
        sorted_nodes.append(n1)

        if n1 != n2:
            sorted_nodes.append(n2)

    available_doors: AvailableDoors = []

    # Pick first room
    node_id = sorted_nodes[0]
    template_type_list = room_templates[graph.nodes[node_id]['type']]
    rnd_index = random.randrange(0, len(template_type_list))
    chosen_template = template_type_list[rnd_index]

    # Uncomment when more room templates are added.
    # del template_type_list[rnd_index]

    composite_arr = chosen_template.data.copy()
    # Replace all doors with walls. The door locations are already stored.
    composite_arr = np.where(composite_arr == Tile.Door, Tile.Wall, composite_arr)
    available_doors.extend([(node_id, direction, y, x) for y, x, direction in chosen_template.doors])

    print(sorted_nodes)

    # Start adding rooms to the cycle.
    for i, node_id in enumerate(sorted_nodes):
        if i == 0:
            # First room has already been added.
            continue
        elif i == len(sorted_nodes) - 1:
            # Last room in a cycle requires special treatment so the cycle actually joins up.
            # Namely, path finding :(
            pass
        elif i + 1 > len(sorted_nodes) // 2 + (len(sorted_nodes) % 2 > 0):
            # i > len(sorted_notes) / 2 (round UP)
            # Past halfway in constructing the cycle, new rooms should be added
            # in a way that brings the cycle back together.
            pass
        else:
            # template_type_list = room_templates[graph.nodes[node_id]['type']]
            # rnd_index = random.randrange(0, len(template_type_list))
            # chosen_template = template_type_list[rnd_index]
            # # del template_type_list[rnd_index]
            # data = chosen_template.data.copy()
            # data = np.where(data == Tile.Door, Tile.Wall, data)

            # # Find previous node that was added. This is slightly more tricky
            # # since the construction commences from either end of the cycle.
            # previous_node = list(graph.in_edges(node_id))[0][0] if i % 2 == 0 else list(graph.out_edges(node_id))[0][1]

            # # Find a pair of doors
            # valid_source_doors = [door_tuple for door_tuple in available_doors if door_tuple[0] == previous_node]
            # (sroot, sdir, sy, sx) = random.choice(valid_source_doors)
            # available_doors.remove((sroot, sdir, sy, sx))

            # valid_target_doors = [door_tuple for door_tuple in chosen_template.doors if door_tuple[2] == opp_dir(sdir)]
            # (ty, tx, tdir) = valid_target_doors.pop(random.randrange(len(valid_target_doors)))
            # available_doors.extend(valid_target_doors)

            # # Calculate how much to expand the 'canvas' by.


            # updated = resize_composite_array(-5, -5, composite_arr, available_doors)
            # composite_arr = updated[0]
            # available_doors = updated[1]

    print(display_tile_array(composite_arr))


def arrange_composites(room_templates: RoomTemplateDict, composite_container: CompositeContainer) -> np.ndarray:
    assembled_composites: List[AssembledComposite] = []
    templates_copy = copy.deepcopy(room_templates)

    for comp_tuple in composite_container.composites:
        assembled_composites.append(
            (arrange_tree_composite if comp_tuple[0] == Graph.Tree else arrange_cycle_composite)
            (templates_copy, comp_tuple)
        )
