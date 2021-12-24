import copy
import glob
import os
import random
from pathlib import Path

import generator


def main() -> None:
    Path('generator_graphs/').mkdir(exist_ok=True)
    generator_files = glob.glob('generator_graphs/*')
    for file in generator_files:
        os.remove(file)

    dungeon_templates = generator.load_dungeon_templates()
    template = copy.deepcopy(random.choice(dungeon_templates))
    print(f'Chose random template \'{template.name}\'')
    generator.add_extra_rooms(template.graph)
    composites = generator.construct_composite(template.graph)
    generator.draw_graph(template.graph)

    room_templates = generator.load_room_templates()

    generator.arrange_composites(room_templates, composites)


if __name__ == '__main__':
    main()
