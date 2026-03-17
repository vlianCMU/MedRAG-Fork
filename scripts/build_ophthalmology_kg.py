import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ophthalmology import OphthalmologyConfig, OphthalmologyKGBuilder


def main() -> None:
    config = OphthalmologyConfig()
    builder = OphthalmologyKGBuilder(config)
    graph = builder.run()
    print(f"Graph built: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")


if __name__ == "__main__":
    main()
