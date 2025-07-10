import argparse
from . import container_runner
from . import catalog
from . import runner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytraction container ep")
    subparsers = parser.add_subparsers(required=True, dest="command")
    container_runner.make_parsers(subparsers)
    catalog.make_parsers(subparsers)
    runner.make_parsers(subparsers)

    args = parser.parse_args()
    args.command(args)
