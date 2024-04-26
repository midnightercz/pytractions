import argparse
import pytraction.container_runner
import pytraction.catalog

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytraction container ep")
    subparsers = parser.add_subparsers(required=True, dest='command')
    pytraction.container_runner.make_parsers(subparsers)
    pytraction.catalog.make_parsers(subparsers)
    args = parser.parse_args()
    args.command(args)



