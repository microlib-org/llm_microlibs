import argparse
import logging
from pathlib import Path
from llm_sepweight import migrate


def main():
    parser = argparse.ArgumentParser(description="Script for migrating to flat sepweight.")
    parser.add_argument('path', type=str, help='Path to the directory containing the state dictionaries')
    args = parser.parse_args()
    path = Path(args.path)
    migrate(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
