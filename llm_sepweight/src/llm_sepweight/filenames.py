import argparse
from typing import List

from llm_sepweight.part_state_dict import PartSpec


def get_filenames(spec: str) -> List[str]:
    res = []
    part_spec: PartSpec = PartSpec.from_string(spec)
    if part_spec.begin:
        res.append(f'begin.pth')
    for layer_range in part_spec.mid:
        for i in layer_range:
            res.append(f'mid.{str(i).zfill(5)}.pth')
    if part_spec.end:
        res.append('end.pth')
    return res


def main(spec):
    for filename in get_filenames(spec):
        print(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process part specification.')
    parser.add_argument('spec', type=str, help='Part specification string (e.g., "b 0-16")')
    args = parser.parse_args()
    main(args.spec)
