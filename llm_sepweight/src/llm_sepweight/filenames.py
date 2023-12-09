import argparse

from llm_sepweight.part_state_dict import PartSpec


def main(spec):
    part_spec: PartSpec = PartSpec.from_string(spec)
    if part_spec.begin is not None:
        print(f'begin.pth')
    for layer_range in part_spec.mid:
        for i in layer_range:
            print(f'mid.{str(i).zfill(5)}.pth')
    if part_spec.end is not None:
        print('end.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process part specification.')
    parser.add_argument('spec', type=str, help='Part specification string (e.g., "b 0-16")')
    args = parser.parse_args()
    main(args.spec)
