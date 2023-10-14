from typing import List, Tuple


def parse_ranges(input_string: str) -> List[Tuple]:
    tokens = input_string.split()
    try:
        numbers = [int(token) for token in tokens]
    except ValueError:
        raise ValueError("Input string contains non-integer values. "
                         "You need to provide a list of intervals, e.g. '0 5 10 15'")
    if len(numbers) % 2 != 0:
        raise ValueError("Input string contains an odd number of integers. "
                         "You need to provide a list of intervals, e.g. '0 5 10 15'")
    ranges = [(numbers[i], numbers[i + 1]) for i in range(0, len(numbers), 2)]
    if len(ranges) <= 0:
        raise ValueError("At least one interval is required.")
    if ranges[0][0] == 0 and len(ranges) > 1:
        raise ValueError("If the first interval starts with 0, only one interval is allowed.")
    range_sizes = [b - a for a, b in ranges]
    if len(set(range_sizes)) > 1:
        raise ValueError("Intervals are of different sizes. "
                         "You need to provide a list of intervals, e.g. '0 5 10 15'")
    return ranges
