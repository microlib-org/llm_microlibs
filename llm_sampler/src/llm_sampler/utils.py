from typing import List


def build_intermediary_continuations(
        input_ids: List[int],
        all_continuation_ids: List[List[int]]
) -> List[List[int]]:
    results = [input_ids.copy()]
    for continuation in all_continuation_ids:
        # start with just the input_ids
        current_combination = input_ids.copy()
        # extend the current_combination by adding elements of the continuation one-by-one
        for item in continuation:
            current_combination.append(item)
            results.append(current_combination.copy())
    return results
