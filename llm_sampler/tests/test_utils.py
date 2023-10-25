from llm_sampler.utils import build_intermediary_continuations


def test_build_intermediary_continuations():
    result = build_intermediary_continuations(
        input_ids=[0, 1, 2],
        all_continuation_ids=[[3, 4], [5], [7, 8]]
    )
    expected_output = [
        [0, 1, 2],
        [0, 1, 2, 3],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 5],
        [0, 1, 2, 7],
        [0, 1, 2, 7, 8]
    ]
    assert result == expected_output, f"Expected {expected_output}, but got {result}"

