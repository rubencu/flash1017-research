import pytest

from flash1017.quality import judge, oracle, validate_code


def test_real_worker_scores_correct_and_wrong_code():
    good = '''def stable_unique(values):
    result = []
    for value in values:
        if not any(value == prior for prior in result):
            result.append(value)
    return result
'''
    assert judge("stable_unique", good)["passed"]
    wrong = "def stable_unique(values):\n    return values\n"
    score = judge("stable_unique", wrong)
    assert not score["passed"] and score["failed_cases"] > 0


def test_mutation_fails_even_with_correct_result():
    code = '''def stable_unique(values):
    result = []
    while values:
        item = values.pop(0)
        if item not in result:
            result.append(item)
    return result
'''
    assert not judge("stable_unique", code)["passed"]


@pytest.mark.parametrize("source", [
    "import os\ndef stable_unique(values): return values",
    "def stable_unique(values): return values.__class__",
    "def stable_unique(values): return __builtins__",
])
def test_unsafe_syntax_rejected_before_execution(source):
    with pytest.raises(ValueError):
        validate_code(source, "stable_unique")


def test_oracle_edge_conditions():
    assert oracle("glob_match", ["", "**"]) is True
    assert oracle("glob_match", ["\n", "?"]) is True
    assert oracle("shortest_path", [[], "missing", "missing"]) == 0
    assert oracle("rotate_ragged", [[[1, 2], [3]], None]) == [[3, 1], [None, 2]]
    assert oracle("topological_order", [{"a": ["b", "b"]}]) == ["a", "b"]
    with pytest.raises(ValueError):
        oracle("topological_order", [{"a": ["a"]}])
