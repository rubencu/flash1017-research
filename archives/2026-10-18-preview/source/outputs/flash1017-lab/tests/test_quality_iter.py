from flash1017.quality import judge


def test_iter_builtin_matches_the_prompt_contract():
    source = '''def stable_unique(values):
    out = []
    for value in iter(values):
        if value not in out:
            out.append(value)
    return out
'''
    assert judge('stable_unique', source)['passed']
