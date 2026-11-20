"""The raw file is read as expected: 48 mice, 4 groups of 12, known column blocks."""
import config
from data.load_data import groups_from_sheet, load_raw


def test_mice_and_groups():
    raw, _ = load_raw()
    assert len(raw) == config.N_MICE == 48
    counts = raw.group.astype(str).value_counts()
    assert set(counts.index) == set(config.GROUPS)
    assert (counts == 12).all()


def test_column_blocks():
    _, blocks = load_raw()
    assert len(blocks["behaviour"]) == len(config.BEHAVIOUR_COLUMNS) == 13
    assert len(blocks["ihc"]) == 15
    assert len(blocks["protein"]) > 4000


def test_groups_match_column_a_of_sheet():
    """config.GROUP_OF_MOUSE (ID ranges) must agree with the labels written in the sheet."""
    sheet = groups_from_sheet()
    assert len(sheet) == 48 and set(sheet) == set(config.GROUPS)
    for mouse, group in config.GROUP_OF_MOUSE.items():
        assert sheet[mouse] == group


def test_mouse_ids_unique_and_complete():
    raw, _ = load_raw()
    assert raw.index.is_unique
    assert sorted(raw.index) == list(range(1, 49))


def test_behaviour_values_in_valid_range():
    raw, blocks = load_raw()
    for c in blocks["behaviour"]:
        v = raw[c].dropna().astype(float)
        assert (v >= 0).all(), c
        if c.lower().startswith("tmaze"):
            assert (v <= 1).all(), f"{c}: T-maze scores must be <= 1"
            assert ((v * 3).round(9) % 1 == 0).all(), f"{c}: T-maze scores must be k/3"
