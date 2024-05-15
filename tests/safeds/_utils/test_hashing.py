from typing import Any

import pytest
from safeds._utils._hashing import _structural_hash, _value_to_bytes
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 13852660066117729964),
        (b"123456789", 14380211418424798930),
        (-42, 7489430509543234423),
        (42, 13109960438326920571),
        (0, 3448420582392008907),
        (True, 2767458027849294907),
        (False, 13852660066117729964),
        ("abc", 13264335307911969754),
        (-1.234, 1303859999365793597),
        ((1, "2", 3.0), 1269800189614394802),
        ([1, "2", 3.0], 1269800189614394802),
        ({1, "2", 3.0}, 17310946488773236131),
        (frozenset({1, "2", 3.0}), 17310946488773236131),
        ({"a": "b", 1: 2}, 17924302838573884393),
        (Table({"col1": [1, 2], "col2:": [3, 4]}), 1655780463045162455),
    ],
    ids=[
        "none",
        "bytes",
        "int_negative",
        "int_positive",
        "int_zero",
        "boolean_true",
        "boolean_false",
        "string",
        "float",
        "tuple",
        "list",
        "set",
        "frozenset",
        "dict",
        "object_table",
    ],
)
def test_structural_hash(value: Any, expected: int) -> None:
    assert _structural_hash(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, b"\0"),
        (b"123456789", b"123456789"),
        (-42, b"\xff\xff\xff\xff\xff\xff\xff\xd6"),
        (42, b"\0\0\0\0\0\0\0*"),
        (0, b"\0\0\0\0\0\0\0\0"),
        (True, b"\1"),
        (False, b"\0"),
        ("abc", b"abc"),
        (-1.234, b"X9\xb4\xc8v\xbe\xf3\xbf"),
        ((1, "2", 3.0), b"\0\0\0\0\0\0\0\x03\0\0\0\0\0\0\0\x012\0\0\0\0\0\0\x08@"),
        ([1, "2", 3.0], b"\0\0\0\0\0\0\0\x03\0\0\0\0\0\0\0\x012\0\0\0\0\0\0\x08@"),
        ({1, "2", 3.0}, b"\0\0\0\0\0\0\0\x03\0\0\0\0\0\0\0\x01\0\0\0\0\0\0\x08@2"),
        (frozenset({1, "2", 3.0}), b"\0\0\0\0\0\0\0\x03\0\0\0\0\0\0\0\x01\0\0\0\0\0\0\x08@2"),
        ({"a": "b", 1: 2}, b"\0\0\0\0\0\0\0\x02\0\0\0\0\0\0\0\x01\0\0\0\0\0\0\0\x02ab"),
        (Table({"col1": [1, 2], "col2:": [3, 4]}), b"\x00\x8a0\xa1\x7fn\xed\xb7"),
    ],
    ids=[
        "none",
        "bytes",
        "int_negative",
        "int_positive",
        "int_zero",
        "boolean_true",
        "boolean_false",
        "string",
        "float",
        "tuple",
        "list",
        "set",
        "frozenset",
        "dict",
        "object_table",
    ],
)
def test_value_to_bytes(value: Any, expected: bytes) -> None:
    assert _value_to_bytes(value) == expected
