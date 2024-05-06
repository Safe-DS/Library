import functools
import operator
import struct
from typing import Any


def _structural_hash(*values: Any) -> int:
    """
    Calculate a deterministic hash value, based on the provided values.

    Parameters
    ----------
    values:
        Variable amount of values to hash

    Returns
    -------
    hash:
        Deterministic hash value
    """
    import xxhash

    return xxhash.xxh3_64(_value_to_bytes(values)).intdigest()


def _value_to_bytes(value: Any) -> bytes:
    """
    Convert any value to a deterministically hashable representation.

    Parameters
    ----------
    value:
        Object to convert to a byte representation for deterministic structural hashing

    Returns
    -------
    bytes:
        Byte representation of the provided value
    """
    if value is None:
        return b"\0"
    elif isinstance(value, bytes):
        return value
    elif isinstance(value, bool):
        return b"\1" if value else b"\0"
    elif isinstance(value, int) and value < 0:
        return value.to_bytes(8, signed=True)
    elif isinstance(value, int) and value >= 0:
        return value.to_bytes(8)
    elif isinstance(value, str):
        return value.encode("utf-8")
    elif isinstance(value, float):
        return struct.pack("d", value)
    elif isinstance(value, list | tuple):
        return functools.reduce(operator.add, [_value_to_bytes(entry) for entry in value], len(value).to_bytes(8))
    elif isinstance(value, frozenset | set):
        return functools.reduce(
            operator.add,
            sorted([_value_to_bytes(entry) for entry in value]),
            len(value).to_bytes(8),
        )
    elif isinstance(value, dict):
        return functools.reduce(
            operator.add,
            sorted([_value_to_bytes(key) + _value_to_bytes(entry) for key, entry in value.items()]),
            len(value).to_bytes(8),
        )
    else:
        return _value_to_bytes(hash(value))
