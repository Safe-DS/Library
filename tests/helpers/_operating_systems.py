import platform

import pytest

os_windows = "Windows"
os_linux = "Linux"
os_mac = "Darwin"


def skip_if_os(os: str | list[str]) -> None:
    if isinstance(os, str):
        os = [os]
    if platform.system() in os:
        pytest.skip(f"This test does not work on {os}")
