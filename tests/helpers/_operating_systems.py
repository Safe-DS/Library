import platform

import pytest

os_windows = "Windows"
os_linux = "Linux"
os_mac = "Darwin"


def skip_os_dependent(os: str | list[str]) -> None:
    if isinstance(os, str):
        os = [os]
    if platform.system() not in os:
        pytest.skip(f"This test only runs on {os}")
