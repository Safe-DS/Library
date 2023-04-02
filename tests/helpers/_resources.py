from pathlib import Path

_resources_root = Path(__file__).parent / ".." / "resources"


def resolve_resource_path(resource_path: str) -> str:
    """
    Resolve a path relative to the `resources` directory to an absolute path.

    Parameters
    ----------
    resource_path : str
        The path to the resource relative to the `resources` directory.

    Returns
    -------
    absolute_path : str
        The absolute path to the resource.
    """
    return str(_resources_root / resource_path)
