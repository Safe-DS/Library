from pathlib import Path

_resources_root = Path(__file__).parent / ".." / "resources"


def resolve_resource_path(resource_path: str | Path | list[str] | list[Path]) -> str | list[str]:
    """
    Resolve a path relative to the `resources` directory to an absolute path.

    Parameters
    ----------
    resource_path : str | Path
        The path to the resource relative to the `resources` directory.

    Returns
    -------
    absolute_path : str
        The absolute path to the resource.
    """
    if isinstance(resource_path, list):
        return [str(_resources_root / path) for path in resource_path]
    return str(_resources_root / resource_path)
