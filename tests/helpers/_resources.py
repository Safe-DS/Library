import os

_resources_root = os.path.join(os.path.dirname(__file__), "..", "resources")


def resolve_resource_path(resource_path: str) -> str:
    """
    Resolves a path relative to the `resources` directory to an absolute path.

    Parameters
    ----------
    resource_path : str
        The path to the resource relative to the `resources` directory.

    Returns
    -------
    absolute_path : str
        The absolute path to the resource.
    """
    return os.path.join(_resources_root, resource_path)
