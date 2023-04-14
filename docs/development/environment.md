# Environment

This document describes how to configure and use your development environment.

!!! note

    All terminal commands listed below are assumed to be run from the root of the repository.

## Initial setup

1. Install [Python 3.10](https://www.python.org/downloads/).
2. Install [poetry](https://python-poetry.org/docs/master/#installing-with-the-official-installer) with the official installer.
3. Install dependencies of this project by running this command:
    ```shell
    poetry install
    ```

## Running the tests

1. Run this command:
    ```shell
    poetry run pytest
    ```

## Serving the documentation

1. Start the server by running this command:
    ```shell
    poetry run mkdocs serve
    ```
2. Check the command output for the URL of the created site and open it in a browser (usually [localhost:8000](http://localhost:8000)).

You can keep the server running while you edit the documentation. The server will automatically rebuild and reload the site when you save changes.
