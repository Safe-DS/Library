# Environment

This document describes how to configure and use your development environment.

## Prerequisites

You must complete these steps once before you can start setting up the project itself:

1. Install [Python 3.10](https://www.python.org/downloads/).
2. Verify that `python` can be launched by running this command in a **new** terminal:
    ```shell
    python --version
    ```
    If this fails, add the directory that contains the `python` executable to your `PATH` environment variable.

3. Install [Poetry](https://python-poetry.org/docs/master/#installing-with-the-official-installer) with the official installer. Follow the instructions in the linked document for your operating system.
4. Verify that `poetry` can be launched by running this command in a **new** terminal:
    ```shell
    poetry --version
    ```
    If this fails, add the directory that contains the `poetry` executable to your `PATH` environment variable.

## Project setup

Follow the instructions for your preferred IDE. If you want to use neither [PyCharm](https://www.jetbrains.com/pycharm/) nor [Visual Studio Code](https://code.visualstudio.com/), use the generic instructions. You only need to do these steps once.

!!! note

    All terminal commands listed in this section are assumed to be run from the root of the repository.

=== "PyCharm"

    1. Clone the repository.
    2. Open the project folder in PyCharm.

=== "Visual Studio Code"

    1. Clone the repository.
    2. Open the project folder in Visual Studio Code.
    3. Install the [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python).
    4. Create a new virtual environment and install the dependencies of this project by running this command:
        ```shell
        poetry install
        ```
    5. Find the path to the virtual environment that was created in step 4 by running this command:
        ```shell
        poetry env info --path
        ```
    6. Open the command palette and search for "Python: Select Interpreter".
    7. Select the virtual environment that matches the output of step 5. It should show up in the list of available interpreters. If it does not, you can pick it manually by choosing "Enter interpreter path..." and pasting the path that was outputted in step 5 into the input field.
    8. Open the command palette and search for "Python: Configure Tests".
    9. Select "pytest" as the test runner.
    10. Select "tests" as the directory containing tests.

=== "Generic"

    1. Clone the repository.
    2. Create a new virtual environment and install the dependencies of this project by running this command:
        ```shell
        poetry install
        ```

## Running the tests

=== "PyCharm"

    .

=== "Visual Studio Code"

    1. Run the tests by opening the command palette and searching for "Test: Run All Tests".

=== "Generic"

    1. Run this command from the root of the repository:
        ```shell
        poetry run pytest
        ```

## Serving the documentation

1. Start the server by running this command from the root of the repository:
    ```shell
    poetry run mkdocs serve
    ```
2. Check the command output for the URL of the created site and open it in a browser (usually [localhost:8000](http://localhost:8000)).

You can keep the server running while you edit the documentation. The server will automatically rebuild and reload the site when you save changes.
