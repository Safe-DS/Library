---
hide:
  - navigation
---

# Safe-DS Python Library

A user-friendly library for Data Science (DS) in Python.

Our goal is to make DS more accessible to a wider audience by providing a simple, intuitive, and consistent API to solve
common tasks on small to moderately sized datasets. As such, a major focus is to provide a learning tool for DS novices.

Instead of implementing DS methods from scratch, we use established DS libraries under the hood such as:

* [polars](https://docs.pola.rs/) for manipulation of tabular data,
* [scikit-learn](https://scikit-learn.org) for classical machine learning,
* [torch](https://pytorch.org) for deep learning, and
* [seaborn](https://seaborn.pydata.org) for visualization.

For more specialized tasks, we recommend using these or other DS libraries directly.

Note that this library is still in development and not yet ready for production. Expect breaking changes in the future
without a major version bump (while in the `0.y.z` version range). Feedback is very welcome, however! If you have a
suggestion or find a bug, please [open an issue](https://github.com/Safe-DS/Library/issues/new/choose). If you have a
question, please [use our discussion forum][forum].

## Installation

Get the latest version from [PyPI](https://pypi.org/project/safe-ds):

```shell
pip install safe-ds
```

On a Windows PC with an NVIDIA graphics card, you may also want to install the CUDA versions of `torch` and
`torchvision`:

```shell
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Contributing

We welcome contributions from everyone. As a starting point, check the following resources:

* [Setting up a development environment](https://library.safeds.com/en/latest/development/environment/)
* [Project guidelines](https://library.safeds.com/en/latest/development/project_guidelines/)
* [Contributing page](https://github.com/Safe-DS/Library/contribute)

If you need further help, please [use our discussion forum][forum].

[forum]: https://github.com/orgs/Safe-DS/discussions
