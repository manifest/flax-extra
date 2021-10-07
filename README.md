# Flax extra

[![readthedocs](https://readthedocs.org/projects/flax_extra/badge/?version=latest&style=flat)](https://flax_extra.readthedocs.io)
[![codecov](https://codecov.io/gh/manifest/flax-extra/branch/main/graph/badge.svg?token=LDR4XJG8B5)](https://codecov.io/gh/manifest/flax-extra)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black#readme)
[![Package version](https://shields.io/pypi/v/flax_extra)](https://pypi.org/project/flax_extra)

The package provides extra flexibility to [Flax](github.com/google/flax) using ideas originated at [Trax](https://github.com/google/trax).

Extras include:
- Support of combinators (serial, parallel, branch, etc.) backed by [Redex](https://github.com/manifest/redex).
- A modular training loop working nicely with [Optax](https://github.com/deepmind/optax) optimizers and metrics.
    - Pluggable logging to stdout.
    - Parallel execution of training and evaluation tasks.
    - Regular and the best metric checkpoints.
    - [Tensorboard](https://www.tensorflow.org/tensorboard) integration.
- Additional linen modules.

Check out [documentation](https://flax-extra.readthedocs.io), introductory tutorials, or examples that include:
- [Perceiver IO](https://arxiv.org/abs/2107.14795)
    - [Classification model](https://flax-extra.readthedocs.io/en/latest/notebooks/perceiver_classification/example) pretrained on images from [ImageNet](https://ieeexplore.ieee.org/document/5206848).
    - [Masked-language model](https://flax-extra.readthedocs.io/en/latest/notebooks/perceiver_language_modeling/example) pretrained using a large text corpus obtained by combining [English Wikipedia and C4](https://arxiv.org/abs/1910.10683).
    - [Autoencoder model](https://flax-extra.readthedocs.io/en/latest/notebooks/perceiver_autoencoding/example) pretrained on multimodal input (audio, video, and label) of the [Kinetics-700-2020](https://arxiv.org/abs/2010.10864) dataset.
