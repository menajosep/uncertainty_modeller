# Overview

Sklearn wrapper for classifications system that computes the prediction uncertainty

This project was generated with [cookiecutter](https://github.com/audreyr/cookiecutter) using [eurecat/aiwork_python_template_01](https://ice.eurecat.org/gitlab/big-data/aiwork_python_template_01.git).

[![PyPI Version](https://img.shields.io/pypi/v/UncertaintyModeler.svg)](http://172.20.61.108/package/UncertaintyModeler)

# Setup

## Requirements

* Python 3.6+
* [Poetry](https://poetry.eustace.io/) (curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3)

## Installation

Install it directly into an activated virtual environment:

```text
$ pip3 install --extra-index-url http://172.20.61.108/ UncertaintyModeler
```

or add it to your [Poetry](https://poetry.eustace.io/) project:

```text
$ poetry config repositories.eurecat http://172.20.61.108/simple
$ poetry config http-basic.eurecat developer 8as9128asd
$ poetry add UncertaintyModeler
```

# Usage

After installation, the package can be imported:

```text
$ python
>>> import uncertainty_modeler
>>> uncertainty_modeler.__version__
```
