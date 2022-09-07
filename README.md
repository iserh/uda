# Unsupervised Domain Adaptation
[![Tests](https://github.com/iserh/uda/actions/workflows/test-main.yml/badge.svg?branch=main)](https://github.com/iserh/uda/actions/workflows/test-main.yml)
[![Lint](https://github.com/iserh/uda/actions/workflows/lint-main.yml/badge.svg?branch=main)](https://github.com/iserh/uda/actions/workflows/lint-main.yml)
## Installation
Simply run
```sh
pip install .
```

If you want to use the `wandb_utils` integrations please install with extra (double quotes only needed in `zsh`):
```sh
pip install ".[wandb]"
```

Configure runs by writing a launch.yaml, e.g.:
```yml
dataset: MAndMs
vendors:
- Philips
- Siemens
- Canon
- GE
vaes:
  Philips: iserh/UDA-MAndMs-VAE/2h2vsaj2
teachers:
  Philips: iserh/UDA-MAndMs/14ampz76
download-model: true
data-root: /tmp/data
config-dir: config
project: UDA-MAndMs
tags:
- example_tag
group: Domain-Adaptation
wandb: true
evaluate: true
evaluate-vendors: true
store-model: true
```
