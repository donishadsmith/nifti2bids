# Nifti2Bids

[![Latest Version](https://img.shields.io/pypi/v/nifti2bids.svg)](https://pypi.python.org/pypi/nifti2bids/)
[![Python Versions](https://img.shields.io/pypi/pyversions/nifti2bids.svg)](https://pypi.python.org/pypi/nifti2bids/)
[![Source Code](https://img.shields.io/badge/Source%20Code-nifti2bids-purple)](https://github.com/donishadsmith/nifti2bids)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Test Status](https://github.com/donishadsmith/nifti2bids/actions/workflows/testing.yaml/badge.svg)](https://github.com/donishadsmith/nifti2bids/actions/workflows/testing.yaml)
[![codecov](https://codecov.io/gh/donishadsmith/nifti2bids/graph/badge.svg?token=PCJ17NA627)](https://codecov.io/gh/donishadsmith/nifti2bids)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/nifti2bids/badge/?version=stable)](http://nifti2bids.readthedocs.io/en/stable/?badge=stable)


A toolkit for post hoc BIDS-ification unstructured NIfTI datasets. Includes utilities for metadata extraction, file renaming, neurobehavioral log parsing (for E-Prime and Presentation), and JSON sidecar generation, designed primarily for datasets where the original DICOMs are unavailable.

## Installation
To install ``nifti2bids`` use one of the following methods:

### Standard Installation
```bash
pip install nifti2bids
```

### Development Version

```bash
git clone --depth 1 https://github.com/donishadsmith/nifti2bids/
cd nifti2bids
pip install -e .
```
