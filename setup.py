#!/usr/bin/env python
from setuptools import setup

setup(
    name='hydroml',
    python_requires='>3.6',
    packages=['hydroml', 'hydroml.scalers'],
    install_requires=[
        'torch',
        'numpy',
        'xarray',
        'numba',
        'dill',
        'scipy',
        'pandas',
    ],
)
