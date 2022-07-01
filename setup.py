#!/usr/bin/env python
from setuptools import setup

setup(
    name='hydroml',
    packages=['hydroml'],
    python_requires='>3.6',
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
