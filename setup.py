import os
from setuptools import find_packages, setup

with open("README.md", "r") as readme:
    long_description = readme.read()

setup(
    name="radarqc",
    version="0.2.0",
    author="John Stanco",
    description="Python package for loading and processing HF radar spectra in Cross-Spectrum file format",
    long_description=long_description,
    packages=find_packages(),
)
