import os

from setuptools import find_packages, setup

setup(
    name="dmc2gym",
    version="1.0.0",
    author="Denis Yarats",
    description=("a gym like wrapper for dm_control"),
    license="",
    keywords="gym dm_control openai deepmind",
    packages=find_packages(),
    install_requires=[
        "gym",
        "dm_control",
    ],
)
