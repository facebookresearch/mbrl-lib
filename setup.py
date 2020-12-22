# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import re
from setuptools import setup, find_packages

def parse_requirements_file(path):
    return [line.rstrip() for line in open(path, "r")]

reqs_main = parse_requirements_file("requirements/main.txt")
reqs_dev = parse_requirements_file("requirements/dev.txt")

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="mbrl_lib",
    version="0.0.1",
    author="Luis Pineda",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fairinternal/mbrl-lib",
    packages=find_packages(),
    install_requires=reqs_main,
    extras_require={"dev": reqs_main + reqs_dev},
    include_package_data=True,
    zip_safe=False,
)
