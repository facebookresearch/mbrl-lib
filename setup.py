# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
import re
from setuptools import setup, find_packages


def parse_requirements_file(path):
    return [line.rstrip() for line in open(path, "r")]


reqs_main = parse_requirements_file("requirements/main.txt")
reqs_dev = parse_requirements_file("requirements/dev.txt")

with open("README.md", "r") as f:
    long_description = f.read()


init_str = Path("mbrl/__init__.py").read_text()
match = re.search(r"^__version__ = \"(?P<version>[\w\.]+?)\"$", init_str, re.MULTILINE)
assert match is not None, "Could not find version in mbrl/__init__.py"
version = match.group("version")

setup(
    name="mbrl_lib",
    version=version,
    author="Facebook AI Research",
    author_email="lep@fb.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/mbrl-lib",
    packages=find_packages(),
    classifiers=[
        "License :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=reqs_main,
    extras_require={"dev": reqs_main + reqs_dev},
    include_package_data=True,
    python_requires=">=3.7",
    zip_safe=False,
)
