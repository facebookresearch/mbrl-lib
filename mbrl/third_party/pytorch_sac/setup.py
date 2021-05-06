import re

from setuptools import find_packages, setup

install_requires = [line.rstrip() for line in open("requirements/main.txt", "r")]

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="pytorch_sac",
    version="0.0.1",
    author="Denis Yarats",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/denisyarats/pytorch_sac",
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False,
)
