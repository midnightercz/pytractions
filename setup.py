# -*- coding: utf-8 -*-

"""setup.py"""

import os
import re
import sys

# import pkg_resources
import sys
from setuptools import setup, find_packages


classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: Implementation :: CPython",
    "Programming Language :: Python :: Implementation :: PyPy",
]

setup(
    name="pytraction",
    version="0.0.1",
    description="pytraction",
    long_description="TBD",
    long_description_content_type="text/x-rst",
    author="Jindrich Luza",
    author_email="jluza@redhat.com",
    url="https://github.com/midnightercz/pytraction",
    classifiers=classifiers,
    packages=find_packages(exclude="tests"),
    data_files=[],
    install_requires=[
        "dill",
        "mypy",
        "typing_inspect",
        "dataclasses_json",
        "pyyaml"
    ],
    include_package_data=True,
)
