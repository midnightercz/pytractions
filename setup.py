# -*- coding: utf-8 -*-

"""setup.py"""

from setuptools import setup, find_packages

from pytractions.pkgutils import traction_entry_points

import pytractions.transformations

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
    name="pytractions",
    version="0.0.1",
    description="pytractions",
    long_description="TBD",
    long_description_content_type="text/x-rst",
    author="Jindrich Luza",
    author_email="jluza@redhat.com",
    url="https://github.com/midnightercz/pytractions",
    classifiers=classifiers,
    packages=find_packages(exclude="tests"),
    data_files=[],
    install_requires=[
        "dill",
        "mypy",
        "dataclasses_json",
        "pyyaml",
        "lark",
        "streamlit",
        "streamlit-extras",
        "streamlit_option_menu"
    ],
    include_package_data=True,
    entry_points={
        "tractions": [
            x for x in traction_entry_points(pytractions.transformations)
        ],
    }
)
