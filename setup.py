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
    version="0.0.4",
    description="""Pytractions is python framework for modular programming.""",
    long_description="""Pytractions is python framework for modular programming.
    It is based on the concept of tractions, which are small, self-contained modules.
    Traction fundamental idea is code which is supposed to transform input data
    to output data while using external functionality provided by libraries, program or
    external services. Strong emphasis is put on typing and data validation.
    Indivudual tractions can be combined into pipelines, which can be executed
    locally, in contianer or as tekton task.
""",
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
        "jsonschema",
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
