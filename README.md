Pytractions
===========

Pytractions is python framework for modular pipeline programming

Getting started
---------------

Pytractions contains documentation packaged in the container. To build the container locally simply run
```
tox -e build
```
Then you can run the container like:
```
podman run pytractions:latest web
```
If you want to explore documentation without running the container, you can install pytractions like
```
pip install -r requirements.txt
pip install .
````
and run:
`python -m pytractions.cli web`
