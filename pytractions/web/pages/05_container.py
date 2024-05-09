import streamlit as st


st.title("Building Pytractions container")
st.markdown(
    """
Building custom pytractions container
=====================================

Pytractions provides base container you can use as base image for building container with your own
tractions. To build pytractions container, simply run the following command:
```
tox -e build
```
This will produce container with with entrypoint providing following commands:
- `run` - run traction or tractor in the container
- `catalog` - list available traction in the container
- `web` - starts this web interface
- `generate_tekton_task` - generate tekton task/pipeline for specified traction
- `generate_tekton_task_run` - generate tekton task/pipeline run for specified traction

Building your own container
---------------------------

To build your own container, use can simply use `pytractions` container as base image.
Here is an example of Dockerfile:

```Dockerfile
FROM localhost/pytraction:latest
COPY ./requirements.txt /
RUN pip install your-tractions-package
RUN mkdir /userdata
VOLUME /userdata
PORT 8051
```

"""
)
