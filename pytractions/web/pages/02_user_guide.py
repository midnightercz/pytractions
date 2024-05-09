import streamlit as st
from streamlit_option_menu import option_menu

st.title("User guide")

# 2. horizontal menu
selected2 = option_menu(
    None,
    [
        "List of available tractions (CLI)",
        "List of available tractions (Web)",
        "Running a traction in locally",
        "Running a traction in container",
        "Running a traction as a tekton task/pipeline",
    ],
    icons=["card-list", "tv-fill", "terminal", "cloud", "cloud-arrow-up"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

if selected2 == "List of available tractions (CLI)":
    st.header("List of available tractions (CLI)")
    st.markdown(
        """
If you're running this as a container applications to list available actions you run

```
podman run <your-container> -ti catalog --output-format json|yaml
```
For local installations please run
```
python -m pytractions.cli catalog --output-format json|yaml
```

which will list all installed tractions in the container. Output will look like this:

```
[
  {
    "name": "tractions-hello-world-package",
    "tags": [<list-of-traction-tags>],
    "tractions": [
      {
        "name": "TractionName",
        "type": "TRACTION",
        "module": "traction.module",
        "docs": "Traction documentation string",
        "inputs": [
            # List of traction inputs
        ],
        "outputs": [
            # List of traction outputs
        ],
        "resources": [
            # List of traction resources
        ],
        "args": [
            # List of traction args
        ]
      },
    ],
    "args": [
        # list of all available argument types
    ],
    "inputs": [
        # list of all available input types
    ],
    "outputs": [
        # list of all available output types
    ],
    "resources": [
        # list of all available resource types
    ]
```
You can use `--tag`, `--type`, `--name` arguments to filter the list of available tractions.
"""
    )

elif selected2 == "Running a traction in locally":
    st.header("Running a traction in locally")
    st.markdown(
        """
To run a traction locally you can use the following command:
```
cat inputs-args-resources.yaml |\
python -m pytraction.runner run -m output-dir traction_module::traction_cls_name
```
where inputs is yaml file with the following structure:
```
---
name: i_traction_input
data: |-
    data:
        field1: value1
        field2: value2
---
name: a_traction_arg
data: |-
    a:
        arg: arg_value
---
name: r_traction_resource
data: |-
    r:
        some_resource_config: resource_config_value
```
If you are not sure what inputs, args and resources are needed for the traction to
start you can run
```
python -m pytraction.runner generate_inputs traction_module::traction_cls_name
```
which will produce yaml file with all needed inputs arguments and resources for the traction with
helper data which tells you about data structure you need to provide for indivual
input, arg or resource.

When traction is running information about it's process is in the output-dir.
If you're running basic traction it will have only one file named by traction uid attribute.
For specialized traction like
`Tractor`, `STMD` directory will contain files representing every single basic traction in the
process. Names of the files will be <parent_traction.uid>::<traction.uid>.json. In the case of STMD
uid consists of original uid + position of input data in the input list.
"""
    )
    st.header("Resubmiting tractor")
    st.markdown(
        """
If you previously ran a tractor, you can resubmit previous run from specific traction using the
following command:
```
python -m pytraction.runner resubmit -m <output-dir> --from-traction <traction-name>
```
"""
    )

elif selected2 == "List of available tractions (Web)":
    st.header("List of available tractions (Web)")
    st.write("You can simply go here: [Catalog](http://localhost:8501/catalog)")

elif selected2 == "Running a traction in container":
    st.header("Running a traction in container")
    st.markdown(
        """
To run a traction in container you can run the following command:
```
cat inputs-args-resources.yaml | podman run <container> run -m traction_module::traction_cls_name
```
where inputs is yaml file with the following structure:
```
---
name: i_traction_input
data: |-
    data:
        field1: value1
        field2: value2
---
name: a_traction_arg
data: |-
    a:
        arg: arg_value
---
name: r_traction_resource
data: |-
    r:
        some_resource_config: resource_config_value
```
If you are not sure what inputs, args and resources are needed for the traction to start
you can run
```
python -m pytraction.runner generat_inputs traction_module::traction_cls_name
```
which will produce yaml file with all needed inputs arguments and resources for the traction with
helper data which tells you about data structure
you need to provide for indivual input, arg or resource.

When traction is running information about it's process is in the output-dir.
If you'rer running basic traction it will have only one file named by traction uid attribute.
For specialized traction like
`Tractor`, `STMD` directory will contain files representing every single basic traction in the
process. Names of the files will be <parent_traction.uid>::<traction.uid>.json. In the case of STMD
uid consists of original uid + position of input data in the input list.
"""
    )

elif selected2 == "Running a traction as a tekton task/pipeline":
    st.header("Running a traction as a tekton task/pipeline")
    st.markdown(
        """
Every traction can be converted to tekton task. Every tractor can be converted to
tekton pipeline or tekton task.
To generate tekton task for a traction/tractor you can use the following command:
```
python -m pytraction.container_runner generate_tekton_task --type traction \
<container_image> <traction_module>:<traction_class>
```
or from container:
```
podman run <container-image> generate_tekton_task --type traction <container_image> \
<traction_module>:<traction_class>
```

To generate tekton pipeline for a tractor you can run following command:
```
python -m pytraction.container_runner generate_tekton_task --type tractor <container-image> \
<traction-module>:<traction-class>
```
or from container:
```
podman run <container-image> generate_tekton_task --type tractor <container-image> \
<traction-module>:<traction-class>
```
For tekton pipeline, invidual tractions in tractor are converted to tekton tasks in the pipeline.
To generate tekton task/pipeline run you can use following command:
```
python -m pytraction.container_runner generate_tekton_task_run --type tractor|traction \
<traction-module>:<traction-class>
```
or from the container
```
podman run <container-image> generate_tekton_task_run --type tractor|traction \
<traction-module>:<traction-class>
```

"""
    )
