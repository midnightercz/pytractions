import streamlit as st
from streamlit_extras.stylable_container import stylable_container

from pytractions.catalog import catalog

tractions, all_inputs, all_outputs, all_resources, all_args, tags = catalog()


st.title("Tractions catalog")

with st.popover("Filter"):

    name_filter = st.text_input("Name filter")

    tags_filter = st.multiselect("Tags filter", tags)

    inputs_filter = st.multiselect("Input types filter", all_inputs)

    outputs_filter = st.multiselect("Output types filter", all_outputs)

    resources_filter = st.multiselect("Resource types filter", all_resources)

    args_filter = st.multiselect("Arg types filter", all_args)

    type_filter = st.multiselect("Traction types filter", ["TRACTION", "TRACTOR", "STMD"])


def inputs_container():
    """Container for inputs."""
    return stylable_container(
        key="inputs_container",
        css_styles="""
{
    border: 1px solid rgba(49, 51, 63, 0.2);
    border-radius: 0.5rem;
    background-color: #c5dec8;
    padding: calc(1em - 1px);
}
""",
    )


def outputs_container():
    """Container for outputs."""
    return stylable_container(
        key="outputs_container",
        css_styles="""
{
    border: 1px solid rgba(49, 51, 63, 0.2);
    border-radius: 0.5rem;
    background-color: #dbd6c1;
    padding: calc(1em - 1px);
}
""",
    )


def resources_container():
    """Container for resources."""
    return stylable_container(
        key="resources_container",
        css_styles="""
{
    border: 1px solid rgba(49, 51, 63, 0.2);
    border-radius: 0.5rem;
    background-color: #dbc9c5;
    padding: calc(1em - 1px);
}
""",
    )


def args_container():
    """Container for args."""
    return stylable_container(
        key="args_container",
        css_styles="""
{
    border: 1px solid rgba(49, 51, 63, 0.2);
    border-radius: 0.5rem;
    background-color: #c1d0db;
    padding: calc(1em - 1px);
}
""",
    )


def indent_container():
    """Container with indent."""
    return stylable_container(
        key="indent_container",
        css_styles="""
{
    padding: calc(1em - 1px);
}
""",
    )


filtered_tractions, _, _, _, _, _ = catalog(
    tag_filter=tags_filter,
    type_filter=type_filter,
    name_filter=name_filter,
    inputs_type_filter=inputs_filter,
    outputs_type_filter=outputs_filter,
    resources_type_filter=resources_filter,
    args_type_filter=args_filter,
    type_to_str=True,
)

for d in filtered_tractions:
    for t in d["tractions"]:
        tc = stylable_container(
            key="tractions",
            css_styles="""
{
border: 1px solid rgba(49, 51, 63, 0.2);
padding: calc(1em - 1px);
border-radius: 0.5rem;
background-color: #f9f9f9;
}
        """,
        )
        with tc:
            tc.write(t["module"] + "." + t["name"])
            tc.write("description: " + (t["docs"] or ""))
            tc.multiselect(
                "Tags",
                key=d["name"] + "." + t["name"] + "-" + "tags",
                options=t["tags"],
                default=t["tags"],
                disabled=True,
            )
            tc.write("Args:")
            c = args_container()
            with c:
                for targ in t["args"]:
                    c_indent = indent_container()
                    c_indent.write(targ["name"] + " : " + targ["type"])
                    c_indent.write("description : " + (targ.get("docs") or ""))
            tc.write("Resources:")
            c = resources_container()
            with c:
                for tres in t["resources"]:
                    c_indent = indent_container()
                    c_indent.write(tres["name"] + " : " + tres["type"])
                    c_indent.write("description : " + (tres.get("docs") or ""))
            tc.write("Inputs:")
            c = inputs_container()
            with c:
                for tin in t["inputs"]:
                    c_indent = indent_container()
                    c_indent.write(tin["name"] + " : " + tin["type"])
                    c_indent.write("description : " + (tin.get("docs") or ""))
            tc.write("Outputs:")
            c = outputs_container()
            with c:
                for tout in t["outputs"]:
                    c_indent = indent_container()
                    c_indent.write(tin["name"] + " : " + tout["type"])
                    c_indent.write("description : " + (tout.get("docs") or ""))
