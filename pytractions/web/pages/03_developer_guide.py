import streamlit as st
import os

st.title("Developer guide")
st.markdown(open(os.path.join(os.path.dirname(__file__), "developer-guide.md")).read())
