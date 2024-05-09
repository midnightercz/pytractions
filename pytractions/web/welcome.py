import streamlit as st

st.title("Welcome")
st.write("Welcome to pytractions documentations")
st.markdown(
    """**New here?**
No need to worry. This page is just for you. If you want to read about pytractions fundamentals
please check [Fundamentals](fundamentals) section. If you want to lean more about who to use
available tractions please see the [User Guide](user_guide) section. Do you want to know how to
write your own tractions? Check the [Developer Guide](developer_guide) section.
"""
)

st.page_link("pages/01_fundamentals.py", label="Fundamentals", icon="🏛️")
st.page_link("pages/02_user_guide.py", label="User guide", icon="📕")
st.page_link("pages/03_developer_guide.py", label="Developer guide", icon="⚒️")
