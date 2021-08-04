import streamlit as st
from PIL import Image

# Import Tools
from streamlitIntroduction import streamlitIntroduction
from streamlitShallowFoundation import streamlitShallowFoundation

# Load favicon.
favicon = Image.open("images/favicon.ico")

# Page setup
st.set_page_config(
    page_title="Groundhog",
    page_icon=favicon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Create sidebar and load logo.
SIDEBAR = st.sidebar
LOGO = Image.open("images/logo.png")
SIDEBAR.image(
    LOGO,
    use_column_width=True,
)

# Tool List will be updated as new tools are loaded.
TOOL_LIST = ["Introduction", "Shallow Foundation Bearing Capacity", "PCPT Graph"]

TOOL_SELECTED = SIDEBAR.selectbox("Tool List", TOOL_LIST)

# Can be converted to Switch-Case with new Python updates.
# TOOL_LIST is compared with text, because the order of names in TOOL_LIST may change in the future, so using TOOL_LIST to compare the selection is risky.
if TOOL_SELECTED == "Introduction":
    streamlitIntroduction()
elif TOOL_SELECTED == "Shallow Foundation Bearing Capacity":
    streamlitShallowFoundation()
elif TOOL_SELECTED == "PCPT Graph":
    st.write("Will be included.")
