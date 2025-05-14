import streamlit as st
from welcome import welcome
from login import login
from display import display

# Page configuration
st.set_page_config(
    page_title="Epilepsy Detection System",
    page_icon="🧠",
    layout="centered",
)
# Sidebar navigation
page = st.sidebar.selectbox("Navigate", ["Welcome", "Login", "Results"])
if page == "Welcome":
    welcome()
elif page == "Login":
    login()
elif page == "Results":
    display()
