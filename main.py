import streamlit as st
from app import capture_and_extract_text
from app3 import real_time_image_captioning

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Capture and Extract Text", "Real-Time Image Captioning"])

# Page selection
if page == "Capture and Extract Text":
    capture_and_extract_text()
elif page == "Real-Time Image Captioning":
    real_time_image_captioning()