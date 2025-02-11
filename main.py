import streamlit as st
from app import capture_and_extract_text
from app3 import real_time_image_captioning

# Custom CSS for title
st.markdown(
    """
    <style>
    .title {
        font-size: 50px;
        font-weight: bold;
        text-align: center;
        color: #800000; /* Maroon color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the title
st.markdown('<div class="title">BeMyVision</div>', unsafe_allow_html=True)

# Display the logo in the sidebar
st.sidebar.image("pngegg.png", width=40)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Capture and Extract Text", "Real-Time Image Captioning"])

# Page selection
if page == "Capture and Extract Text":
    capture_and_extract_text()
elif page == "Real-Time Image Captioning":
    real_time_image_captioning()