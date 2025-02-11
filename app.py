import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def extract_text_with_easyocr(image):
    result = reader.readtext(image)
    text = " ".join([detection[1] for detection in result])  # Combine all detected text
    return text

def take_photo():
    # Capture image from webcam
    video_capture = cv2.VideoCapture(1)  # Start webcam
    
    if not video_capture.isOpened():
        st.error("Could not open webcam.")
        return None

    ret, frame = video_capture.read()  # Capture a single frame
    if not ret:
        st.error("Failed to capture image.")
        return None

    video_capture.release()  # Release the webcam

    # Convert to RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to Image for Streamlit display
    pil_img = Image.fromarray(frame_rgb)
    
    return pil_img

def capture_and_extract_text():
    # Streamlit UI
    st.title("Capture and Extract Text from Photo")

    # Capture Button
    capture_button = st.button("Capture Photo")

    if capture_button:
        st.write("Capturing photo...")

        # Take photo from webcam
        img = take_photo()

        if img:
            # Display captured image
            st.image(img, caption="Captured Image", use_container_width=True)

            # Extract text from image using EasyOCR
            extracted_text = extract_text_with_easyocr(np.array(img))
            
            st.subheader("Extracted Text:")
            st.write(extracted_text)