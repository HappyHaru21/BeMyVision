import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import win32com.client
import pythoncom
import threading

# Load YOLO model
yolo_net = cv2.dnn.readNet(r"C:\study\ML-Projects\yolov3.weights", 
                           r"yolov3.cfg")
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Load class labels
with open(r"coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load BLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

def initialize_sapi():
    pythoncom.CoInitialize()
    return win32com.client.Dispatch("SAPI.SpVoice")

sapi = initialize_sapi()

def generate_caption(image):
    # Resize image to a reasonable size
    image = image.resize((224, 224))

    inputs = processor(image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=60,
        min_length=10,
        do_sample=True,
        output_scores=True,
        return_dict_in_generate=True,
        temperature=0.3,
        top_p=0.9,
        num_beams=2,
        repetition_penalty=1.5,
        length_penalty=1,
        early_stopping=True,
    )
    
    generated_sequence = generated_ids.sequences[0]
    caption = processor.decode(generated_sequence, skip_special_tokens=True)

    # Confidence calculation
    log_probs = torch.stack(generated_ids.scores, dim=1).log_softmax(dim=-1)
    token_probs = torch.gather(log_probs, 2, generated_ids.sequences[:, 1:, None]).exp()
    avg_prob = token_probs.mean().item()

    return caption, avg_prob

def generate_audio(caption):
    def speak():
        try:
            sapi.Speak(caption)
        except Exception as e:
            print(f"Error in text-to-speech: {e}")

    threading.Thread(target=speak).start()

def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, indexes, class_ids

def real_time_image_captioning():
    # Streamlit UI
    st.title("📸 Real-Time Image Captioning with BLIP")

    # Live Video Feed
    st.title("Live Video Feed")
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        st.error("Failed to open webcam.")
        st.stop()

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture image from webcam.")
            break

        frame_count += 1

        # Process every 10th frame
        if frame_count % 10 == 0:
            # Convert frame to PIL Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Generate Caption
            caption, confidence = generate_caption(image)
            st.write(f"Generated caption: {caption} (Confidence: {confidence:.2f})")
            generate_audio(caption)

            # Perform YOLO object detection
            boxes, indexes, class_ids = detect_objects(frame)

            # Draw bounding boxes and print dimensions
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = f"{classes[class_ids[i]]}: {w}x{h}"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    st.write(f"{classes[class_ids[i]]}: {w}x{h}")

            # Display Result
            if confidence > 0.3:
                st.success(f"📝 Caption: {caption} (Confidence: {confidence:.2f})")
            else:
                st.warning("Low confidence caption discarded")

        # Display the frame
        FRAME_WINDOW.image(frame)

    cap.release()