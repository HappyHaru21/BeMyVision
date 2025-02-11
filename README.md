# Real-Time Image Captioning for the Visually Impaired

This project aims to enable visually impaired individuals to visualize their environment through real-time image captioning and object detection. The system captures live video feed from a webcam, generates descriptive captions using a BLIP model, and performs object detection using YOLOv3. The generated captions are then converted to speech using the Windows Speech API (SAPI). Additionally, the system enables visually impaired individuals to read un-brailled text using OCR (Optical Character Recognition).

## Features

- Real-time image captioning using the BLIP model
- Object detection using YOLOv3
- Text-to-speech conversion using Windows Speech API (SAPI)
- OCR for reading un-brailled text
- Streamlit-based user interface for live video feed and results display

## Requirements

- Python 3.7 or higher
- Streamlit
- OpenCV
- NumPy
- Pillow
- PyTorch
- Transformers
- pywin32
- pytesseract

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/nexathon-mikroKitty.git
    cd nexathon-mikroKitty
    ```

2. Install the required Python packages:
    ```sh
    pip install streamlit opencv-python-headless numpy pillow torch transformers pywin32 pytesseract
    ```

3. Download the YOLOv3 weights file from the following Google Drive link and place it in the same folder as the other files:
    [YOLOv3 Weights](https://drive.google.com/file/d/1k9Cn3oy8krzG8iO-mkuxxI9Jl0lSEbIL/view?usp=sharing)

4. Ensure the following files are in the same directory:
    - [yolov3.weights](http://_vscodecontentref_/1)
    - [yolov3.cfg](http://_vscodecontentref_/2)
    - [coco.names](http://_vscodecontentref_/3)
    - [app3.py](http://_vscodecontentref_/4)

## Usage

1. Run the Streamlit app:
    ```sh
    streamlit run app3.py
    ```

2. The Streamlit interface will open in your default web browser. The live video feed from your webcam will be displayed along with the generated captions and detected objects.

## Configuration

- **Webcam Index**: The default webcam index is set to `1`. If you are using a different webcam or the default webcam, you may need to change the index in the [app3.py](http://_vscodecontentref_/5) file:
    ```python
    cap = cv2.VideoCapture(0)  # Change the index if needed
    ```

- **YOLOv3 Configuration**: Ensure that the [yolov3.cfg](http://_vscodecontentref_/6) and [coco.names](http://_vscodecontentref_/7) files are in the same directory as the [yolov3.weights](http://_vscodecontentref_/8) file.

## Acknowledgements

- [YOLOv3](https://pjreddie.com/darknet/yolo/) for object detection
- [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-large) for image captioning
- [Streamlit](https://streamlit.io/) for the user interface
- [Windows Speech API (SAPI)](https://docs.microsoft.com/en-us/previous-versions/windows/desktop/ee125663(v=vs.85)) for text-to-speech conversion
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for optical character recognition

## License

This project is licensed under the MIT License. See the [LICENSE](http://_vscodecontentref_/9) file for details.