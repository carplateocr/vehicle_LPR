# Ultralytics YOLO 🚀, AGPL-3.0 license
# Located at:
# E:\miniconda3\Lib\site-packages\ultralytics\solutions\

import io
import time
import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.checks import check_requirements
import easyocr
from datetime import datetime
import yt_dlp

# Vehicle class IDs from COCO dataset (e.g., car, truck, bus, motorcycle)
vehicle_classes = [2, 3, 5, 7]  # COCO IDs: 2 = car, 3 = motorcycle, 5 = bus, 7 = truck

# Initialize the EasyOCR Reader with English language support and GPU enabled
reader = easyocr.Reader(['en'], gpu=True)
output_file_path = r'C:\Users\Administrator\Documents\Notebook Files\yolov8_streamlit_v2\detected_text.txt'

def perform_ocr_on_image(vehicle_class_name, cropped_img, output_file_path=output_file_path):
    # Define an allowlist for OCR to recognize only specific characters
    allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    # Convert the cropped image to grayscale
    gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
    
    # Apply bilateral filter for noise reduction
    bfilter = cv2.bilateralFilter(gray_img, 11, 17, 17)

    # Perform OCR using EasyOCR with the defined allowlist and enable paragraph mode
    results = reader.readtext(
        bfilter, 
        allowlist=allowlist,
        blocklist=' ',
        detail=1,
        paragraph=True,
        x_ths=1000, 
        y_ths=1000
    )

    # Initialize an empty string to store the recognized text
    text = ""
    
    # Loop through the OCR results and filter based on text length and confidence
    for res in results:
        if len(results) == 1 or (len(res[1]) > 6 and res[2] > 0.1):
            text = res[1]
    
    # Remove spaces from the recognized text
    text = text.replace(" ", "")
    
    # Convert the recognized text to string
    recognized_text = str(text)
    
    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Remove the last 3 digits to get milliseconds

    # Prepare the text to append
    text_to_append = f"{timestamp}: Vehicle [ {vehicle_class_name} ] - Plate Number [ {recognized_text} ]\n"
    
    
    # Append the detected text with timestamp to a file
    with open(output_file_path, 'a') as file:
        file.write(text_to_append)

    # Return the recognized text as a string and the appended text
    return recognized_text, text_to_append

def get_youtube_live_url(youtube_url):
    ydl_opts = {'quiet': True, 'format': 'best'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        video_url = info_dict['url']
    return video_url

def inference():
    """Runs two-stage object detection on video input using Ultralytics YOLOv8 in a Streamlit application."""
    check_requirements("streamlit>=1.29.0")
    import streamlit as st

    # Hide main menu style
    menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""

    # Main title of streamlit application
    main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px; 
                             font-family: 'Archivo', sans-serif; margin-top:-50px;margin-bottom:20px;">
                    Menumbok Two-Stage Car Plate Recognition 2.0
                    </h1></div>"""

    # Set HTML page configuration
    st.set_page_config(page_title="Ultralytics Two-Stage Detection App", layout="wide", initial_sidebar_state="auto")

    # Append the custom HTML
    st.markdown(menu_style_cfg, unsafe_allow_html=True)
    st.markdown(main_title_cfg, unsafe_allow_html=True)

    # Add elements to vertical setting menu
    st.sidebar.title("User Configuration")

    # Add video source selection dropdown
    source = st.sidebar.selectbox(
        "Video",
        ("webcam", "video", "youtube_live", "android_phone"),
    )

    # Buttons for enabling/disabling the display of org_frame and ann_frame
    display_org_frame = st.sidebar.checkbox("Display Original Frame", value=True)
    display_ann_frame = st.sidebar.checkbox("Display Annotated Frame", value=True)

    vid_file_name = ""
    if source == "video":
        vid_file = st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
        if vid_file is not None:
            g = io.BytesIO(vid_file.read())  # BytesIO Object
            vid_location = "ultralytics.mp4"
            with open(vid_location, "wb") as out:  # Open temporary file as bytes
                out.write(g.read())  # Read bytes into file
            vid_file_name = "ultralytics.mp4"
    elif source == "webcam":
        vid_file_name = 0
    elif source == "youtube_live":
        youtube_url = st.sidebar.text_input("Enter the YouTube Live URL:")
        if youtube_url:
            vid_file_name = get_youtube_live_url(youtube_url)
    elif source == "android_phone":
        ip_address = st.sidebar.text_input("Enter the IP address of the Android phone camera (e.g., http://192.168.1.2:8080/video):")
        vid_file_name = ip_address

   
    # Hardcoded vehicle detection model path
    custom_vehicle_model_path = r'C:\Users\Administrator\Documents\Notebook Files\yolov8_streamlit_v2\yolov8n.pt'
    with st.spinner("Loading model..."):
        vehicle_model = YOLO(custom_vehicle_model_path)
    st.success("Vehicle detection model loaded successfully!")
    
    # Hardcoded license plate detection model path
    custom_plate_model_path = r'C:\Users\Administrator\Documents\Notebook Files\yolov8_streamlit_v2\weights\license640.pt'
    with st.spinner("Loading model..."):
        plate_model  = YOLO(custom_plate_model_path)
    st.success("License plate detection odel loaded successfully!")

    enable_trk = st.sidebar.radio("Enable Tracking", ("Yes", "No"))
    conf = float(st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01))
    iou = float(st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.01))
    skip_frame = st.sidebar.slider("Skip Frames", 1, 10, 5)  # Slider for skip_frame

    # display elements
    col1, col2 = st.columns(2)
    org_frame = col1.empty()
    ann_frame = col2.empty()
    
    text_log = st.empty()
    text_logs = [] # Initialize text logs
    
    # FPS calculation variables
    frame_count = 0
    start_time = time.time()
    fps_display = st.sidebar.empty()  # Placeholder for FPS display

    if st.sidebar.button("Start"):
        videocapture = cv2.VideoCapture(vid_file_name)  # Capture the video

        if not videocapture.isOpened():
            st.error("Could not open webcam.")

        stop_button = st.button("Stop")  # Button to stop the inference

        while videocapture.isOpened():
            success, frame = videocapture.read()
            if not success:
                st.warning("Failed to read frame from webcam. Please make sure the webcam is connected properly.")
                break

            prev_time = time.time()
            
            # Skip frames to optimize performance
            frame_count += 1
            if frame_count % skip_frame != 0:
                continue

            # Stage 1: Vehicle Detection
            if enable_trk == "Yes":
                vehicle_results = vehicle_model.track(frame, conf=conf, iou=iou, persist=True)
            else:
                vehicle_results = vehicle_model(frame, conf=conf, iou=iou)

            annotated_frame = frame.copy()

            class_ids = vehicle_results[0].boxes.cls.cpu().numpy().astype(int)  # Convert to integer numpy array

            for class_id, vehicle_box in zip(class_ids, vehicle_results[0].boxes.xyxy):
                x1, y1, x2, y2 = map(int, vehicle_box.tolist())
                if class_id in vehicle_classes:
                    vehicle_class_name = vehicle_model.names[class_id]
                    vehicle_crop = frame[y1:y2, x1:x2]
    
                    # Stage 2: License Plate Detection
                    plate_results = plate_model(vehicle_crop, conf=conf, iou=iou)
    
                    if len(plate_results[0].boxes) > 0:
                        plate_box = plate_results[0].boxes.xyxy[0]
                        px1, py1, px2, py2 = map(int, plate_box.tolist())
    
                        # Adjust coordinates to original frame
                        px1, py1, px2, py2 = px1 + x1, py1 + y1, px2 + x1, py2 + y1
    
                        # Crop the license plate
                        plate_crop = frame[py1:py2, px1:px2]
    
                        # Perform OCR
                        text_ocr, text_to_append = perform_ocr_on_image(vehicle_class_name, plate_crop)
    
                        # Add the new detected text to the list
                        text_logs.append(text_to_append)
    
                        # Limit the number of lines in text_logs to fit within the height of ann_frame
                        max_lines = 10  # Fixed number of lines
                        if len(text_logs) > max_lines:
                            text_logs = text_logs[-max_lines:]
    
                        # Draw bounding boxes and text
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vehicle
                        cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (255, 0, 0), 2)  # License plate
    
                        # Display OCR result
                        font_scale = 1
                        (text_width, text_height), _ = cv2.getTextSize(text_ocr, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                        buffer_factor = 1.5
                        rectangle_width = int(text_width * buffer_factor)
                        rectangle_height = int(text_height * buffer_factor)
                        bottom_right_x = px2 + rectangle_width
                        bottom_right_y = py2 - rectangle_height
                        cv2.rectangle(annotated_frame, (px2, py2), (bottom_right_x, bottom_right_y), (0, 0, 255), cv2.FILLED)
                        text_x = px2 + (rectangle_width - text_width) // 2
                        text_y = py2 - (rectangle_height - text_height) // 2
                        cv2.putText(annotated_frame, text_ocr, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

            # Calculate FPS
            curr_time = time.time()
            elapsed_time = curr_time - start_time
            fps = frame_count / elapsed_time

            # Display frame conditionally based on user selection
            if display_org_frame:
                org_frame.image(frame, channels="BGR")
            if display_ann_frame:
                ann_frame.image(annotated_frame, channels="BGR")
            
            # Update the text log with a unique key
            log_text = "".join(text_logs)
            # Generate a unique key for the text area
            text_log.text_area("Detected Car Plate Logs", log_text, height=300, max_chars=10000, key=f"text_log_area_{time.time()}")

            if stop_button:
                videocapture.release()  # Release the capture
                torch.cuda.empty_cache()  # Clear CUDA memory
                st.stop()  # Stop Streamlit app

            # Display FPS in sidebar
            fps_display.metric("FPS", f"{fps:.2f}")

        # Release the capture
        videocapture.release()

    # Clear CUDA memory
    torch.cuda.empty_cache()

    # Destroy window
    cv2.destroyAllWindows()