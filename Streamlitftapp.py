# Top: Imports and page config
import streamlit as st
st.set_page_config(page_title="People Counter", layout="centered")

import cv2
from ultralytics import YOLO
import cvzone
import tempfile
import os

# Streamlit UI
st.title("👥 People Counter using YOLOv8")
uploaded_file = st.file_uploader("📤 Upload a video (.mp4)", type=["mp4"])

#  Download YOLO model if not present
model_path = "yolov8n.pt"
if not os.path.exists(model_path):
    with st.spinner(" Downloading YOLOv8 model..."):
        from urllib.request import urlretrieve
        urlretrieve("https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt", model_path)

if uploaded_file:
    st.success(" File uploaded. Starting processing...")

    # Save video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    output_path = "output_counted.mp4"

    model = YOLO(model_path)
    names = model.names
    cap = cv2.VideoCapture(video_path)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = 1020, 600
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Tracking and counting setup
    hist = {}
    in_count = 0
    out_count = 0
    line_x = 443
    frame_count = 0

    with st.spinner("🔍 Processing video..."):
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 2 != 0:
                continue

            frame = cv2.resize(frame, (width, height))
            results = model.track(frame, persist=True, classes=[0])  # Track only people

            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                class_ids = results[0].boxes.cls.int().cpu().tolist()

                for box, track_id, class_id in zip(boxes, ids, class_ids):
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    label = names[class_id]

                    if track_id in hist:
                        prev_cx, _ = hist[track_id]
                        if prev_cx < line_x <= cx:
                            in_count += 1
                        elif prev_cx > line_x >= cx:
                            out_count += 1
                    hist[track_id] = (cx, cy)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{label.upper()}', (x1, y1 - 10), 1, 1,
                                       colorT=(255, 255, 255), colorR=(0, 0, 255))
                    cvzone.putTextRect(frame, f'ID: {track_id}', (x1, y2 + 10), 1, 2,
                                       colorT=(255, 255, 255), colorR=(0, 255, 0))

            cvzone.putTextRect(frame, f'IN: {in_count}', (40, 60), 2, 2,
                               colorT=(255, 255, 255), colorR=(0, 128, 0))
            cvzone.putTextRect(frame, f'OUT: {out_count}', (40, 100), 2, 2,
                               colorT=(255, 255, 255), colorR=(0, 0, 255))
            cv2.line(frame, (line_x, 0), (line_x, height), (0, 0, 255), 1)

            out.write(frame)

        cap.release()
        out.release()

    st.success("🎉 Processing complete!")
    with open(output_path, 'rb') as f:
        st.download_button("📥 Download Processed Video", f, file_name="people_counted.mp4")

else:
    st.info("👆 Please upload a video file to get started.")
