# Step 1: Install necessary libraries
!pip install ultralytics opencv-python cvzone

# Step 2: Download the pretrained YOLOv8n model (nano, small and fast)
!wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt

# Step 3: Upload your video file (e.g., p.mp4)
from google.colab import files
print("📤 Upload your .mp4 video file")
uploaded = files.upload()
video_path = list(uploaded.keys())[0]  # Get uploaded video filename

# Step 4: Run detection, tracking, and people counting
import cv2
from ultralytics import YOLO
import cvzone

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Pretrained small model
names = model.names

# Open uploaded video
cap = cv2.VideoCapture(video_path)

# Output video writer setup
output_path = "output_counted.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = 1020, 600
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Vertical line x-position and counters
line_x = 443
hist = {}
in_count = 0
out_count = 0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 2 != 0:
        continue
    frame = cv2.resize(frame, (width, height))

    results = model.track(frame, persist=True, classes=[0])  # Track only persons

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, ids, class_ids):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            label = names[class_id]

            # Draw bounding box and labels
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cvzone.putTextRect(frame, f'{label.upper()}', (x1, y1 - 10), scale=1, thickness=1,
                               colorT=(255, 255, 255), colorR=(0, 0, 255), offset=5, border=2)
            cvzone.putTextRect(frame, f'ID: {track_id}', (x1, y2 + 10), scale=1, thickness=2,
                               colorT=(255, 255, 255), colorR=(0, 255, 0), offset=5, border=2)

            if track_id in hist:
                prev_cx, _ = hist[track_id]
                if prev_cx < line_x <= cx:
                    in_count += 1
                elif prev_cx > line_x >= cx:
                    out_count += 1

            hist[track_id] = (cx, cy)

    # Show in/out counters and line
    cvzone.putTextRect(frame, f'IN: {in_count}', (40, 60), scale=2, thickness=2,
                       colorT=(255, 255, 255), colorR=(0, 128, 0))
    cvzone.putTextRect(frame, f'OUT: {out_count}', (40, 100), scale=2, thickness=2,
                       colorT=(255, 255, 255), colorR=(0, 0, 255))
    cv2.line(frame, (line_x, 0), (line_x, height), (0, 0, 255), 1)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Processing complete. Output video saved as:", output_path)

# Step 5: Download the output video
from google.colab import files
files.download(output_path)
