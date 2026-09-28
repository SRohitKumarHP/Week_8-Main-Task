import os
import csv
from datetime import datetime

import cv2
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/yolov11s_best.pt"
IMAGE_PATH = "input/images/test_image.jpg"

OUTPUT_IMAGE = "output/annotated/final_inspection.jpg"
LOG_FILE = "output/logs/final_inspection_log.csv"

CONFIDENCE_THRESHOLD = 0.25


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs("output/annotated", exist_ok=True)
os.makedirs("output/logs", exist_ok=True)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("=" * 65)
print("             YOLOv11s METAL INSPECTION")
print("=" * 65)

print("Loading YOLOv11s model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")
print()


# ============================================================
# RUN YOLO INFERENCE
# ============================================================

print(f"Input image : {IMAGE_PATH}")
print("Running inspection...")
print()

results = model(
    IMAGE_PATH,
    imgsz=640,
    conf=CONFIDENCE_THRESHOLD,
    verbose=False
)

result = results[0]


# ============================================================
# PROCESS DETECTIONS
# ============================================================

detections = []

if result.boxes is not None:

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        width = x2 - x1
        height = y2 - y1
        area = width * height

        class_name = result.names[class_id]

        # Demo severity thresholds
        if area < 500:
            severity = "LOW"
        elif area < 2000:
            severity = "MEDIUM"
        else:
            severity = "HIGH"

        detections.append({
            "class_name": class_name,
            "confidence": confidence,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": width,
            "height": height,
            "area": area,
            "severity": severity
        })


# ============================================================
# PASS / FAIL DECISION
# ============================================================

if len(detections) == 0:
    inspection_result = "PASS"
else:
    inspection_result = "FAIL"


# ============================================================
# PART ID
# ============================================================

part_id = "PART_001"

timestamp = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)


# ============================================================
# ANNOTATED IMAGE
# ============================================================

annotated_image = result.plot()

cv2.imwrite(
    OUTPUT_IMAGE,
    annotated_image
)


# ============================================================
# CSV LOGGING
# ============================================================

file_exists = os.path.exists(LOG_FILE)

with open(
    LOG_FILE,
    "a",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    if not file_exists:

        writer.writerow([
            "Part_ID",
            "Timestamp",
            "Result",
            "Defect_Count",
            "Defect_Type",
            "Confidence",
            "Width_px",
            "Height_px",
            "Area_px2",
            "Severity"
        ])

    if detections:

        for detection in detections:

            writer.writerow([
                part_id,
                timestamp,
                inspection_result,
                len(detections),
                detection["class_name"],
                f"{detection['confidence'] * 100:.2f}",
                f"{detection['width']:.2f}",
                f"{detection['height']:.2f}",
                f"{detection['area']:.2f}",
                detection["severity"]
            ])

    else:

        writer.writerow([
            part_id,
            timestamp,
            inspection_result,
            0,
            "None",
            "0.00",
            "0.00",
            "0.00",
            "0.00",
            "None"
        ])


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 65)
print("                 INSPECTION RESULT")
print("=" * 65)

print(f"Part ID       : {part_id}")
print(f"Result        : {inspection_result}")
print(f"Defect Count  : {len(detections)}")

if detections:

    print()
    print("Defects:")

    for i, detection in enumerate(detections):

        print(
            f"  {i + 1}. "
            f"{detection['class_name']} | "
            f"Confidence: {detection['confidence'] * 100:.2f}% | "
            f"Size: {detection['width']:.2f} x "
            f"{detection['height']:.2f} px | "
            f"Area: {detection['area']:.2f} px² | "
            f"Severity: {detection['severity']}"
        )

else:

    print("Defects: None")


print()
print(f"Annotated image : {OUTPUT_IMAGE}")
print(f"Log file        : {LOG_FILE}")

print("=" * 65)
print("             INSPECTION COMPLETED")
print("=" * 65)