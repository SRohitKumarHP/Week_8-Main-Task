from ultralytics import YOLO
import cv2
import os

from logger import log_inspection
from measurement import calculate_measurement
from measurement import calculate_severity

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "models/yolov11s_best.pt"
IMAGE_PATH = "input/images/test_image.jpg"
OUTPUT_PATH = "output/annotated/inspection_result.jpg"

PART_ID = "PART_001"


# --------------------------------------------------
# Load YOLOv11s model
# --------------------------------------------------

print("Loading YOLOv11s model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")


# --------------------------------------------------
# Run inference
# --------------------------------------------------

print("Running inspection...")

results = model(IMAGE_PATH)


# --------------------------------------------------
# Extract detection information
# --------------------------------------------------

result = results[0]

boxes = result.boxes


# Number of detected defects
defect_count = len(boxes)


# --------------------------------------------------
# Get defect names and confidence
# --------------------------------------------------

defect_types = []
confidences = []


measurements = []


for box in boxes:

    class_id = int(box.cls[0])
    confidence = float(box.conf[0])

    class_name = model.names[class_id]

    # Get bounding-box coordinates
    coordinates = box.xyxy[0].tolist()

    x1, y1, x2, y2 = coordinates

    # Calculate measurement
    width, height, area = calculate_measurement(
        (x1, y1, x2, y2)
    )

    # Calculate severity
    severity = calculate_severity(area)

    # Store defect information
    defect_types.append(class_name)
    confidences.append(confidence)

    measurements.append({
        "class": class_name,
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


# --------------------------------------------------
# PASS / FAIL decision
# --------------------------------------------------

if defect_count == 0:

    inspection_result = "PASS"

else:

    inspection_result = "FAIL"


# --------------------------------------------------
# Create annotated image
# --------------------------------------------------

annotated_image = result.plot()

# --------------------------------------------------
# Log inspection
# --------------------------------------------------

log_inspection(
    part_id=PART_ID,
    result=inspection_result,
    defect_count=defect_count,
    measurements=measurements
)

# --------------------------------------------------
# Save annotated image
# --------------------------------------------------

os.makedirs("output/annotated", exist_ok=True)

cv2.imwrite(
    OUTPUT_PATH,
    annotated_image
)


# --------------------------------------------------
# Display inspection summary
# --------------------------------------------------

print()
print("=" * 50)
print("        METAL QUALITY INSPECTION")
print("=" * 50)

print(f"Part ID       : {PART_ID}")
print(f"Result        : {inspection_result}")
print(f"Defect Count  : {defect_count}")

print()

if defect_count > 0:

    print("Defect Measurements:")
    print("-" * 50)

    for index, measurement in enumerate(
        measurements,
        start=1
    ):

        print(
            f"Defect #{index}"
        )

        print(
            f"  Type       : {measurement['class']}"
        )

        print(
            f"  Confidence : "
            f"{measurement['confidence'] * 100:.2f}%"
        )

        print(
            f"  Width      : "
            f"{measurement['width']:.2f} px"
        )

        print(
            f"  Height     : "
            f"{measurement['height']:.2f} px"
        )

        print(
            f"  Area       : "
            f"{measurement['area']:.2f} px²"
        )

        print(
            f"  Severity   : "
            f"{measurement['severity']}"
        )

        print("-" * 50)

if defect_count > 0:

    print(f"Defect Types  : {', '.join(defect_types)}")

    print("Confidence    :")

    for defect, confidence in zip(defect_types, confidences):

        print(
            f"  {defect} : {confidence * 100:.2f}%"
        )

else:

    print("Defect Types  : None")


print("=" * 50)

print()
print(f"Annotated image saved to:")
print(OUTPUT_PATH)