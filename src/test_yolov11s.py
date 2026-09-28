from ultralytics import YOLO
import cv2
import os


# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "models/yolov11s_best.pt"
IMAGE_PATH = "input/images/test_image.jpg"
OUTPUT_PATH = "output/annotated/test_result.jpg"


# -----------------------------
# Load YOLOv11s model
# -----------------------------
print("Loading YOLOv11s model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")


# -----------------------------
# Run inference
# -----------------------------
print("Running inference...")

results = model(IMAGE_PATH)


# -----------------------------
# Get annotated image
# -----------------------------
annotated_image = results[0].plot()


# -----------------------------
# Save result
# -----------------------------
os.makedirs("output/annotated", exist_ok=True)

cv2.imwrite(OUTPUT_PATH, annotated_image)

print("Inference completed.")
print(f"Annotated image saved to: {OUTPUT_PATH}")