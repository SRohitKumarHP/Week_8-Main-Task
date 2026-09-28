from ultralytics import YOLO
import os


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "models/yolov11s_best.pt"


# --------------------------------------------------
# Load YOLOv11s model
# --------------------------------------------------

print("Loading YOLOv11s model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")


# --------------------------------------------------
# Export to ONNX
# --------------------------------------------------

print()
print("Exporting YOLOv11s to ONNX...")
print()


onnx_path = model.export(
    format="onnx",
    imgsz=640,
    simplify=True,
    opset=12
)


# --------------------------------------------------
# Display result
# --------------------------------------------------

print()
print("=" * 50)
print("        ONNX EXPORT COMPLETE")
print("=" * 50)

print(f"ONNX Model: {onnx_path}")

print("=" * 50)