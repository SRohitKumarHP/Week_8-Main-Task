import cv2
import time
import os
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/yolov11s_best.pt"

VIDEO_PATH = "input/videos/2.mp4"

OUTPUT_PATH = "output/annotated/realtime_inspection.mp4"

CONFIDENCE_THRESHOLD = 0.25


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 65)
print("           YOLOv11s REAL-TIME INSPECTION")
print("=" * 65)

print("Loading YOLOv11s model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")
print()


# ============================================================
# SELECT INPUT
# ============================================================

print("Select input source:")
print()
print("1. Video file")
print("2. Camera")
print()

choice = input("Enter your choice (1/2): ").strip()


if choice == "1":

    source = VIDEO_PATH

    print()
    print(f"Input video: {source}")

    cap = cv2.VideoCapture(source)

elif choice == "2":

    source = 0

    print()
    print("Opening camera...")

    cap = cv2.VideoCapture(source)

else:

    print("Invalid choice.")
    exit()


# ============================================================
# CHECK INPUT
# ============================================================

if not cap.isOpened():

    print()
    print("ERROR: Could not open input source.")
    exit()


# ============================================================
# VIDEO INFORMATION
# ============================================================

source_fps = cap.get(cv2.CAP_PROP_FPS)

if source_fps <= 0:
    source_fps = 30.0

frame_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

frame_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


# ============================================================
# VIDEO WRITER
# ============================================================

writer = None

if choice == "1":

    os.makedirs(
        "output/annotated",
        exist_ok=True
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        source_fps,
        (frame_width, frame_height)
    )


# ============================================================
# FPS VARIABLES
# ============================================================

frame_count = 0

fps_start_time = time.time()

display_fps = 0.0


print()
print("=" * 65)
print("                 INSPECTION STARTED")
print("=" * 65)

print(f"Resolution : {frame_width} x {frame_height}")
print(f"Source FPS : {source_fps:.2f}")
print()
print("Press Q to stop.")
print()


# ============================================================
# REAL-TIME LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # --------------------------------------------------------
    # YOLO INFERENCE
    # --------------------------------------------------------

    start_time = time.perf_counter()

    results = model(
        frame,
        imgsz=640,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    inference_time = (
        time.perf_counter() - start_time
    ) * 1000

    result = results[0]

    # --------------------------------------------------------
    # DEFECT COUNT
    # --------------------------------------------------------

    defect_count = 0

    if result.boxes is not None:

        defect_count = len(
            result.boxes
        )

    # --------------------------------------------------------
    # PASS / FAIL
    # --------------------------------------------------------

    if defect_count > 0:
        inspection_result = "FAIL"
    else:
        inspection_result = "PASS"

    # --------------------------------------------------------
    # DRAW YOLO ANNOTATIONS
    # --------------------------------------------------------

    annotated_frame = result.plot()

    # --------------------------------------------------------
    # CALCULATE DISPLAY FPS
    # --------------------------------------------------------

    elapsed = time.time() - fps_start_time

    if elapsed >= 1.0:

        display_fps = (
            frame_count / elapsed
        )

        frame_count = 0
        fps_start_time = time.time()

    # --------------------------------------------------------
    # DISPLAY INFORMATION
    # --------------------------------------------------------

    cv2.putText(
        annotated_frame,
        f"Result: {inspection_result}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0)
        if inspection_result == "PASS"
        else (0, 0, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Defects: {defect_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"FPS: {display_fps:.2f}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Inference: {inference_time:.2f} ms",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # SAVE VIDEO
    # --------------------------------------------------------

    if writer is not None:

        writer.write(
            annotated_frame
        )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    cv2.imshow(
        "YOLOv11s Metal Inspection",
        annotated_frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

if writer is not None:
    writer.release()

cv2.destroyAllWindows()


print()
print("=" * 65)
print("              INSPECTION COMPLETED")
print("=" * 65)

if choice == "1":
    print(f"Output video: {OUTPUT_PATH}")

print("=" * 65)