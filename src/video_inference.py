from ultralytics import YOLO
import cv2
import time
import os


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "models/yolov11s_best.pt"
VIDEO_PATH = "input/videos/2.mp4"

OUTPUT_PATH = "output/annotated/inspection_video.mp4"


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("Loading YOLOv11s model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")


# --------------------------------------------------
# Open video
# --------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("ERROR: Could not open video.")

    exit()


# --------------------------------------------------
# Get video properties
# --------------------------------------------------

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

input_fps = cap.get(
    cv2.CAP_PROP_FPS
)


print()
print("Video Information")
print("----------------------------")
print(f"Width       : {width}")
print(f"Height      : {height}")
print(f"Input FPS   : {input_fps:.2f}")
print("----------------------------")


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

os.makedirs(
    "output/annotated",
    exist_ok=True
)


# --------------------------------------------------
# Video writer
# --------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    input_fps,
    (width, height)
)


# --------------------------------------------------
# Processing variables
# --------------------------------------------------

frame_count = 0

total_inference_time = 0

display_fps = 0


# --------------------------------------------------
# Process video
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:

        break


    frame_count += 1


    # ----------------------------------------------
    # Start timer
    # ----------------------------------------------

    start_time = time.perf_counter()


    # ----------------------------------------------
    # YOLO inference
    # ----------------------------------------------

    results = model(
        frame,
        verbose=False
    )


    # ----------------------------------------------
    # End timer
    # ----------------------------------------------

    end_time = time.perf_counter()

    inference_time = (
        end_time - start_time
    )

    total_inference_time += inference_time


    # ----------------------------------------------
    # Calculate FPS
    # ----------------------------------------------

    if inference_time > 0:

        display_fps = 1 / inference_time


    # ----------------------------------------------
    # Get result
    # ----------------------------------------------

    result = results[0]

    defect_count = len(
        result.boxes
    )


    # ----------------------------------------------
    # PASS / FAIL
    # ----------------------------------------------

    if defect_count > 0:

        inspection_result = "FAIL"

    else:

        inspection_result = "PASS"


    # ----------------------------------------------
    # Annotate frame
    # ----------------------------------------------

    annotated_frame = result.plot()


    # ----------------------------------------------
    # Add inspection information
    # ----------------------------------------------

    cv2.putText(
        annotated_frame,
        f"Result: {inspection_result}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255)
        if inspection_result == "FAIL"
        else (0, 255, 0),
        2
    )


    cv2.putText(
        annotated_frame,
        f"FPS: {display_fps:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.putText(
        annotated_frame,
        f"Inference: {inference_time * 1000:.1f} ms",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ----------------------------------------------
    # Write frame to output video
    # ----------------------------------------------

    out.write(
        annotated_frame
    )


    # ----------------------------------------------
    # Display frame
    # ----------------------------------------------

    cv2.imshow(
        "Metal Quality Inspection",
        annotated_frame
    )


    # ----------------------------------------------
    # Exit with Q
    # ----------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

cap.release()

out.release()

cv2.destroyAllWindows()


# --------------------------------------------------
# Final statistics
# --------------------------------------------------

if frame_count > 0:

    average_inference_time = (
        total_inference_time /
        frame_count
    )

    average_fps = (
        1 /
        average_inference_time
    )

else:

    average_inference_time = 0
    average_fps = 0


print()
print("=" * 50)
print("       VIDEO INSPECTION COMPLETE")
print("=" * 50)

print(f"Frames Processed     : {frame_count}")

print(
    f"Average Inference    : "
    f"{average_inference_time * 1000:.2f} ms"
)

print(
    f"Average Inference FPS: "
    f"{average_fps:.2f}"
)

print()
print("Output Video:")
print(OUTPUT_PATH)

print("=" * 50)