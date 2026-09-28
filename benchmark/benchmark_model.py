from ultralytics import YOLO
import cv2
import time
import statistics


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "models/yolov11s_best.pt"
VIDEO_PATH = "input/videos/2.mp4"


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
# Video information
# --------------------------------------------------

video_fps = cap.get(
    cv2.CAP_PROP_FPS
)

frame_count = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


print()
print("Video Information")
print("----------------------------")
print(f"Video FPS       : {video_fps:.2f}")
print(f"Total Frames    : {frame_count}")
print("----------------------------")


# --------------------------------------------------
# Benchmark variables
# --------------------------------------------------

inference_times = []

processed_frames = 0


# --------------------------------------------------
# Run benchmark
# --------------------------------------------------

print()
print("Running benchmark...")
print()


while True:

    ret, frame = cap.read()

    if not ret:

        break


    # ----------------------------------------------
    # Start timer
    # ----------------------------------------------

    start_time = time.perf_counter()


    # ----------------------------------------------
    # YOLO inference
    # ----------------------------------------------

    model(
        frame,
        verbose=False
    )


    # ----------------------------------------------
    # End timer
    # ----------------------------------------------

    end_time = time.perf_counter()


    # ----------------------------------------------
    # Calculate inference time
    # ----------------------------------------------

    inference_time = (
        end_time - start_time
    )


    inference_times.append(
        inference_time
    )

    processed_frames += 1


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

cap.release()


# --------------------------------------------------
# Calculate statistics
# --------------------------------------------------

if processed_frames == 0:

    print("No frames were processed.")

    exit()


average_time = statistics.mean(
    inference_times
)

minimum_time = min(
    inference_times
)

maximum_time = max(
    inference_times
)

median_time = statistics.median(
    inference_times
)

average_fps = 1 / average_time


# --------------------------------------------------
# Display benchmark
# --------------------------------------------------

print()
print("=" * 55)
print("              YOLOv11s BENCHMARK")
print("=" * 55)

print(
    f"Frames Processed       : "
    f"{processed_frames}"
)

print(
    f"Average Inference     : "
    f"{average_time * 1000:.2f} ms"
)

print(
    f"Minimum Inference     : "
    f"{minimum_time * 1000:.2f} ms"
)

print(
    f"Maximum Inference     : "
    f"{maximum_time * 1000:.2f} ms"
)

print(
    f"Median Inference      : "
    f"{median_time * 1000:.2f} ms"
)

print(
    f"Average Inference FPS : "
    f"{average_fps:.2f}"
)

print("=" * 55)