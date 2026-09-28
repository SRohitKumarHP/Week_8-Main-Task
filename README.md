# Week_8-Main-Task
# Week 8 — Metal Part Quality Inspection

## Pipeline Integration, Optimization & Deployment

A complete **YOLOv11s-based metal surface inspection system** for detecting defects in metal parts.

The project integrates:

- YOLOv11s defect detection
- Image inspection
- Video inspection
- Real-time inspection
- PASS / FAIL decision
- Defect measurement
- Severity classification
- Inspection logging
- ONNX model export
- Model benchmarking
- Local Streamlit dashboard
- User authentication and role management

---

# 1. Project Folder Structure

```text
Week_8_Metal_Inspection/
│
├── models/
│   ├── yolov11s_best.pt
│   └── yolov11s_best.onnx
│
├── input/
│   ├── images/
│   │   └── test_image.jpg
│   │
│   └── videos/
│       ├── test_video.mp4
│       └── 2.mp4
│
├── output/
│   ├── annotated/
│   ├── logs/
│   └── reports/
│
├── src/
│   ├── test_yolov11s.py
│   ├── inspection.py
│   ├── measurement.py
│   ├── logger.py
│   ├── metal_inspection.py
│   ├── video_inference.py
│   └── realtime_inspection.py
│
├── benchmark/
│   ├── benchmark_model.py
│   └── benchmark_onnx.py
│
├── export/
│   └── export_onnx.py
│
├── app/
│   ├── dashboard.py
│   ├── auth.py
│   └── users.db
│
├── requirements.txt
└── README.md
```

> `app/users.db` is created locally by the authentication system and should normally be excluded from GitHub using `.gitignore`.

---

# 2. YOLO Detection Classes

The YOLOv11s model detects the following metal defects:

```text
crazing
inclusion
patches
pitted_surface
rolled-in_scale
scratches
```

---

# 3. Requirements

### Software

- Windows
- Python 3.10
- VS Code
- Git
- FFmpeg
- YOLOv11s model

### Hardware

The project can run on CPU.

Test environment:

```text
CPU: 12th Gen Intel Core i5-1240P
GPU: Not required
```

---

# 4. Setup the Project

## Step 1 — Open the Project

Open the project folder in VS Code:

```text
D:\Machine Wise\Internship\Week_8\Week_8_Metal_Inspection
```

Open the VS Code terminal:

```text
Terminal → New Terminal
```

---

# 5. Create Virtual Environment

Run:

```powershell
python -m venv venv
```

This creates an isolated Python environment for the project.

---

# 6. Activate Virtual Environment

For Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

If Command Prompt is being used:

```cmd
venv\Scripts\activate
```

After activation, the terminal should show:

```text
(venv)
```

---

# 7. Install Required Libraries

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the project dependencies:

```powershell
pip install -r requirements.txt
```

---

# 8. Verify Installation

Check Python:

```powershell
python --version
```

Check PyTorch:

```powershell
python -c "import torch; print(torch.__version__)"
```

Check Ultralytics:

```powershell
python -c "import ultralytics; print(ultralytics.__version__)"
```

Check ONNX:

```powershell
python -c "import onnx; print(onnx.__version__)"
```

Check ONNX Runtime:

```powershell
python -c "import onnxruntime; print(onnxruntime.__version__)"
```

---

# 9. Add the YOLO Model

Place the trained YOLOv11s model inside:

```text
models/
```

Required model:

```text
models/yolov11s_best.pt
```

The ONNX model can also be stored here:

```text
models/yolov11s_best.onnx
```

---

# 10. Add Input Image

Place the test image inside:

```text
input/images/
```

Example:

```text
input/images/test_image.jpg
```

---

# 11. Test YOLOv11s

Run:

```powershell
python src\test_yolov11s.py
```

This verifies that the YOLOv11s model is working correctly.

The annotated result is saved inside:

```text
output/annotated/
```

---

# 12. Run the Complete Image Inspection

Run:

```powershell
python src\metal_inspection.py
```

The complete pipeline performs:

```text
Input Image
     ↓
YOLOv11s Detection
     ↓
Defect Detection
     ↓
Confidence
     ↓
Bounding Box
     ↓
Measurement
     ↓
Severity
     ↓
PASS / FAIL
     ↓
Save Result
     ↓
Save Log
```

---

# 13. Inspection Output

The system provides:

```text
Defect Count
Defect Type
Confidence
Width
Height
Area
Severity
PASS / FAIL
```

Example:

```text
Defect Count: 3

Defect 1:
Type: inclusion
Confidence: 64.34%
Severity: HIGH

Defect 2:
Type: inclusion
Confidence: 58.64%
Severity: HIGH

Defect 3:
Type: inclusion
Confidence: 47.66%
Severity: HIGH

Result: FAIL
```

---

# 14. Output Files

Annotated images are saved in:

```text
output/annotated/
```

Inspection logs are saved in:

```text
output/logs/
```

Example:

```text
output/
│
├── annotated/
│   └── final_inspection.jpg
│
├── logs/
│   ├── inspection_log.csv
│   └── final_inspection_log.csv
│
└── reports/
```

---

# 15. Run Video Inspection

Place the video inside:

```text
input/videos/
```

Example:

```text
input/videos/test_video.mp4
```

Run:

```powershell
python src\video_inference.py
```

The annotated video is saved in:

```text
output/annotated/
```

---

# 16. Run Real-Time Inspection

Run:

```powershell
python src\realtime_inspection.py
```

The program provides two options:

```text
1. Video File
2. Camera
```

For video:

```text
Select: 1
```

Then provide the video path, for example:

```text
input/videos/2.mp4
```

For camera:

```text
Select: 2
```

The system displays:

```text
PASS / FAIL
Defect Count
FPS
Inference Time
YOLO Bounding Boxes
Defect Classes
Confidence
```

The processed video can be saved to:

```text
output/annotated/realtime_inspection.mp4
```

---

# 17. ONNX Export

The trained YOLOv11s model can be exported to ONNX.

Run:

```powershell
python export\export_onnx.py
```

Output:

```text
models/yolov11s_best.onnx
```

---

# 18. Model Benchmarking

### PyTorch Benchmark

Run:

```powershell
python benchmark\benchmark_model.py
```

### ONNX Benchmark

Run:

```powershell
python benchmark\benchmark_onnx.py
```

The benchmark measures:

```text
Average Inference Time
Median Inference Time
Minimum Inference Time
Maximum Inference Time
Average FPS
```

---

# 19. Run the Dashboard

The project includes a local Streamlit dashboard.

Run:

```powershell
streamlit run app\dashboard.py
```

The dashboard will open in the browser.

Typical local address:

```text
http://localhost:8501
```

---

# 20. Dashboard Features

The dashboard provides:

- User login
- Role-based access
- Image upload
- YOLOv11s detection
- Confidence threshold
- Annotated image
- Defect count
- Defect type
- Confidence
- Measurement
- Severity
- PASS / FAIL
- Inspection history
- Inspection logging
- Admin user management
- Admin record deletion
- Logout

---

# 21. Login

The default administrator account is:

```text
Username: admin
Password: admin123
Role: Admin
```

Change the default password when using the system outside the development environment.

---

# 22. User Roles

### Admin

Can:

- Perform inspections
- View inspection history
- Create users
- Delete users
- Delete inspection records
- Manage the system

### Inspector

Can:

- Perform inspections
- View inspection history
- Generate inspection results

### Viewer

Can:

- View inspection history
- View inspection results

Viewer users cannot perform inspections or modify records.

---

# 23. Complete Execution Order

For a fresh setup, follow this order:

```text
1. Open project in VS Code
        ↓
2. Open terminal
        ↓
3. Create virtual environment
        ↓
4. Activate virtual environment
        ↓
5. Install requirements
        ↓
6. Verify installation
        ↓
7. Place YOLOv11s model
        ↓
8. Place test image/video
        ↓
9. Test YOLOv11s
        ↓
10. Run image inspection
        ↓
11. Run video inspection
        ↓
12. Run real-time inspection
        ↓
13. Export ONNX model
        ↓
14. Run benchmarks
        ↓
15. Start Streamlit dashboard
        ↓
16. Login
        ↓
17. Perform inspection
        ↓
18. View final result and logs
```

---

# 24. Final Pipeline

```text
                INPUT
                  │
        ┌─────────┴─────────┐
        │                   │
      Image               Video
        │                   │
        └─────────┬─────────┘
                  ↓
            YOLOv11s Model
                  ↓
          Defect Detection
                  ↓
       ┌──────────┼──────────┐
       │          │          │
   Class       Confidence   Box
       │          │          │
       └──────────┼──────────┘
                  ↓
             Measurement
                  ↓
              Severity
                  ↓
             PASS / FAIL
                  ↓
        ┌─────────┴─────────┐
        │                   │
   Annotated Output       CSV Log
        │
        ↓
   Streamlit Dashboard
```

---

# 25. Important Notes

### CPU Execution

This project is configured to run on CPU and does not require CUDA.

### Measurement

The current measurement is based on bounding-box pixel dimensions:

```text
Width  = x2 - x1
Height = y2 - y1
Area   = Width × Height
```

The current severity thresholds are demonstration thresholds based on pixel area.

They are not physical engineering limits.

### Local Deployment

The dashboard runs locally using Streamlit.

No cloud service is required for the application.

---

# 26. Final Result

The completed system provides an end-to-end metal inspection pipeline:

```text
YOLOv11s
   ↓
Defect Detection
   ↓
Measurement
   ↓
Severity
   ↓
PASS / FAIL
   ↓
Logging
   ↓
Video / Real-Time Inspection
   ↓
ONNX Export
   ↓
Benchmarking
   ↓
Streamlit Dashboard
   ↓
Authentication & User Management
```

---

# 27. Project Status

```text
[✓] YOLOv11s Detection
[✓] Image Inspection
[✓] PASS / FAIL
[✓] Defect Measurement
[✓] Severity Classification
[✓] CSV Logging
[✓] Video Inspection
[✓] Real-Time Inspection
[✓] ONNX Export
[✓] Model Benchmarking
[✓] Streamlit Dashboard
[✓] Authentication
[✓] Role Management
[✓] Admin User Management
[✓] Inspection Record Management
```

---

# Conclusion

The Week 8 project integrates the trained YOLOv11s model into a complete local metal-part inspection system with image, video, real-time, logging, benchmarking, ONNX export, and Streamlit deployment capabilities.
