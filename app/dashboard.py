import streamlit as st
import pandas as pd
import cv2
import os
from datetime import datetime
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

from auth import (
    initialize_database,
    authenticate_user,
    create_user,
    get_users,
    delete_user
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "yolov11s_best.pt"

INPUT_DIR = BASE_DIR / "input" / "images"

OUTPUT_DIR = BASE_DIR / "output" / "annotated"

LOG_DIR = BASE_DIR / "output" / "logs"

LOG_FILE = LOG_DIR / "inspection_log.csv"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

INPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATABASE
# ============================================================

initialize_database()


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Metal Inspection Dashboard",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "role" not in st.session_state:
    st.session_state.role = ""


# ============================================================
# LOGIN PAGE
# ============================================================

def login_page():

    st.title("🔐 Metal Inspection System")

    st.subheader("Login")

    st.write(
        "Please enter your credentials to access the inspection dashboard."
    )

    with st.form("login_form"):

        username = st.text_input(
            "Username"
        )

        password = st.text_input(
            "Password",
            type="password"
        )

        login_button = st.form_submit_button(
            "Login",
            width="stretch"
        )

        if login_button:

            if not username or not password:

                st.error(
                    "Please enter username and password."
                )

            else:

                user = authenticate_user(
                    username,
                    password
                )

                if user:

                    st.session_state.authenticated = True
                    st.session_state.username = user["username"]
                    st.session_state.role = user["role"]

                    st.success(
                        "Login successful."
                    )

                    st.rerun()

                else:

                    st.error(
                        "Invalid username or password."
                    )


# ============================================================
# SHOW LOGIN IF NOT AUTHENTICATED
# ============================================================

if not st.session_state.authenticated:

    login_page()

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return YOLO(
        str(MODEL_PATH)
    )


model = load_model()


# ============================================================
# CSV LOG FUNCTIONS
# ============================================================

def load_history():

    if not LOG_FILE.exists():

        return pd.DataFrame()

    try:

        df = pd.read_csv(
            LOG_FILE
        )

        return df

    except Exception:

        return pd.DataFrame()


def save_record(record):

    df_new = pd.DataFrame(
        [record]
    )

    if LOG_FILE.exists():

        try:

            df_old = pd.read_csv(
                LOG_FILE
            )

            df = pd.concat(
                [
                    df_old,
                    df_new
                ],
                ignore_index=True
            )

        except Exception:

            df = df_new

    else:

        df = df_new

    df.to_csv(
        LOG_FILE,
        index=False
    )


# ============================================================
# DELETE SELECTED RECORDS
# ============================================================

def delete_records(row_indexes):

    if not row_indexes:
        return

    if not LOG_FILE.exists():
        return

    try:

        df = pd.read_csv(
            LOG_FILE
        )

        df = df.drop(
            index=row_indexes
        )

        df = df.reset_index(
            drop=True
        )

        df.to_csv(
            LOG_FILE,
            index=False
        )

    except Exception as error:

        st.error(
            f"Unable to delete records: {error}"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ Control Panel")

    st.write(
        f"**User:** {st.session_state.username}"
    )

    st.write(
        f"**Role:** {st.session_state.role}"
    )

    st.divider()

    if st.button(
        "🚪 Logout",
        width="stretch"
    ):

        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.role = ""

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🔍 Metal Part Quality Inspection"
)

st.caption(
    "YOLOv11s-based local inspection and monitoring system"
)

st.divider()


# ============================================================
# ADMIN USER MANAGEMENT
# ============================================================

if st.session_state.role == "Admin":

    st.header(
        "👥 User Management"
    )

    users = get_users()

    if users:

        users_df = pd.DataFrame(
            users,
            columns=[
                "ID",
                "Username",
                "Role",
                "Created At"
            ]
        )

        st.dataframe(
            users_df,
            width="stretch",
            hide_index=True
        )

    st.subheader(
        "Create New User"
    )

    with st.form(
        "create_user_form"
    ):

        new_username = st.text_input(
            "Username",
            key="new_username"
        )

        new_password = st.text_input(
            "Password",
            type="password",
            key="new_password"
        )

        new_role = st.selectbox(
            "Role",
            [
                "Admin",
                "Inspector",
                "Viewer"
            ]
        )

        create_button = st.form_submit_button(
            "Create User",
            width="stretch"
        )

        if create_button:

            if not new_username or not new_password:

                st.error(
                    "Username and password are required."
                )

            else:

                success = create_user(
                    new_username.strip(),
                    new_password,
                    new_role
                )

                if success:

                    st.success(
                        f"User '{new_username}' created successfully."
                    )

                    st.rerun()

                else:

                    st.error(
                        "Username already exists."
                    )


    # --------------------------------------------------------
    # DELETE USERS
    # --------------------------------------------------------

    st.subheader(
        "Delete User"
    )

    users = get_users()

    deletable_users = [
        user[1]
        for user in users
        if user[1] != st.session_state.username
    ]

    if deletable_users:

        selected_user = st.selectbox(
            "Select User",
            deletable_users,
            key="delete_user_select"
        )

        if st.button(
            "🗑️ Delete Selected User",
            width="stretch"
        ):

            success, message = delete_user(
                selected_user,
                st.session_state.username
            )

            if success:

                st.success(
                    message
                )

                st.rerun()

            else:

                st.error(
                    message
                )

    else:

        st.info(
            "No other users are available to delete."
        )

    st.divider()


# ============================================================
# DASHBOARD SUMMARY
# ============================================================

history_df = load_history()

col1, col2, col3, col4 = st.columns(4)

if history_df.empty:

    total_inspections = 0
    total_pass = 0
    total_fail = 0
    total_defects = 0

else:

    total_inspections = len(
        history_df
    )

    if "Result" in history_df.columns:

        total_pass = (
            history_df["Result"]
            .astype(str)
            .str.upper()
            .eq("PASS")
            .sum()
        )

        total_fail = (
            history_df["Result"]
            .astype(str)
            .str.upper()
            .eq("FAIL")
            .sum()
        )

    else:

        total_pass = 0
        total_fail = 0

    if "Defect_Count" in history_df.columns:

        total_defects = pd.to_numeric(
            history_df["Defect_Count"],
            errors="coerce"
        ).fillna(0).sum()

    else:

        total_defects = 0


with col1:

    st.metric(
        "Total Inspections",
        total_inspections
    )


with col2:

    st.metric(
        "PASS",
        total_pass
    )


with col3:

    st.metric(
        "FAIL",
        total_fail
    )


with col4:

    st.metric(
        "Total Defects",
        int(total_defects)
    )


st.divider()


# ============================================================
# INSPECTION SECTION
# ============================================================

st.header(
    "📷 Inspection"
)


if st.session_state.role in [
    "Admin",
    "Inspector"
]:

    uploaded_file = st.file_uploader(
        "Upload Metal Part Image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp"
        ]
    )

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.25,
        step=0.05
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Original Image"
            )

            st.image(
                image,
                width="stretch"
            )


        if st.button(
            "🔍 Inspect Image",
            width="stretch"
        ):

            with st.spinner(
                "Running YOLOv11s inspection..."
            ):

                results = model.predict(
                    source=image,
                    conf=confidence_threshold,
                    verbose=False
                )

            result = results[0]

            detections = []

            if result.boxes is not None:

                for box in result.boxes:

                    confidence = float(
                        box.conf[0]
                    )

                    class_id = int(
                        box.cls[0]
                    )

                    class_name = result.names[
                        class_id
                    ]

                    x1, y1, x2, y2 = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                    )

                    width = float(
                        x2 - x1
                    )

                    height = float(
                        y2 - y1
                    )

                    area = float(
                        width * height
                    )

                    if area < 500:

                        severity = "LOW"

                    elif area < 2000:

                        severity = "MEDIUM"

                    else:

                        severity = "HIGH"

                    detections.append(
                        {
                            "Class": class_name,
                            "Confidence": confidence,
                            "Width_px": width,
                            "Height_px": height,
                            "Area_px2": area,
                            "Severity": severity
                        }
                    )


            defect_count = len(
                detections
            )

            if defect_count == 0:

                inspection_result = "PASS"

            else:

                inspection_result = "FAIL"


            # ------------------------------------------------
            # ANNOTATED IMAGE
            # ------------------------------------------------

            annotated = result.plot()

            annotated_rgb = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            )

            annotated_pil = Image.fromarray(
                annotated_rgb
            )


            with col2:

                st.subheader(
                    "Inspection Result"
                )

                st.image(
                    annotated_pil,
                    width="stretch"
                )


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            if inspection_result == "PASS":

                st.success(
                    "✅ PASS — No defects detected."
                )

            else:

                st.error(
                    f"❌ FAIL — {defect_count} defect(s) detected."
                )


            # ------------------------------------------------
            # DETECTION DETAILS
            # ------------------------------------------------

            st.subheader(
                "Detection Details"
            )

            if detections:

                detection_df = pd.DataFrame(
                    detections
                )

                display_df = detection_df.copy()

                display_df[
                    "Confidence"
                ] = (
                    display_df[
                        "Confidence"
                    ] * 100
                ).round(2).astype(str) + "%"

                display_df[
                    "Width_px"
                ] = display_df[
                    "Width_px"
                ].round(2)

                display_df[
                    "Height_px"
                ] = display_df[
                    "Height_px"
                ].round(2)

                display_df[
                    "Area_px2"
                ] = display_df[
                    "Area_px2"
                ].round(2)

                st.dataframe(
                    display_df,
                    width="stretch",
                    hide_index=True
                )

            else:

                st.info(
                    "No defects detected."
                )


            # ------------------------------------------------
            # SAVE ANNOTATED IMAGE
            # ------------------------------------------------

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            output_filename = (
                f"inspection_{timestamp}.jpg"
            )

            output_path = (
                OUTPUT_DIR /
                output_filename
            )

            cv2.imwrite(
                str(output_path),
                annotated
            )


            # ------------------------------------------------
            # LOG RECORD
            # ------------------------------------------------

            if detections:

                defect_types = ", ".join(
                    [
                        detection["Class"]
                        for detection in detections
                    ]
                )

                confidences = ", ".join(
                    [
                        f"{detection['Confidence'] * 100:.2f}%"
                        for detection in detections
                    ]
                )

                widths = ", ".join(
                    [
                        f"{detection['Width_px']:.2f}"
                        for detection in detections
                    ]
                )

                heights = ", ".join(
                    [
                        f"{detection['Height_px']:.2f}"
                        for detection in detections
                    ]
                )

                areas = ", ".join(
                    [
                        f"{detection['Area_px2']:.2f}"
                        for detection in detections
                    ]
                )

                severities = ", ".join(
                    [
                        detection["Severity"]
                        for detection in detections
                    ]
                )

            else:

                defect_types = "None"
                confidences = "None"
                widths = "0"
                heights = "0"
                areas = "0"
                severities = "NONE"


            record = {

                "Part_ID":
                    uploaded_file.name,

                "Timestamp":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "Result":
                    inspection_result,

                "Defect_Count":
                    defect_count,

                "Defect_Type":
                    defect_types,

                "Confidence":
                    confidences,

                "Width_px":
                    widths,

                "Height_px":
                    heights,

                "Area_px2":
                    areas,

                "Severity":
                    severities,

                "Username":
                    st.session_state.username,

                "Role":
                    st.session_state.role
            }


            save_record(
                record
            )


            st.success(
                "Inspection completed and record saved."
            )

            st.info(
                f"Annotated image saved to: {output_path}"
            )


else:

    st.info(
        "Viewer accounts have read-only access to inspection history."
    )


st.divider()


# ============================================================
# INSPECTION HISTORY
# ============================================================

st.header(
    "📊 Inspection History"
)

history_df = load_history()

if history_df.empty:

    st.info(
        "No inspection records available."
    )

else:

    st.dataframe(
        history_df,
        width="stretch",
        hide_index=True
    )


# ============================================================
# ADMIN RECORD MANAGEMENT
# ============================================================

if st.session_state.role == "Admin":

    st.divider()

    st.header(
        "🗑️ Inspection Record Management"
    )

    st.warning(
        "Admin control: deleting records permanently removes them from the inspection history CSV."
    )

    history_df = load_history()

    if history_df.empty:

        st.info(
            "There are no inspection records to delete."
        )

    else:

        # Create display IDs without changing the CSV itself.
        record_options = []

        for index, row in history_df.iterrows():

            result = str(
                row.get(
                    "Result",
                    ""
                )
            )

            part_id = str(
                row.get(
                    "Part_ID",
                    ""
                )
            )

            timestamp = str(
                row.get(
                    "Timestamp",
                    ""
                )
            )

            record_label = (
                f"Record {index + 1} | "
                f"{part_id} | "
                f"{timestamp} | "
                f"{result}"
            )

            record_options.append(
                (
                    record_label,
                    index
                )
            )


        selected_records = st.multiselect(
            "Select records to delete",
            options=[
                label
                for label, index in record_options
            ]
        )


        selected_indexes = [
            index
            for label, index in record_options
            if label in selected_records
        ]


        if selected_indexes:

            st.write(
                f"Selected records: {len(selected_indexes)}"
            )

            confirm_delete = st.checkbox(
                "I confirm that I want to permanently delete the selected records."
            )

            if st.button(
                "🗑️ Delete Selected Records",
                width="stretch"
            ):

                if not confirm_delete:

                    st.error(
                        "Please confirm the deletion first."
                    )

                else:

                    delete_records(
                        selected_indexes
                    )

                    st.success(
                        f"{len(selected_indexes)} record(s) deleted successfully."
                    )

                    st.rerun()

        else:

            st.info(
                "Select one or more records above to delete them."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Metal Inspection System | YOLOv11s | Local Windows Deployment"
)