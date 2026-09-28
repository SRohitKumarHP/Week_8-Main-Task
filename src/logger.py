import csv
import os
from datetime import datetime


LOG_FILE = "output/logs/inspection_log.csv"


def log_inspection(
    part_id,
    result,
    defect_count,
    measurements
):
    """
    Save inspection results and defect measurements
    to the CSV log.
    """

    # Create log directory
    os.makedirs("output/logs", exist_ok=True)

    # Current timestamp
    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # Check if CSV already exists
    file_exists = os.path.exists(LOG_FILE)

    with open(
        LOG_FILE,
        mode="a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        # Create header for a new CSV
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

        # If there are no defects
        if not measurements:

            writer.writerow([
                part_id,
                timestamp,
                result,
                0,
                "None",
                "None",
                "None",
                "None",
                "None",
                "None"
            ])

        # If defects exist
        else:

            for measurement in measurements:

                writer.writerow([
                    part_id,
                    timestamp,
                    result,
                    defect_count,
                    measurement["class"],
                    f"{measurement['confidence'] * 100:.2f}%",
                    f"{measurement['width']:.2f}",
                    f"{measurement['height']:.2f}",
                    f"{measurement['area']:.2f}",
                    measurement["severity"]
                ])

    print(f"Inspection logged: {LOG_FILE}")