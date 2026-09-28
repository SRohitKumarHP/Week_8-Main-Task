def calculate_measurement(box):
    """
    Calculate bounding-box dimensions and area.

    Parameters:
        box: YOLO bounding box

    Returns:
        width
        height
        area
    """

    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    area = width * height

    return width, height, area

def calculate_severity(area):
    """
    Calculate defect severity based on bounding-box area.

    These thresholds are demonstration thresholds
    and should be calibrated using your actual dataset
    for a production system.
    """

    if area < 500:
        return "LOW"

    elif area < 2000:
        return "MEDIUM"

    else:
        return "HIGH"