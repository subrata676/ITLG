from pathlib import Path
from collections import Counter

from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO

import cv2
import numpy as np


# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MAIN_DIR = (BASE_DIR.parent / "main").resolve()


# ---------------------------------------------------
# FLASK APP
# ---------------------------------------------------

app = Flask(__name__)


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = YOLO(str(BASE_DIR / "best.pt"))


# ---------------------------------------------------
# CLASS GROUPING LOGIC
# ---------------------------------------------------

group_map = {

    # Fine Tea
    "1B-1L-F": "fine",
    "1B-2L-F": "fine",

    # Added into fine
    "1B-1L": "fine",
    "1B-2L": "fine",

    # Coarse Tea
    "1B-3L": "coarse",
    "1B-4F": "coarse",

    # Banjhi
    "banjhi": "banjhi",

    # Rejected
    "other": "rejected",
    "unsure": "rejected"
}

# ---------------------------------------------------
# HOME ROUTE
# ---------------------------------------------------

@app.route("/")
@app.route("/app")
def home():
    return send_from_directory(MAIN_DIR, "index.html")


# ---------------------------------------------------
# PREDICTION ROUTE
# ---------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # -----------------------------------------------
    # CHECK IMAGE
    # -----------------------------------------------

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    # -----------------------------------------------
    # IMAGE DECODE
    # -----------------------------------------------

    image_bytes = file.read()

    npimg = np.frombuffer(image_bytes, np.uint8)

    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({
            "error": "Invalid image"
        }), 400

    # -----------------------------------------------
    # RUN YOLO
    # -----------------------------------------------

    results = model(
        frame,
        imgsz=640,
        conf=0.4,
        device=0
    )

    # -----------------------------------------------
    # RAW CLASS COUNTS
    # -----------------------------------------------

    raw_counts = Counter()

    for r in results:

        for box in r.boxes:

            cls = int(box.cls[0])

            class_name = model.names[cls]

            raw_counts[class_name] += 1

    # -----------------------------------------------
    # GROUP COUNTS
    # -----------------------------------------------

    group_counts = Counter()

    for cls_name, count in raw_counts.items():

        if cls_name in group_map:

            group = group_map[cls_name]

            group_counts[group] += count

    # -----------------------------------------------
    # TOTAL DETECTIONS
    # -----------------------------------------------

    total = sum(group_counts.values())

    if total == 0:
        return jsonify({
            "error": "No tea leaves detected"
        })

    # -----------------------------------------------
    # PERCENTAGES
    # -----------------------------------------------

    percentages = {}

    for group, count in group_counts.items():

        percent = (count / total) * 100

        percentages[group] = round(percent, 1)

    # Ensure missing groups show 0%

    for key in ["fine", "coarse", "banjhi", "rejected"]:

        if key not in percentages:
            percentages[key] = 0.0

        if key not in group_counts:
            group_counts[key] = 0

    # -----------------------------------------------
    # QUALITY SCORE
    # -----------------------------------------------

    quality_score = (
        percentages["fine"] * 1.0 +
        percentages["coarse"] * 0.4 -
        percentages["banjhi"] * 0.5 -
        percentages["rejected"] * 1.0
    )

    quality_score = round(max(0, quality_score), 1)

    # -----------------------------------------------
    # QUALITY STATUS
    # -----------------------------------------------

    if quality_score >= 75:
        quality_status = "Premium"

    elif quality_score >= 55:
        quality_status = "Good"

    elif quality_score >= 35:
        quality_status = "Average"

    else:
        quality_status = "Low Quality"

    # -----------------------------------------------
    # FINAL RESPONSE
    # -----------------------------------------------

    return jsonify({

        # Counts
        "fine_count": group_counts["fine"],
        "coarse_count": group_counts["coarse"],
        "banjhi_count": group_counts["banjhi"],
        "rejected_count": group_counts["rejected"],

        # Percentages
        "fine_percent": percentages["fine"],
        "coarse_percent": percentages["coarse"],
        "banjhi_percent": percentages["banjhi"],
        "reject_percent": percentages["rejected"],

        # Analytics
        "total_detected": total,
        "quality_score": quality_score,
        "quality_status": quality_status,

        # Raw detailed classes
        "raw_class_counts": raw_counts
    })


# ---------------------------------------------------
# RUN SERVER
# ---------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )