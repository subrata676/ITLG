from flask import Flask, request, jsonify
from flask import send_file
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Load your trained model
model = YOLO("server/best.pt")

@app.route("/app")
def app_ui():
    return send_file("../main/index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    results = model(img, device=0, imgsz=416, conf=0.5)

    sb, l1, l2 = 0, 0, 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0: sb += 1
            elif cls == 1: l1 += 1
            elif cls == 2: l2 += 1

    total = sb + l1 + l2
    PN = (sb / total * 100) if total > 0 else 0

    grade = "Grade 1" if PN > 70 else "Grade 2" if PN > 40 else "Grade 3"

    return jsonify({
        "sb": sb,
        "l1": l1,
        "l2": l2,
        "PN": PN,
        "grade": grade
    })

app.run(host="0.0.0.0", port=5000)