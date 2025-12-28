from ultralytics import YOLO
import cv2
import numpy as np

from sort.sort import Sort
from util import get_car, read_license_plate, write_csv


# =========================
# INITIALIZE
# =========================
results = {}
mot_tracker = Sort()

# Load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# Load your video
cap = cv2.VideoCapture('C:/Users/deonb/Videos/cars.mp4')

vehicles = [2, 3, 5, 7]  # car, motorbike, bus, truck

frame_nmr = -1


# =========================
# PROCESS VIDEO
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_nmr += 1

    # Always initialize frame dict
    results[frame_nmr] = {}

    # -------------------------
    # VEHICLE DETECTION
    # -------------------------
    detections = coco_model(frame)[0]
    detections_ = []

    for det in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = det
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    detections_ = np.asarray(detections_)
    track_ids = mot_tracker.update(detections_) if len(detections_) else np.empty((0, 5))

    # -------------------------
    # LICENSE PLATE DETECTION
    # -------------------------
    license_plates = license_plate_detector(frame)[0]

    for lp in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = lp

        xcar1, ycar1, xcar2, ycar2, car_id = get_car(lp, track_ids)

        if car_id == -1:
            continue

        car_id = int(car_id)

        # -------------------------
        # OCR (OPTIONAL)
        # -------------------------
        license_plate_text = ""
        license_plate_text_score = 0

        try:
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
            text, text_score = read_license_plate(thresh)

            if text is not None:
                license_plate_text = text
                license_plate_text_score = text_score

        except:
            pass

        # -------------------------
        # ALWAYS WRITE RESULT
        # -------------------------
        results[frame_nmr][car_id] = {
            'car': {
                'bbox': [xcar1, ycar1, xcar2, ycar2]
            },
            'license_plate': {
                'bbox': [x1, y1, x2, y2],
                'text': license_plate_text,
                'bbox_score': score,
                'text_score': license_plate_text_score
            }
        }


# =========================
# SAVE RESULTS
# =========================
print("Frames with detections:", len(results))
write_csv(results, './test.csv')

cap.release()
print("✅ main.py finished successfully")