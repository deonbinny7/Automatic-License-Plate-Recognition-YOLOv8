import ast
import cv2
import numpy as np
import pandas as pd


def draw_border(img, top_left, bottom_right,
                color=(0, 255, 0), thickness=10,
                line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


# =========================
# LOAD INTERPOLATED CSV
# =========================
results = pd.read_csv('./test_interpolated.csv')

# =========================
# LOAD YOUR VIDEO (FIXED)
# =========================
video_path = 'C:/Users/deonb/Videos/cars.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"❌ Cannot open video: {video_path}")

# =========================
# VIDEO WRITER (SAFE)
# =========================
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
    fps = 25

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('out.mp4', fourcc, fps, (width, height))

if not out.isOpened():
    raise RuntimeError("❌ VideoWriter failed")

print(f"✅ Input video: {video_path}")
print(f"✅ Output video: out.mp4 ({width}x{height} @ {fps} FPS)")


# =========================
# CACHE BEST LICENSE PLATES
# =========================
license_plate = {}

for car_id in np.unique(results['car_id']):
    best_row = results[results['car_id'] == car_id] \
        .sort_values(by='license_number_score', ascending=False) \
        .iloc[0]

    license_plate[car_id] = {
        'license_crop': None,
        'license_plate_number': best_row['license_number']
    }

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(best_row['frame_nmr']))
    ret, frame = cap.read()
    if not ret:
        continue

    x1, y1, x2, y2 = ast.literal_eval(
        best_row['license_plate_bbox']
        .replace('[ ', '[')
        .replace('   ', ' ')
        .replace('  ', ' ')
        .replace(' ', ',')
    )

    crop = frame[int(y1):int(y2), int(x1):int(x2)]
    if crop.size == 0:
        continue

    crop = cv2.resize(
        crop,
        (int((x2 - x1) * 400 / (y2 - y1)), 400)
    )

    license_plate[car_id]['license_crop'] = crop


# =========================
# DRAW & WRITE FRAMES
# =========================
frame_nmr = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_nmr += 1
    frame_rows = results[results['frame_nmr'] == frame_nmr]

    for _, row in frame_rows.iterrows():
        car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(
            row['car_bbox']
            .replace('[ ', '[')
            .replace('   ', ' ')
            .replace('  ', ' ')
            .replace(' ', ',')
        )

        draw_border(
            frame,
            (int(car_x1), int(car_y1)),
            (int(car_x2), int(car_y2)),
            (0, 255, 0),
            25
        )

        lp_x1, lp_y1, lp_x2, lp_y2 = ast.literal_eval(
            row['license_plate_bbox']
            .replace('[ ', '[')
            .replace('   ', ' ')
            .replace('  ', ' ')
            .replace(' ', ',')
        )

        cv2.rectangle(
            frame,
            (int(lp_x1), int(lp_y1)),
            (int(lp_x2), int(lp_y2)),
            (0, 0, 255),
            12
        )

        car_id = row['car_id']
        if car_id not in license_plate:
            continue

        crop = license_plate[car_id]['license_crop']
        if crop is None:
            continue

        H, W, _ = crop.shape

        try:
            frame[int(car_y1) - H - 100:int(car_y1) - 100,
                  int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2)] = crop

            frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                  int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2)] = (255, 255, 255)

            text = license_plate[car_id]['license_plate_number']
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17)

            cv2.putText(
                frame,
                text,
                (int((car_x2 + car_x1 - tw) / 2),
                 int(car_y1 - H - 250 + th / 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                4.3,
                (0, 0, 0),
                17
            )
        except:
            pass

    out.write(frame)

print("✅ Video generation complete")

out.release()
cap.release()
