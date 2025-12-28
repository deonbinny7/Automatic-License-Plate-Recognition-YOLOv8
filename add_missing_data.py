import csv
import numpy as np
from scipy.interpolate import interp1d


# =========================
# HELPER: SAFE BBOX PARSER
# =========================
def parse_bbox(bbox_str):
    """
    Parses bbox strings like:
    '[123 45 678 910]' OR '[123, 45, 678, 910]'
    """
    bbox_str = bbox_str.replace('[', '').replace(']', '').replace(',', ' ')
    return list(map(float, bbox_str.split()))


# =========================
# INTERPOLATION FUNCTION
# =========================
def interpolate_bounding_boxes(data):

    if len(data) == 0:
        raise RuntimeError("❌ test.csv is empty. Run main.py first and ensure detections are written.")

    # Extract columns safely
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])

    car_bboxes = np.array([parse_bbox(row['car_bbox']) for row in data])
    license_plate_bboxes = np.array([parse_bbox(row['license_plate_bbox']) for row in data])

    interpolated_data = []

    unique_car_ids = np.unique(car_ids)

    for car_id in unique_car_ids:

        # Filter rows for this car
        car_mask = car_ids == car_id

        car_frame_numbers = frame_numbers[car_mask]
        car_frame_numbers = np.sort(car_frame_numbers)   # 🔴 CRITICAL FIX

        car_bboxes_filtered = car_bboxes[car_mask]
        license_bboxes_filtered = license_plate_bboxes[car_mask]

        first_frame = car_frame_numbers[0]
        last_frame = car_frame_numbers[-1]

        car_bboxes_interpolated = []
        license_bboxes_interpolated = []

        for i in range(len(car_frame_numbers)):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes_filtered[i]
            license_bbox = license_bboxes_filtered[i]

            if i > 0:
                prev_frame = car_frame_numbers[i - 1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_bbox = license_bboxes_interpolated[-1]

                gap = frame_number - prev_frame

                if gap > 1:
                    x = np.array([prev_frame, frame_number])
                    x_new = np.linspace(prev_frame, frame_number, gap, endpoint=False)

                    car_interp = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0)
                    lic_interp = interp1d(x, np.vstack((prev_license_bbox, license_bbox)), axis=0)

                    car_bboxes_interpolated.extend(car_interp(x_new)[1:])
                    license_bboxes_interpolated.extend(lic_interp(x_new)[1:])

            car_bboxes_interpolated.append(car_bbox)
            license_bboxes_interpolated.append(license_bbox)

        # =========================
        # WRITE INTERPOLATED ROWS
        # =========================
        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame + i

            row = {
                'frame_nmr': str(frame_number),
                'car_id': str(car_id),
                'car_bbox': ' '.join(map(str, car_bboxes_interpolated[i])),
                'license_plate_bbox': ' '.join(map(str, license_bboxes_interpolated[i]))
            }

            # Check if original frame exists
            original_rows = [
                r for r in data
                if int(r['frame_nmr']) == frame_number and int(float(r['car_id'])) == car_id
            ]

            if len(original_rows) == 0:
                # Interpolated frame
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = ''
                row['license_number_score'] = '0'
            else:
                # Original frame
                original = original_rows[0]
                row['license_plate_bbox_score'] = original.get('license_plate_bbox_score', '0')
                row['license_number'] = original.get('license_number', '')
                row['license_number_score'] = original.get('license_number_score', '0')

            interpolated_data.append(row)

    return interpolated_data


# =========================
# LOAD INPUT CSV
# =========================
with open('test.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

# =========================
# RUN INTERPOLATION
# =========================
interpolated_data = interpolate_bounding_boxes(data)

# =========================
# WRITE OUTPUT CSV
# =========================
header = [
    'frame_nmr',
    'car_id',
    'car_bbox',
    'license_plate_bbox',
    'license_plate_bbox_score',
    'license_number',
    'license_number_score'
]

with open('test_interpolated.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)

print("✅ test_interpolated.csv generated successfully")
