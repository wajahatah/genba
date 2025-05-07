import json
import os

# Load COCO keypoints JSON
with open('C:/wajahat/personal/genba/genba_kp/v2/annotations/person_keypoints_default.json') as f:
    coco_data = json.load(f)

images = {img['id']: img for img in coco_data['images']}
annotations = coco_data['annotations']
output_dir = "C:/wajahat/personal/genba/genba_kp/v2/yolo_labels"
os.makedirs(output_dir, exist_ok=True)

# Class ID for "person" in COCO is 0
class_id = 0

for ann in annotations:
    image_info = images[ann['image_id']]
    width = image_info['width']
    height = image_info['height']

    # COCO keypoints: [x1, y1, v1, x2, y2, v2, ..., xk, yk, vk]
    keypoints = ann['keypoints']
    num_kpts = len(keypoints) // 3

    # Normalize bbox center x, y and width, height
    x, y, w, h = ann['bbox']
    x_center = (x + w / 2) / width
    y_center = (y + h / 2) / height
    w_norm = w / width
    h_norm = h / height

    # Normalize keypoints and format them as x y v
    kpt_list = []
    for i in range(num_kpts):
        kpt_x = keypoints[i * 3] / width
        kpt_y = keypoints[i * 3 + 1] / height
        kpt_v = keypoints[i * 3 + 2]
        kpt_list.extend([f"{kpt_x:.6f}", f"{kpt_y:.6f}", str(kpt_v)])

    yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f} " + " ".join(kpt_list)

    # Write to a .txt file per image
    file_name = os.path.splitext(image_info['file_name'])[0] + ".txt"
    with open(os.path.join(output_dir, file_name), "a") as out_f:
        out_f.write(yolo_line + "\n")

print("Conversion completed. YOLO label files are in:", output_dir)
