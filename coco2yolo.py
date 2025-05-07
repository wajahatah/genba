import os
import zipfile
import json

# === CONFIGURATION ===
input_dir = "C:/wajahat/personal/genba/genba_kp/zip"  # Folder containing ZIP files
output_root = "C:/wajahat/personal/genba/genba_kp"  # Where ZIPs will be extracted
# "C:\wajahat\personal\genba\genba_kp\zip"

# === MAIN PROCESSING LOOP ===
for zip_file in os.listdir(input_dir):
    if not zip_file.endswith(".zip"):
        continue

    zip_path = os.path.join(input_dir, zip_file)
    unzip_folder_name = os.path.splitext(zip_file)[0]
    unzip_folder_path = os.path.join(output_root, unzip_folder_name)

    # 1. UNZIP
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_folder_path)
    print(f"Extracted: {zip_file}")

    # 2. FIND ANNOTATION FILE
    annotations_folder = os.path.join(unzip_folder_path, "annotations")
    if not os.path.exists(annotations_folder):
        print(f"No annotations folder in {zip_file}, skipping.")
        continue

    json_files = [f for f in os.listdir(annotations_folder) if f.endswith('.json')]
    if not json_files:
        print(f"No JSON annotation found in {annotations_folder}, skipping.")
        continue

    annotation_path = os.path.join(annotations_folder, json_files[0])
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    images = {img['id']: img for img in coco_data['images']}
    annotations = coco_data['annotations']

    # 3. SETUP LABEL OUTPUT FOLDER
    labels_dir = os.path.join(unzip_folder_path, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    # 4. CONVERT COCO TO YOLO
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

        yolo_line = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f} " + " ".join(kpt_list)

        # Save per image file
        image_filename = os.path.splitext(image_info['file_name'])[0] + ".txt"
        label_path = os.path.join(labels_dir, image_filename)
        with open(label_path, "a") as out_f:
            out_f.write(yolo_line + "\n")

    print(f"Conversion complete for: {zip_file}")

print("All ZIPs processed.")
