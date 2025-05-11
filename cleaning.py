import os

# Set paths to your labels and images folders
labels_dir = r"C:/wajahat/personal/genba/genba_kp/train_data/train/labels"   # <- update this
images_dir = r"C:/wajahat/personal/genba/genba_kp/train_data/train/images"   # <- update this
# "C:/wajahat/personal/genba/genba_kp/train_data/val/labels"

def delete_empty_keypoint_files(labels_dir, images_dir):
    deleted = 0
    checked = 0

    # print(f"\nScanning folder: {labels_dir}")
    # txt_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    # print(f"Found {len(txt_files)} .txt files:")
    # for f in txt_files:
    #     print(" -", f)

    for file in os.listdir(labels_dir):
        if file.endswith(".txt"):
            label_path = os.path.join(labels_dir, file)
            checked += 1

            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    contents = f.read().strip()

                if not contents:
                    # File is truly empty: delete it and the corresponding image
                    image_name = os.path.splitext(file)[0] + ".png"
                    image_path = os.path.join(images_dir, image_name)

                    os.remove(label_path)
                    print(f"❌ Deleted empty label: {label_path}")

                    if os.path.exists(image_path):
                        os.remove(image_path)
                        print(f"❌ Deleted image: {image_path}")

                    deleted += 1
                else:
                    print(f"✅ Kept file: {label_path}")

            except Exception as e:
                print(f"⚠️ Error reading {label_path}: {e}")

    print(f"\n🔍 Checked {checked} label files.")
    print(f"🗑️ Deleted {deleted} empty label/image pairs.")

delete_empty_keypoint_files(labels_dir, images_dir)