import os
import shutil

# Define your folder paths
v = 14
images_folder = f"C:/wajahat/personal/genba/genba_kp/v{v}/images"
labels_folder = f"C:/wajahat/personal/genba/genba_kp/v{v}/labels"
output_folder = f"C:/wajahat/personal/genba/genba_kp/v{v}/matched_images"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get all label names (without .txt extension)
label_names = {os.path.splitext(f)[0] for f in os.listdir(labels_folder) if f.endswith('.txt')}

# Supported image extensions
image_extensions = ['.PNG', '.jpeg', '.png']

# Go through images folder
for img_file in os.listdir(images_folder):
    img_name, ext = os.path.splitext(img_file)
    if ext.lower() in image_extensions and img_name in label_names:
        src_path = os.path.join(images_folder, img_file)
        dst_path = os.path.join(output_folder, img_file)
        
        # Copy or move the image
        # shutil.copy(src_path, dst_path)  # Use this to copy
        shutil.move(src_path, dst_path)    # Use this to move
        print(f"Moved: {img_file}")

print("Done.")
