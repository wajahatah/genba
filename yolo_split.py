import os
import random
import shutil

def split_dataset(input_folder, output_folder, train_ratio=0.8, seed=42):
    # Define source folders for images and labels inside the "zip" folder.
    src_images_folder = os.path.join(input_folder, 'images')
    src_labels_folder = os.path.join(input_folder, 'labels')
    
    # Define destination directories for training and validation sets.
    train_images_dir = os.path.join(output_folder, 'train', 'images')
    train_labels_dir = os.path.join(output_folder, 'train', 'labels')
    val_images_dir   = os.path.join(output_folder, 'val', 'images')
    val_labels_dir   = os.path.join(output_folder, 'val', 'labels')
    
    # Create destination directories if they don't exist.
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # List and sort all images (adjust extensions if needed)
    images = [f for f in os.listdir(src_images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()
    
    # Shuffle images for random splitting
    random.seed(seed)
    random.shuffle(images)
    
    total = len(images)
    train_count = int(total * train_ratio)
    
    train_images = images[:train_count]
    val_images = images[train_count:]
    
    # Process training images and labels
    for image in train_images:
        # Move the image file
        src_image_path = os.path.join(src_images_folder, image)
        dst_image_path = os.path.join(train_images_dir, image)
        shutil.move(src_image_path, dst_image_path)
        
        # Determine the corresponding label file (assumes .txt extension)
        label_file = os.path.splitext(image)[0] + '.txt'
        src_label_path = os.path.join(src_labels_folder, label_file)
        if os.path.exists(src_label_path):
            dst_label_path = os.path.join(train_labels_dir, label_file)
            shutil.move(src_label_path, dst_label_path)
        else:
            print(f"Warning: Label file '{label_file}' not found for image '{image}'")
    
    # Process validation images and labels
    for image in val_images:
        src_image_path = os.path.join(src_images_folder, image)
        dst_image_path = os.path.join(val_images_dir, image)
        shutil.move(src_image_path, dst_image_path)
        
        label_file = os.path.splitext(image)[0] + '.txt'
        src_label_path = os.path.join(src_labels_folder, label_file)
        if os.path.exists(src_label_path):
            dst_label_path = os.path.join(val_labels_dir, label_file)
            shutil.move(src_label_path, dst_label_path)
        else:
            print(f"Warning: Label file '{label_file}' not found for image '{image}'")
    
    print(f"Total images processed: {total}")
    print(f"Train images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")

if __name__ == '__main__':
    # Modify these paths to point to your directories.
    input_folder = "C:/wajahat/personal/genba/genba_kp/train_data"#"C:/wajahat/Qiyas_Gaze_Estimation/miscellaneous_yolo_pose/zip"         # Folder containing 'images' and 'labels'
    output_folder = "C:/wajahat/personal/genba/genba_kp/train_data/data"     # Destination folder for train/val split
    split_dataset(input_folder, output_folder)

