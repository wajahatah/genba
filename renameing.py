import os
import re
import shutil

def get_b_folders(base_dir):
    """ Get all folders in the form 'b<A>_<B>' sorted by <B> """
    folder_pattern = r"v(\d+)"  # Pattern to extract <B>
    b_folders = []

    for folder_name in os.listdir(base_dir):
        match = re.match(folder_pattern, folder_name)
        if match:
            b_value = int(match.group(1))  # Extract <B> value
            b_folders.append((folder_name, b_value))
    
    # Sort folders by <B> value
    return sorted(b_folders, key=lambda x: x[1])

def copy_files_in_folder(images_folder, labels_folder, start_c, output_images_folder, output_labels_folder):
    """ 
    Copy files from 'images' and 'labels' folders, rename them in sequence, and save in separate output folders. 
    Ensure filenames are unique by checking in the target folders.
    """
    current_c = start_c

    # Get all image and label files
    image_files = sorted([f for f in os.listdir(images_folder) if (f.endswith('.PNG') or f.endswith('.png'))])
    label_files = sorted([f for f in os.listdir(labels_folder) if f.endswith('.txt')])

    for img_file, lbl_file in zip(image_files, label_files):
        while True:
            new_c = f"{current_c:03d}"

            new_img_name = f"frame_{new_c}.png"
            new_lbl_name = f"frame_{new_c}.txt"

            new_img_path = os.path.join(output_images_folder, new_img_name)
            new_lbl_path = os.path.join(output_labels_folder, new_lbl_name)

            if os.path.exists(new_img_path) or os.path.exists(new_lbl_path):
                print(f"Skipping C={new_c} as {new_img_name} or {new_lbl_name} already exists in output folders.")
                current_c += 1  # Skip to the next number
            else:
                old_img_path = os.path.join(images_folder, img_file)
                old_lbl_path = os.path.join(labels_folder, lbl_file)

                shutil.copy2(old_img_path, new_img_path)
                shutil.copy2(old_lbl_path, new_lbl_path)

                current_c += 1
                break

    return current_c

def rename_across_folders(base_dir, output_images_folder, output_labels_folder):
    """ 
    Main function to copy and rename files across all 'b<A>_<B>' folders and save them into 
    separate output folders for images and labels.
    """
    b_folders = get_b_folders(base_dir)
    current_c = 0  # Start C from 0 (as "frame_00000000")

    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)

    for folder_name, b_value in b_folders:
        print(f"Processing folder: {folder_name} (B={b_value})")

        # Define paths to the 'train/images' and 'train/labels' folders
        images_folder = os.path.join(base_dir, folder_name, "matched_images" )
        labels_folder = os.path.join(base_dir, folder_name, "labels")

        if os.path.exists(images_folder) and os.path.exists(labels_folder):
            # Copy and rename files in this folder, starting with the current C value
            current_c = copy_files_in_folder(images_folder, labels_folder, current_c, output_images_folder, output_labels_folder)
        else:
            print(f"Skipping folder {folder_name}: 'images/train' or 'labels/train' not found.")

# Example usage
base_directory = "C:/wajahat/personal/genba/genba_kp"#"C:/Users/LAMBDA THETA/Downloads/remain"        # Path to the base directory containing the 'b<A>_<B>' folders
output_images_folder = "C:/wajahat/personal/genba/genba_kp/train_data/images"     # Path to the output folder where copied images will be saved
output_labels_folder = "C:/wajahat/personal/genba/genba_kp/train_data/labels"#"C:/OsamaEjaz/Qiyas_Gaze_Estimation/Wajahat_Yolo_keypoint/updated_dataset/train/labels"     # Path to the output folder where copied labels will be saved

# Copy and rename files from all folders, starting from "frame_00000000"
rename_across_folders(base_directory, output_images_folder, output_labels_folder)