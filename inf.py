import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO


# detector = torch.load('best_3.pt')
detector = YOLO('yolo11n.pt')
# , map_location='cpu')
detector.eval()

# pose_model = torch.load('yolo11n-pose.pt')
pose_model = YOLO('yolo11n-pose.pt')
# , map_location='cpu')
pose_model.eval()

# -------------------------------
# 2. Preprocessing functions
# -------------------------------
def preprocess_for_detector(frame, target_size=640):
    """
    Preprocess the input image for the object detector.
    Resize, convert color, normalize, and convert to tensor.
    """
    # Resize image (assuming detector expects 640x640)
    frame_resized = cv2.resize(frame, (target_size, target_size))
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    # Convert to float and normalize to [0, 1]
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    return tensor

def preprocess_for_pose(roi, target_size=256):
    """
    Preprocess the cropped region for the pose model.
    Resize the ROI to the pose model's expected input size,
    convert to tensor, and normalize.
    """
    roi_resized = cv2.resize(roi, (target_size, target_size))
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(roi_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)
    return tensor

# -------------------------------
# 3. Inference helper functions
# -------------------------------
def run_detector(model, input_tensor, conf_threshold=0.5):
    """
    Run the object detector and filter detections for the human class.
    Here we assume:
      - The model output is a dictionary with keys: 'boxes', 'scores', 'labels'.
      - Class index 0 corresponds to 'human'.
    """
    with torch.no_grad():
        outputs = model(input_tensor)
    # Convert outputs to numpy (this depends on your model API)
    boxes = outputs['boxes'].cpu().numpy()   # shape (N, 4) [x1, y1, x2, y2]
    scores = outputs['scores'].cpu().numpy()   # shape (N,)
    labels = outputs['labels'].cpu().numpy()   # shape (N,)
    
    detections = []
    for bbox, score, label in zip(boxes, scores, labels):
        if label == 0 and score >= conf_threshold:  # Filter for human detections
            detections.append({'bbox': bbox, 'score': score, 'class': 'human'})
    return detections

def run_pose_model(model, input_tensor):
    """
    Run the pose model on the input tensor.
    Assume the output is a dictionary with a key 'keypoints' that returns
    an array of shape (1, num_keypoints, 2) with coordinates in the resized ROI.
    """
    with torch.no_grad():
        outputs = model(input_tensor)
    keypoints = outputs['keypoints'].cpu().numpy()[0]
    return keypoints

# -------------------------------
# 4. Integration Pipeline Function
# -------------------------------
def process_frame(frame):
    """
    Process one video frame:
      1. Run detector and filter human detections.
      2. For each human detection, crop ROI and run the pose model.
      3. Map keypoints to original frame coordinates.
      4. Draw bounding boxes and keypoints.
    Returns the processed frame (as a PIL image).
    """
    # Copy frame for drawing (convert to PIL for drawing)
    vis_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(vis_image)
    
    # Preprocess frame for object detection
    detector_input = preprocess_for_detector(frame, target_size=640)
    detections = run_detector(detector, detector_input, conf_threshold=0.5)
    
    # Process each detected human box
    for det in detections:
        bbox = det['bbox']
        # If detector was run on resized image, scale bbox to original image size.
        # Here we assume the detector outputs coordinates relative to original dimensions.
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Crop ROI from the original frame
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        
        # Preprocess ROI for pose estimation
        pose_input = preprocess_for_pose(roi, target_size=256)
        keypoints = run_pose_model(pose_model, pose_input)
        
        # Map keypoints from 256x256 scale to ROI size
        roi_h, roi_w = roi.shape[:2]
        scale_x = roi_w / 256.0
        scale_y = roi_h / 256.0
        keypoints_scaled = keypoints.copy()
        keypoints_scaled[:, 0] *= scale_x
        keypoints_scaled[:, 1] *= scale_y
        
        # Map keypoints back to original frame coordinates
        keypoints_original = keypoints_scaled + np.array([x1, y1])
        
        # Draw bounding box for the detected human
        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        
        # Draw keypoints on the frame
        for kp in keypoints_original:
            kp_x, kp_y = kp
            r = 3  # keypoint radius
            draw.ellipse((kp_x - r, kp_y - r, kp_x + r, kp_y + r), fill="blue")
    
    return vis_image

# def process_video(input_video_path, output_video_path, display=False):
def process_video(input_video_path):
    # Open video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file.")
    
    # Get video properties
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # # Define video writer to save output video
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video
        
        # Process the frame: detect humans and keypoints
        processed_frame = process_frame(frame)
        # Convert processed frame (PIL image) back to OpenCV image (BGR) for writing
        processed_frame_cv = cv2.cvtColor(np.array(processed_frame), cv2.COLOR_RGB2BGR)
        
        cv2.imshow('frame', processed_frame_cv)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Write the frame to the output video
        # out_writer.write(processed_frame_cv)
        
        # (Optional) display the frame in a window
        # if display:
        #     cv2.imshow("Processed Frame", processed_frame_cv)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        
        # frame_count += 1
        # print(f"Processed frame {frame_count}", end='\r')
    
    cap.release()
    cv2.destroyAllWindows
    # out_writer.release()
    # if display:
    #     cv2.destroyAllWindows()
    # print("\nVideo processing complete.")

# -------------------------------
# 6. Run the video inference
# -------------------------------
if __name__ == '__main__':
    input_video = "F:/Genba/video_20241205_071159_chunk_23.mp4"    # Path to input video file
    # output_video = "output_video.mp4"  # Path to save the output video
    process_video(input_video)
    # process_video(input_video, output_video, display=False)
