from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load the YOLO detection and pose models
detector = YOLO("best_3.pt")        # Detection model (adjust to your model)
pose_model = YOLO("yolo11n-pose.pt")   # Pose model

names = detector.model.names  # e.g., {0: "human", 1: "paddle", 2: "pig"}

# Open a video file or a webcam stream (0 for default camera)
cap = cv2.VideoCapture("test2.mp4")  # Replace with 0 to use a webcam

person_box = []
paddle_box = []

def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes."""
    x1, y1, x2, y2 = box1  # Box 1 coordinates
    x1_p, y1_p, x2_p, y2_p = box2  # Box 2 coordinates

    # Calculate intersection coordinates
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)

    # Compute intersection area
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height

    # Compute areas
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)

    # Compute union
    union = area1 + area2 - intersection

    # Compute IoU
    iou = intersection / union if union > 0 else 0
    return iou

def boxes_are_close(box1, box2, distance_threshold=100):
    """Check if the centers of two boxes are within a given pixel distance."""
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    distance = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    return distance <= distance_threshold

def check_paddle_in_hand(person_boxes, paddle_boxes, iou_threshold=0.01,close_threshold=100):
    """Check if a paddle is in hand based on IoU threshold."""
    alert = False
    iou = 0
    for person_box in person_boxes:
        for paddle_box in paddle_boxes:
            if boxes_are_close(person_box, paddle_box, distance_threshold=close_threshold):
                iou = compute_iou(person_box, paddle_box)
                if iou >= iou_threshold:
                    print(f"Alert: Paddle in hand! IoU = {iou:.2f}")
                    alert = True
                else:
                    print(f"IoU = {iou:.2f}, No alert.")
                    
    return alert,iou

# with open("genba_keypoints.txt", "w") as f:
    # f.write("frame, person, keypoints\n")
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

        # frame_for_detection = frame.copy()

        # Run the detector on the pristine copy
    person_boxes = []
    paddle_boxes = []

    results = detector(frame)
    
    # Process each detection from the detector
    for det in results[0].boxes:
        # Extract bounding box coordinates and predicted label
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        label = names[int(det.cls[0])].lower()
        
        # # Draw bounding box and label on the original frame (for all classes)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putText(frame, label, (x1, max(y1 - 10, 0)), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # If the detected object is "person", run pose estimation on that region
        # if label.lower() == "person":
        if label == "person":
            # Crop the region from the pristine copy
            # crop_frame = frame[y1:y2, x1:x2]
            # if crop_frame.size == 0:
            #     continue
            # cv2.putText(frame, f"({int(x1)},{int(y1)})", (x1, y1),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # cv2.putText(frame, f"({int(x2)},{int(y1)})", (x2, y1),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # cv2.circle(frame, (int(x1), int(y1)), 2, (0, 255, 0), -1)
            # cv2.circle(frame, (int(x1), int(y2)), 2, (0, 255, 0), -1)
            # cv2.circle(frame, (int(x2), int(y1)), 2, (0, 255, 0), -1)
            # cv2.circle(frame, (int(x2), int(y2)), 2, (0, 255, 0), -1)
            # person_box = [x1, y1, x2, y2]
            person_boxes.append([x1, y1, x2, y2])
            hmx = int((x1+x2)/2)
            cv2.circle(frame, (int(hmx), (int(y1) + 10)), 5, (150, 110, 135), -1)
            print(f"Person box: {person_boxes}")

        # if label.lower() == "paddle":
        if label == "paddle":
            # cv2.putText(frame, f"({int(x1)},{int(y1)})", (x2, y1),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # cv2.putText(frame, f"({int(x2)},{int(y2)})", (x2, y2),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # cv2.circle(frame, (int(x1), int(y1)), 2, (0, 255, 0), -1)
            # cv2.circle(frame, (int(x1), int(y2)), 2, (0, 255, 0), -1)
            # cv2.circle(frame, (int(x2), int(y1)), 2, (0, 255, 0), -1)
            # cv2.circle(frame, (int(x2), int(y2)), 2, (0, 255, 0), -1)
            # paddle_box = [x1, y1, x2, y2]
            paddle_boxes.append([x1, y1, x2, y2])
            rmy = int((y1+y2)/2)
            cv2.circle(frame, ((int(x2) - 10), int(rmy)), 5, (205, 220, 210), -1)
            print(f"Paddle box: {paddle_boxes}")

    # Check if a paddle is in hand
    # if person_box is not None and paddle_box is not None:
    if person_box and paddle_box:
        alert,iou = check_paddle_in_hand([person_box], [paddle_box], iou_threshold=0.3, close_threshold=100)
        # iou = compute_iou(person_box, paddle_box)
        cv2.putText(frame, str(iou), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if alert:
            cv2.putText(frame, "ALERT: Paddle in hand!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if rmy > hmx:
                cv2.putText(frame, "ALERT: Paddle raised", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                # out.write(frame)
    # Display the processed frame
    cv2.imshow("Video Inference", frame)
    
    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
