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
    person_box = []
    paddle_box = []

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
            person_box = [x1, y1, x2, y2]
            # person_boxes.append([x1, y1, x2, y2])
            hmx = int((x1+x2)/2)
            cv2.circle(frame, (int(hmx), (int(y1) + 10)), 5, (150, 110, 135), -1)
            print(f"Person box: {person_box}")

        # if label.lower() == "paddle":
        if label == "paddle":
            paddle_box = [x1, y1, x2, y2]
    # Crop the paddle region from the frame
            paddle_region = frame[y1:y2, x1:x2]
            
            # Convert the cropped region to HSV color space
            hsv_region = cv2.cvtColor(paddle_region, cv2.COLOR_BGR2HSV)
            
            # Define two ranges for the red color in HSV
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([179, 255, 255])
            
            # Create masks for red color (accounting for the hue wrap-around)
            mask1 = cv2.inRange(hsv_region, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv_region, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            # Find contours of the red regions within the paddle box
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Choose the largest red contour
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    # Compute centroid relative to the cropped region
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # Adjust the centroid coordinates relative to the full frame
                    cX += x1
                    cY += y1
                else:
                    # Fallback to the center of the paddle_box if contour area is too small
                    cX = int((x1 + x2) / 2)
                    cY = int((y1 + y2) / 2)
                print(f"Paddle red keypoint: {(cX, cY)}")
            else:
                # Fallback: if no red pixels detected, use the center of the paddle_box
                cX = int((x1 + x2) / 2)
                cY = int((y1 + y2) / 2)
                print("No red pixels detected in paddle; using bounding box center.")

            rmy =(cX, cY)

            # Draw the keypoint on the frame at the detected centroid
            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)  # Change the color here if needed`

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
