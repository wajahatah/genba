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

with open("genba_keypoints.txt", "w") as f:
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
        results = detector(frame)
        
        # Process each detection from the detector
        for det in results[0].boxes:
            # Extract bounding box coordinates and predicted label
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            label = names[int(det.cls[0])]
            
            # # Draw bounding box and label on the original frame (for all classes)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(frame, label, (x1, max(y1 - 10, 0)), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # If the detected object is "person", run pose estimation on that region
            if label.lower() == "person":
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
                hmx = (x1+x2)/2
                cv2.circle(frame, (int(hmx), (int(y1) + 10)), 5, (150, 110, 135), -1)

            if label.lower() == "paddle":
                # cv2.putText(frame, f"({int(x1)},{int(y1)})", (x2, y1),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # cv2.putText(frame, f"({int(x2)},{int(y2)})", (x2, y2),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # cv2.circle(frame, (int(x1), int(y1)), 2, (0, 255, 0), -1)
                # cv2.circle(frame, (int(x1), int(y2)), 2, (0, 255, 0), -1)
                # cv2.circle(frame, (int(x2), int(y1)), 2, (0, 255, 0), -1)
                # cv2.circle(frame, (int(x2), int(y2)), 2, (0, 255, 0), -1)
                rmy = (y1+y2)/2
                cv2.circle(frame, ((int(x2) - 20), int(rmy)), 5, (205, 220, 210), -1)
                    
                # out.write(frame)
    # Display the processed frame
        cv2.imshow("Video Inference", frame)
        
        # Press 'q' to exit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

cap.release()
cv2.destroyAllWindows()
