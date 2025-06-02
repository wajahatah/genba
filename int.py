from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load the YOLO detection and pose models
detector = YOLO("best_3.pt")        # Detection model (adjust to your model)
pose_model = YOLO("yolo11n-pose.pt")   # Pose model

names = detector.model.names  # e.g., {0: "human", 1: "paddle", 2: "pig"}

# Open a video file or a webcam stream (0 for default camera)
cap = cv2.VideoCapture("test1.mp4")  # Replace with 0 to use a webcam

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
            
            # Draw bounding box and label on the original frame (for all classes)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # If the detected object is "person", run pose estimation on that region
            if label.lower() == "person":
                # Crop the region from the pristine copy
                crop_frame = frame[y1:y2, x1:x2]
                if crop_frame.size == 0:
                    continue

                # Run pose estimation on the cropped image
                pose_results = pose_model(crop_frame)

                for result in pose_results:
                    keypoints = result.keypoints  # Access the keypoints object

                    # default_keypoints = [(0, 0, 0)] * 17

                    if keypoints is not None:
                        # Get the data attribute, which contains x, y, and confidence values
                        keypoints_data = keypoints.data
                        for person_idx, person_keypoints in enumerate(keypoints_data):
                            f.write(f"Frame{frame_count}, Person{person_idx}")
                            # keypoint_list = []

                            for kp_idx, keypoint in enumerate(person_keypoints):
                                cx, cy, confidence = keypoint[0].item(), keypoint[1].item(), keypoint[2].item()
                                # keypoint_list.append((x,y,confidence))
                                x,y = int(cx + x1), int(cy + y1)
                                # default_keypoints[kp_idx] = (x, y, confidence)

                                # for kp_idx, (x, y, confidence) in enumerate(default_keypoints):
                                f.write(f"  Keypoint {kp_idx}: (x={x:.2f}, y={y:.2f}, confidence={confidence:.2f})\n")

                                cv2.circle(frame, (int(x), int(y)), 5, (0, 255,0), -1)
                                # cv2.putText(frame, f"({int(x)}, {int(y)})", (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                frame_filename = os.path.join("C:/wajahat/personal/genba/frames", f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_filename,frame)
                # out.write(frame)
    # Display the processed frame
        cv2.imshow("Video Inference", frame)
        
        # Press 'q' to exit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

cap.release()
cv2.destroyAllWindows()
