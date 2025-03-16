from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLO detection and pose models
detector = YOLO("best_3.pt")        # Detection model (adjust to your model)
pose_model = YOLO("yolo11n-pose.pt")   # Pose model

names = detector.model.names  # e.g., {0: "human", 1: "paddle", 2: "pig"}

# Open a video file or a webcam stream (0 for default camera)
cap = cv2.VideoCapture("test1.mp4")  # Replace with 0 to use a webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_for_detection = frame.copy()

    # Run the detector on the pristine copy
    results = detector(frame_for_detection)
    
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
            crop_frame = frame_for_detection[y1:y2, x1:x2]
            if crop_frame.size == 0:
                continue

            # Run pose estimation on the cropped image
            pose_results = pose_model(crop_frame)

            for result in pose_results:
                keypoints = result.keypoints  # Access the keypoints object

                if keypoints is not None:
                    # Get the data attribute, which contains x, y, and confidence values
                    keypoints_data = keypoints.data
                    for person_idx, person_keypoints in enumerate(keypoints_data):
                        # keypoint_list = []

                        for kp_idx, keypoint in enumerate(person_keypoints):
                            cx, cy, confidence = keypoint[0].item(), keypoint[1].item(), keypoint[2].item()
                            # keypoint_list.append((x,y,confidence))
                            x,y = int(cx + x1), int(cy + y1)

                            cv2.circle(crop_frame, (int(x), int(y)), 5, (0, 255,0), -1)

            # cv2.imshow("Cropped Pose", crop_frame)
            # For each pose detection, extract keypoints and map them to original frame coordinates
            # for pose in pose_results:
            #         keypoints = pose.keypoints.xy
            #     #     if hasattr(keypoints, "cpu"):
            #     #         keypoints = keypoints.cpu().numpy()
            #     #     else:
            #     #         keypoints = keypoints.numpy() if hasattr(keypoints, "numpy") else np.array(keypoints)
            #     # # Expecting keypoints shape: (num_keypoints, 2)
            #         if hasattr(keypoints, "cpu"):
            #             keypoints = keypoints.cpu().numpy()
            #         else:
            #             keypoints = np.array(keypoints)
            #         # If there are no keypoints detected, skip
            #         if keypoints.size == 0 or keypoints.shape[0] == 0:
            #             continue

            #         for kp in keypoints:
            #             kp = np.array(kp).flatten()
            #         # Check that we have at least two values (x, y)
            #             if kp.size < 5:
            #                 continue
            #             kp_x = int(kp[0]) + x1  # Map back to original frame coordinates
            #             kp_y = int(kp[1]) + y1
                        # kp_x = int(kp[0].item()) + x1
                        # kp_y = int(kp[1].item()) + y1

                # for keypoint in pose.keypoints.xy:
                #     pts = keypoint.tolist()
                #     # Skip if pts is empty
                #     if not pts:
                #         continue
                #     # If the list is nested (e.g., [[x, y]]), flatten it
                #     if isinstance(pts[0], list):
                #         pts = pts[0]
                #     # Ensure there are at least two elements (x and y)
                #     if len(pts) < 2:
                #         continue
                #     # Map keypoint from crop coordinates back to original frame coordinates
                #     kp = (int(pts[0]) + x1, int(pts[1]) + y1)
                    # cv2.circle(frame, kp, 3, (0, 0, 255), -1)
                    # cv2.circle(frame, (kp_x, kp_y), 3, (0, 0, 255), -1)

    # Run the YOLO detector on the frame
    # results = detector(frame)

    # for det in results[0].boxes:
    #     # Get bounding box coordinates and predicted label
    #     x1, y1, x2, y2 = map(int, det.xyxy[0])
    #     label = names[int(det.cls[0])]
        
    #     # Draw the bounding box and label on the original frame
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.putText(frame, label, (x1, max(y1 - 10, 0)), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    #     # If the detected object is "person", run pose estimation
    #     if label.lower() == "person":
    #         # Crop the region of interest
    #         crop_frame = frame[y1:y2, x1:x2]
    #         if crop_frame.size == 0:
    #             continue

    #         # Run the pose estimation model on the cropped image
    #         pose_results = pose_model(crop_frame)
            
    #         # For each pose detection, get the keypoints and map them back to the original frame
    #         for pose in pose_results:
    #             for keypoint in pose.keypoints.xy:
    #                 pts = keypoint.tolist()
    #                 # If the list is nested, flatten it
    #                 if isinstance(pts[0], list):
    #                     pts = pts[0]
    #                 # Map the keypoint from the crop to the original frame coordinates by adding the offset
    #                 kp = (int(pts[0]) + x1, int(pts[1]) + y1)
    #                 cv2.circle(frame, kp, 3, (0, 0, 255), -1)
    # print("results: ", results)
    
    # Filter detections to keep only persons (assuming 'person' is class 0)
    # person_detections = [det for det in results[0].boxes if int(det.cls[0]) == 0]
    # person_detections = []
    # for det in results[0].boxes:
    #     # Get predicted label from model names
    #     label = names[int(det.cls[0])]
    #     if label.lower() == "human":
    #         person_detections.append(det)


    # # Process each detected person
    # for det in person_detections:
    # # for det in human_detections:
    #     # Extract bounding box coordinates
    #     x1, y1, x2, y2 = map(int, det.xyxy[0])
    #     # Draw the detection bounding box on the original frame
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    #     # Crop the detected person region
    #     crop_frame = frame[y1:y2, x1:x2]
    #     if crop_frame.size == 0:
    #         continue  # Skip if the crop is empty

    #     # Run pose estimation on the cropped region
    #     pose_results = pose_model(crop_frame)

    #     # Draw keypoints on the cropped region for visualization
    #     for pose in pose_results:
    #         for keypoint in pose.keypoints.xy:
    #             # kp = tuple(int(x) for x in keypoint.tolist())
    #             pts = keypoint.tolist()
    #             # If the list is nested, flatten it
    #             if isinstance(pts[0], list):
    #                 pts = pts[0]
    #             # Now convert each coordinate to an integer
    #             kp = tuple(int(x) for x in pts)
    #             cv2.circle(frame, kp, 3, (0, 0, 255), -1)
                
        # Optionally, overlay the cropped result back onto the frame
        # Here we simply show the bounding box and keypoints on the crop separately

    # Display the processed frame
    cv2.imshow("Video Inference", frame)
    
    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
