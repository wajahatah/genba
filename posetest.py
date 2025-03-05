from ultralytics import YOLO
import os
import cv2
import numpy as np

# Avoid potential library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == "__main__":
    
    model = YOLO("yolo11n-pose.pt")

    video_path = "F:/Genba/video_20241205_071159_chunk_23.mp4"
    cap = cv2.VideoCapture(video_path)

    with open('keypoints_output.txt', 'w') as f:

        # Check if the video capture opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

    # Read and process the video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit the loop if no more frames are available

            # Run inference on the current frame
            frame = cv2.resize(frame,(1280,720))
            results = model(frame)

            # Iterate over each detected object and print their keypoints
            for result in results:
                keypoints = result.keypoints  # Access the keypoints object

                if keypoints is not None:
                    # Get the data attribute, which contains x, y, and confidence values
                    keypoints_data = keypoints.data
                    for person_idx, person_keypoints in enumerate(keypoints_data):
                        keypoint_list = []

                        for kp_idx, keypoint in enumerate(person_keypoints):
                            x, y, confidence = keypoint[0].item(), keypoint[1].item(), keypoint[2].item()
                            keypoint_list.append((x,y,confidence))

                            cv2.circle(frame, (int(x), int(y)), 5, (0, 255,0), -1)

            # Display the frame with keypoints and values
            cv2.imshow('Pose Detection', frame)

            # Delay of 50 milliseconds to slow down the video playback
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

    # Release the video capture and close display window
    cap.release()
    cv2.destroyAllWindows()