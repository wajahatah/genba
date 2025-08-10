from ultralytics import YOLO
import os
import cv2
import numpy as np

# Avoid potential library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    model = YOLO("runs/pose/3rd_itteration_6fl/weights/best_v3_6f.pt")

    video_path = "C:/Users/LAMBDA THETA/Downloads/keypoints_wajahat/keypoints_wajahat/video_20241205_071159_chunk_19_chunk_6.mp4" 
    # "video_20241205_095942_chunk_8_chunk_1.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Read and process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no more frames are available

        frame = cv2.resize(frame,(1280,720))
        results = model(frame)

        for result in results:
            keypoints = result.keypoints  # Access the keypoints object
            if keypoints is not None and keypoints.data.shape[1] > 0:
                keypoints_data = keypoints.data
                for person_idx, person_keypoints in enumerate(keypoints_data):
                    keypoint_list = []

                    for kp_idx, keypoint in enumerate(person_keypoints):
                        x, y, confidence = keypoint[0].item(), keypoint[1].item(), keypoint[2].item()
                        # keypoint_list.append((x,y,confidence))
                    
                    # if confidence > 0.5:
                    hx, hy = person_keypoints[0,0], person_keypoints[0,1]
                    px, py = person_keypoints[-1,0], person_keypoints[-1,1]

                        # cv2.circle(frame, (int(x), int(y)), 5, (0, 255,0), -1)

                        # print("list:", keypoint_list)

                    # head = keypoint_list[0]
                    # paddle = keypoint_list[-1]
                    # hx,hy = keypoint[0], head[1]
                    # px,py = paddle[0], paddle[1]

                    cv2.circle(frame, (int(hx), int(hy)), 5, (0, 255,0), -1)
                    cv2.circle(frame, (int(px), int(py)), 5, (0, 255,0), -1)

                    if py < hy:
                        cv2.putText(frame, "Paddle is above the head", (int(px), int(py)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


            cv2.imshow('Pose Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()