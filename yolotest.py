from ultralytics import YOLO
import os
import cv2
import numpy as np

# Avoid potential library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def draw_lines(frame, keypoints, connections):
    for start_idx, end_idx in connections:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            x1, y1, conf1 = keypoints[start_idx]
            x2, y2, conf2 = keypoints[end_idx]

            # Ensure confidence is above a threshold to draw
            if conf1 > 0.5 and conf2 > 0.5:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)  # Yellow lines

if __name__ == "__main__":
    # Load your trained YOLOv8 model
    model = YOLO("runs/pose/3rd_itteration_6fl/weights/best_v3_6f.pt")

    # Open the video file
    # video_path = "C:/Users/LAMBDA THETA/Downloads/test_bench_02/test_bench_02/Cam_19_02.mp4"#"Cam_19_14.mp4"
    video_path = "C:/Users/LAMBDA THETA/Downloads/keypoints_wajahat/keypoints_wajahat/video_20241205_095942_chunk_8_chunk_1.mp4"
    cap = cv2.VideoCapture(video_path)

    # Get video properties for saving the output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
    out = cv2.VideoWriter('output_with_keypoints.mp4', fourcc, fps, (frame_width, frame_height))

    connections = [
        (0, 1), (0, 2), (0, 3),  # Keypoint 0 to 1, 2, and 3
        (3, 4), (3, 5),          # Keypoint 1 to 4 and 7
        (5, 6), (6, 9),           # Keypoint 4 to 5, and 5 to 6
        (7, 8), (4, 7)           # Keypoint 7 to 8, and 8 to 9
    ]

    # Open a text file to save keypoints
    with open('keypoints_output.txt', 'w') as f:

        # Check if the video capture opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        frame_count = 0  # To track frame number

    # Read and process the video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit the loop if no more frames are available
            
            # black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

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
                        f.write(f"Frame {frame_count}, Person {person_idx}:\n")
                        keypoint_list = []

                        for kp_idx, keypoint in enumerate(person_keypoints):
                            x, y, confidence = keypoint[0].item(), keypoint[1].item(), keypoint[2].item()
                            keypoint_list.append((x,y,confidence))

                            # Save the keypoints to the text file
                            f.write(f"  Keypoint {kp_idx}: (x={x:.2f}, y={y:.2f}, confidence={confidence:.2f})\n")

                            # Draw a circle at each keypoint
                            # cv2.circle(black_frame, (int(x), int(y)), 5, (0,0, 255), -1)

                            cv2.circle(frame, (int(x), int(y)), 5, (0, 255,0), -1)
                            # draw_lines(frame,keypoint_list,connections)

                            # Put the keypoint values on the frame
                            # cv2.putText(
                            #     frame,
                            #     f"({int(x)}, {int(y)}, {confidence:.2f})",
                            #     (int(x) + 5, int(y) - 5),
                            #     cv2.FONT_HERSHEY_SIMPLEX,
                            #     0.4,
                            #     (255, 0, 0),
                            #     1
                            # )

                            # if confidence > 0.5:
                            #     points.append((int(x),int(y)))

                            # else:
                            #     points.append(None)

                            # for i in range (len(points) -1):
                            #     if points[i] and points[i+1]:
                            #         cv2.line(frame, points[i], points[i+1], (0,225,0),2)

        # Write the processed frame to the output video file
            out.write(frame)
            frame_filename = os.path.join("C:/wajahat/personal/genba/op/t1", f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename,frame)
            out.write(frame)

            # Display the frame with keypoints and values
            cv2.imshow('Pose Detection', frame)
            # cv2.imshow('Pose Detection', black_frame)

            # Delay of 50 milliseconds to slow down the video playback
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

            frame_count += 1  # Increment frame counter

    # Release the video capture and close display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()