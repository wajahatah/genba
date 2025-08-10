from ultralytics import YOLO
import cv2

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == "__main__":

    model = YOLO("best_3.pt")
    model.predict("test1.mp4", show=True)