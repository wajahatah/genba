from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == "__main__":
    model = YOLO("yolo11l-pose.pt")
    # print(model)
    model.train(data="C:/wajahat/personal/genba/genba_data.yaml", epochs=300, imgsz=640, batch=0.4, device=0, patience=5, freeze=6, name="3rd_itteration_6fl")

