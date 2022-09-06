import torch
import numpy as np
import time
import sys
import os
from utilities import Utils


class BaseDetector():
    def __init__(self, model='default', thresh_bbox=0.1, device='cpu', verbose = 0):
        
        self.model.classes=0 #running only person detection
        self.thresh_bbox=thresh_bbox
        self.detection=np.array([0, 0, 0, 0]) 
        self.verbose=verbose
        if self.verbose:
            print(f"Created YOLOv5 ({model}) detector with verbose={verbose}.")

    def load_model(self, model):
        if model=='small' or model=='s':
            yolo_version="yolov5s"
        if model=='default' or model=='medium' or model=='m':
            yolo_version="yolov5m"
        if model=='large' or model=='l':
            yolo_version="yolov5l"
        if model=='xlarge' or model=='xl':
            yolo_version="yolov5xl"
        os.chdir("detectors/weights")
        self.model = torch.hub.load('ultralytics/yolov5', yolo_version)
        os.chdir("../..")
        print(f"-> Using {yolo_version} for multi bbox detection.")
        



class Yolov7Detector():
    def __init__(self, model='default', thresh_bbox=0.1, device='cpu', verbose = 0):
        if model=='small' or model=='s':
            yolo_version="yolov7-tiny"
        if model=='default' or model=='medium' or model=='m':
            yolo_version="yolov7"
        if model=='large' or model=='l':
            print("No weights for yolov7 M loading same weights as medium")
            yolo_version="yolov7"
        if model=='xlarge' or model=='xl':
            yolo_version="yolov7x"
        self.weights = "detectors/weights/"+yolo_version+".pt"
        self.classes=0
        self.thresh_bbox=thresh_bbox
        self.detection=np.array([0, 0, 0, 0])
        self.verbose=verbose
        if self.verbose:
            print(f"Created YOLOv7 ({model}) detector with verbose={verbose}.")

        # Load model
        #self.device=1 #'cpu'
        # Initialize
        set_logging()
        if isinstance(device, int):
            device='cuda:'+str(device)
            self.device = torch.device(device)
        else:
            self.device = select_device()
        #print(f"device selected: {torch.device}, device is {self.device}")
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        with torch.no_grad():
            self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
            self.stride = int(self.model.stride.max())  # model stride
            self.imgsz= 640
            self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size
            self.model = TracedModel(self.model, self.device, self.imgsz)
            if self.half:
                self.model.half()  # to FP16
            self.init=True
