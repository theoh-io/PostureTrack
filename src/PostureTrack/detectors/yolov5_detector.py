import torch
import numpy as np
import time
import sys
import os
from utilities import Utils


class Yolov5Detector():
    def __init__(self, model='default', thresh_bbox=0.1, verbose = 0):
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
        self.model.classes=0 #running only person detection
        self.thresh_bbox=thresh_bbox
        self.detection=np.array([0, 0, 0, 0]) 
        self.verbose=verbose
        if self.verbose:
            print(f"Created YOLOv5 ({model}) detector with verbose={verbose}.")

    def bbox_format(self):
        #detection format xmin, ymin, xmax,ymax, conf, class, 'person'
        #bbox format xcenter, ycenter, width, height
        if(self.detection.shape[0]==1):
          self.detection=np.squeeze(self.detection)
          xmin=self.detection[0]
          ymin=self.detection[1]
          xmax=self.detection[2]
          ymax=self.detection[3]
          x_center=(xmin+xmax)/2
          y_center=(ymin+ymax)/2
          width=xmax-xmin
          height=ymax-ymin
          bbox=[x_center, y_center, width, height]
          bbox=np.expand_dims(bbox, axis=0)
          return bbox
        else:
          bbox_list=[]
          for i in range(self.detection.shape[0]):
            xmin=self.detection[i][0]
            ymin=self.detection[i][1]
            xmax=self.detection[i][2]
            ymax=self.detection[i][3]
            x_center=(xmin+xmax)/2
            y_center=(ymin+ymax)/2
            width=xmax-xmin
            height=ymax-ymin
            bbox_unit=np.array([x_center, y_center, width, height])
            bbox_list.append(bbox_unit)
          bbox_list=np.vstack(bbox_list)
          #bbox_list=bbox_list.tolist()
          return bbox_list

            

    # def best_detection(self):
    #     #tensor dim is now (number of detections, 7)
    #     #output dim is (1,7)
    #     N=self.detection.shape[0]
    #     if(N != 1):
    #         if self.verbose is True: print("multiple persons detected")
    #         #extracting the detection with max confidence
    #         idx=np.argmax(self.detection[range(N),4])
    #         self.detection=np.expand_dims(self.detection[idx], 0)
    #     #else: #1 detection
    #         #self.detection=np.squeeze(self.detection)


    # def unique_predict(self, image, thresh=0.01):
    #     #threshold for confidence detection
    #     # Inference
    #     results = self.model(image) #might need to specify the size
    #     #results.xyxy: [xmin, ymin, xmax, ymax, conf, class]
    #     detect_pandas=results.pandas().xyxy
    #     self.detection=np.array(detect_pandas)
    #     if self.verbose is True: print(self.detection)
    #     if (self.detection.shape[1]!=0):
    #         #print("DETECTED SOMETHING !")
    #         #use np.squeeze to remove 0 dim from the tensor
    #         self.detection=np.squeeze(self.detection,axis=0) 

    #         #class function to decide which detection to keep
    #         self.best_detection()
    #         if(self.detection[0][4]>thresh):
    #             label=True
    #         #modify the format of detection for bbox
    #         bbox=self.bbox_format()
    #         return bbox, label
    #     return None,False

    def predict(self, image):
        # Inference
        results = self.model(image) #might need to specify the size
        detect_pandas=results.pandas().xyxy  #results.xyxy: [xmin, ymin, xmax, ymax, conf, class]
        self.detection=np.array(detect_pandas)
        if self.verbose >=3: 
            print("shape of the detection: ", self.detection.shape)
        if (self.detection.shape[1]!=0):
            self.detection=np.squeeze(self.detection,axis=0)   #use np.squeeze to remove 0 dim from the tensor
            bbox=[]
            for idx, det in enumerate(self.detection):
                if det[4]>= self.thresh_bbox:
                    det=det[:4]
                    new_bbox=Utils.bbox_x1y1x2y2_to_xcentycentwh(det)
                    bbox.append(new_bbox)
            if not bbox:
                if self.verbose >=3:
                    print("Every detection below bbox thresh for yolo")
                return None
            #bbox=self.bbox_format() #modify the format of detection for [xcenter, ycenter, width, height]
            if self.verbose >= 3:
                print(f"Yolov5 Detection bboxes {bbox}")
            return bbox
        else:
            if self.verbose:
                print("Yolov5 no detections !")
        return None