import numpy as np
import os 
import time
from PIL import Image
import cv2


from utilities import Utils
from perceptors.base_perceptor import BasePerceptor




class MotPerceptor(BasePerceptor):
    def forward(self, img):
        # Detection
        image=img
        #tic1 = time.perf_counter()
        # bbox_list = self.detector.predict(image)
        # print(f"bbox yolo {bbox_list}")
        # toc1 = time.perf_counter()
        # if self.verbose:
        #     print(f"Elapsed time for detector forward pass: {(toc1 - tic1) * 1e3:.1f}ms")

        # Tracking
        tic2 = time.perf_counter()
        #if bbox_list is not None and self.tracker:
        bbox_list = self.tracker.forward(image)
        print(f"bbox tracker{bbox_list}")
        toc2 = time.perf_counter()
        if self.verbose:
            print(f"Elapsed time for tracker forward pass: {(toc2 - tic2) * 1e3:.1f}ms")
        #elif not self.tracker: 
            #No trackers provided just output the list of all detections
            #bbox=bbox_list
        # else: 
        #     bbox=None

        # toc3 = time.perf_counter()
        # if self.verbose:
        #         print(f"Elapsed time for perceptor forward pass: {(toc3 - tic1) * 1e3:.1f}ms")

        return bbox_list