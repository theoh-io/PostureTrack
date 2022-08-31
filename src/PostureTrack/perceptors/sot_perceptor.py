import numpy as np
import os 
import time
from PIL import Image
import cv2


from utilities import Utils
from perceptors.base_perceptor import BasePerceptor




class SotPerceptor(BasePerceptor):
    
    def forward(self, img):
        # Detection

        tic1 = time.perf_counter()
        bbox_list = self.detector.predict(img)
        toc1 = time.perf_counter()
        if self.verbose >= 2:
            print(f"Elapsed time for detector forward pass: {(toc1 - tic1) * 1e3:.1f}ms")

        # #Solve this to make it clearer always bbox_list = None if no detections
        # if bbox_list is not None:
        #     # if self.use_img_transform:
        #     cut_imgs = Utils.crop_img_bbox(bbox_list,image)
        #     # if self.verbose and cut_imgs is not None: print("in preprocessing: ", bbox_list)
        #     # else:
        #     #     cut_imgs = None
        # else:
        #     if self.verbose is True: print("No person detected.")
        cut_imgs = None

        # Tracking
        tic2 = time.perf_counter()
        if bbox_list is not None and self.tracker:
            bbox = self.tracker.forward(bbox_list,img)
            toc2 = time.perf_counter()
            if self.verbose >=2 :
                print(f"Elapsed time for tracker forward pass: {(toc2 - tic2) * 1e3:.1f}ms")
        #elif not self.tracker: 
            #No trackers provided just output the list of all detections
            #bbox=bbox_list
        else: 
            bbox=None

        return bbox
