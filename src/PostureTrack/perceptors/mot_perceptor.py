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
        # Tracking
        tic2 = time.perf_counter()
        #if bbox_list is not None and self.tracker:
        bbox_list = self.tracker.forward(image)
        if self.verbose >=3 :
            print(f"bbox tracker {bbox_list}")
        toc2 = time.perf_counter()
        if self.verbose >= 2:
            print(f"Elapsed time for tracker forward pass: {(toc2 - tic2) * 1e3:.1f}ms")
      
        return bbox_list