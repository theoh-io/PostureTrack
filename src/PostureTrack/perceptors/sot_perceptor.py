import numpy as np
import os 
import time
from PIL import Image
import cv2


from utilities import Utils
from perceptors.base_perceptor import BasePerceptor




class SotPerceptor(BasePerceptor):
    def preproc(self,image):
        # Adapt image to detector requirements
        pil_image = Image.frombytes('RGB', (self.width,self.height), image)
        opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        opencvImage = cv2.cvtColor(opencvImage,cv2.COLOR_BGR2RGB)

        if self.type_input == "opencv":
            image = opencvImage
        
        elif self.type_input == "pil":
            image = pil_image

        return image

    def forward(self, img):
        #image=self.preproc(img)
        # Detection
        image=img
        tic1 = time.perf_counter()
        bbox_list = self.detector.predict(image)
        print(f"bbox yolo {bbox_list}")
        toc1 = time.perf_counter()
        if self.verbose:
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
            bbox = self.tracker.forward(cut_imgs,bbox_list,image)
            print(f"bbox tracker{bbox}")
            toc2 = time.perf_counter()
            if self.verbose:
                print(f"Elapsed time for tracker forward pass: {(toc2 - tic2) * 1e3:.1f}ms")
        #elif not self.tracker: 
            #No trackers provided just output the list of all detections
            #bbox=bbox_list
        else: 
            bbox=None

        toc3 = time.perf_counter()
        if self.verbose:
                print(f"Elapsed time for perceptor forward pass: {(toc3 - tic1) * 1e3:.1f}ms")

        return bbox
