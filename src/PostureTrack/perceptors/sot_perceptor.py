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
        obj_detection_time=(toc1 - tic1) * 1e3
        if self.verbose >= 2:
            print(f"Elapsed time for detector forward pass: {obj_detection_time:.1f}ms")
        self.add_inference_time("Object Detection", obj_detection_time)
        
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
            tracker_time=(toc2 - tic2) * 1e3
            if self.verbose >=2 :
                print(f"Elapsed time for tracker forward pass: {tracker_time:.1f}ms")
            self.add_inference_time("Tracker", tracker_time)
        #elif not self.tracker: 
            #No trackers provided just output the list of all detections
            #bbox=bbox_list
        else: 
            bbox=None
        
        #Pose estimation
        if bbox :
            tic3=time.perf_counter()
            if self.keypoints3D_activ:
                res_keypoints=self.keypoints.inference_3Dkeypoints(img, bbox)
                #handle result of keypoints
            elif self.keypoints:
                img_kpts, keypts= self.keypoints.inference_keypoints(img, bbox)
                self.img_kpts=img_kpts
            toc3=time.perf_counter()
            keypoints_time=(toc3 - tic3) * 1e3
            if self.keypoints or self.keypoints3D_activ:
                #add the inference time of the keypoints to the list of inference times
                self.add_inference_time("Pose Estimation", keypoints_time)
        
        toc4=time.perf_counter()
        total_time=(toc4 - tic1) * 1e3
        if self.verbose >=2 :
            print(f"Elapsed time for full perceptor forward pass: {total_time:.1f}ms")
        self.add_inference_time("Total", total_time)

        return bbox
