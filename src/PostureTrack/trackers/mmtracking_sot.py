import torch
import numpy as np
from mmtrack.apis import inference_sot, init_model
import os
from utilities import Utils
from trackers.base_mmtracking import BaseTracker


class SotaTracker(BaseTracker):

    def forward(self, detections: list, img: np.ndarray) -> list:
        '''
        detections: bboxes from YOLO detector
        img: original image
        -> bbox
        '''
        if self.frame==0:
            init_bbox=detections[0]
            self.new_bbox=Utils.bbox_xcentycentwh_to_x1y1x2y2(init_bbox)

        #input of the bbox format is x1, y1, x2, y2
        result = inference_sot(self.tracker, img, self.new_bbox, frame_id=self.frame)
        
        self.frame+=1
        track_bbox=result['track_bboxes']
        #remove last index -1
        confidence=track_bbox[4]
        if self.verbose:
            print(f"Tracking conf is: {confidence}")
        bbox=track_bbox[:4]#[test_bbox[0], test_bbox[1], test_bbox[2]-test_bbox[0], test_bbox[3]-test_bbox[1]]
        
        if confidence>self.conf_thresh:
            #changing back format from (x1, y1, x2, y2) to (xcenter, ycenter, width, height) before writing
            bbox=Utils.bbox_x1y1x2y2_to_xcentycentwh(bbox)
            bbox = [int(x) for x in bbox]
        else:
            if self.verbose:
                print("Under Tracking threshold")
            #bbox=[0, 0, 0, 0]
            bbox=None

        return bbox

