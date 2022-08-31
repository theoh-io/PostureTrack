import numpy as np
from mmtrack.apis import inference_mot
import os
from utilities import Utils
from trackers.base_mmtracking import BaseTracker


class MotTracker(BaseTracker):

    def forward(self, img: np.ndarray) -> list:
        '''
        cut_imgs: img parts cut from img at bbox positions
        detections: bboxes from YOLO detector
        img: original image
        -> bbox
        '''
        bbox_list=[]
        result = inference_mot(self.tracker, img, frame_id=self.frame)
        self.frame+=1
        track_bboxes=result['track_bboxes']
        track_bboxes=np.squeeze(track_bboxes)
        if np.array(track_bboxes).ndim==1:
            track_bboxes=[track_bboxes]
        for detection in track_bboxes:
            id=detection[0]
            bbox=detection[1:5]
            conf=detection[5]
            bbox=Utils.bbox_x1y1x2y2_to_xcentycentwh(bbox)
            bbox = [int(x) for x in bbox]
            if self.verbose >=3: 
                print(f"id {id} with bbox{bbox}")
            bbox_list.append(bbox)
        if bbox_list:
            return bbox_list
        else:
            return None

