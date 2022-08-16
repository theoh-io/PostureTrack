import torch
import numpy as np
from mmtrack.apis import inference_mot, init_model
import os
from utilities import Utils


class MotTracker():
    def __init__(self, tracker_name="ByteTrack", tracking_conf="0.5", device='gpu', verbose="False"):
        '''
        init_model parameters: 
        tracker_name (ByteTrack) 
        desired device to specify cpu if wanted
        '''
        #FIX: download weights and set the path to chckpt and cfg
        desired_device = device
        if tracker_name=="ByteTrack" or tracker_name=="bytetrack":
            #print(os.getcwd())
            path_config=os.path.abspath(os.path.join(os.getcwd(),"../../libs/mmtracking/configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private.py"))#"configs/stark_st2_r50_50e_lasot.py"
            path_model=os.path.abspath(os.path.join(os.getcwd(),"trackers/weights/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth"))
        else:
            print("Other MOT trackers still not implemented")


        cpu = 'cpu' == desired_device
        cuda = not cpu and torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.tracker = init_model(path_config, path_model, self.device) 
        #prog_bar = mmcv.ProgressBar(len(imgs))
        self.conf_thresh=tracking_conf
        self.frame=0
        self.verbose=verbose
        if self.verbose:
            print(f"-> Using tracker {tracker_name} from MMTracking")


    def forward(self, img: np.ndarray) -> list:
        '''
        cut_imgs: img parts cut from img at bbox positions
        detections: bboxes from YOLO detector
        img: original image
        -> bbox
        '''
        # if self.frame==0:
        #     init_bbox=detections[0]
        #     self.new_bbox=Utils.bbox_xcentycentwh_to_x1y1x2y2(init_bbox)

        #input of the bbox format is x1, y1, x2, y2
        bbox_list=[]
        result = inference_mot(self.tracker, img, frame_id=self.frame)
        print(f"frame {self.frame}")
        self.frame+=1
        #print(f"results: {result}")
        track_bboxes=np.squeeze(result['track_bboxes'])
        #print(track_bboxes)
        for detection in track_bboxes:
            id=detection[0]
            bbox=detection[1:5]
            conf=detection[5]
            bbox=Utils.bbox_x1y1x2y2_to_xcentycentwh(bbox)
            bbox = [int(x) for x in bbox]
            print(f"id {id} with bbox{bbox}")
            bbox_list.append(bbox)
        # if self.verbose:
        #     print(f"Tracking conf is: {confidence}")
        # #bbox=track_bbox[:4]#[test_bbox[0], test_bbox[1], test_bbox[2]-test_bbox[0], test_bbox[3]-test_bbox[1]]
        
        # if confidence>self.conf_thresh:
        #     #changing back format from (x1, y1, x2, y2) to (xcenter, ycenter, width, height) before writing
        #     bbox=Utils.bbox_x1y1x2y2_to_xcentycentwh(bbox)
        #     bbox = [int(x) for x in bbox]
        # else:
        #     if self.verbose:
        #         print("Under Tracking threshold")
        #     #bbox=[0, 0, 0, 0]
        #     bbox=None

        return bbox_list

