import cv2
import numpy as np
import copy
from pathlib import Path
from utilities import Utils
import os
import sys

#CWD is PostureTrack
path_cwd=os.getcwd()
path_outputs=path_cwd+"/outputs/"
path_mmpose=os.path.join((Path(os.getcwd()).parents[1]), "libs/mmpose") 
sys.path.append(path_mmpose)
from mmpose.apis import (collect_multi_frames, extract_pose_sequence,
                         get_track_id, inference_pose_lifter_model,
                         inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result, vis_3d_pose_result)
from mmpose.core import Smoother
from mmpose.datasets import DatasetInfo
from mmpose.models import PoseLifter, TopDown

class Keypoints2D():
    def __init__(self, path_weights, path_config, activated3D, path_weights3D, path_config3D, device, img_resolution, show3D=False, save_video_keypoints=False,smooth=False):
        self.path_weights=path_weights
        self.path_config=path_config
        self.activated3D=activated3D
        if self.activated3D:
            self.path_weights3D=path_weights3D
            self.path_config3D=path_config3D
        self.device=device
        self.resolution=img_resolution
        self.smooth= smooth
        self.frame_idx=0
        self.show_3Dkeypoints=show3D
        self.save_video_keypoints=bool(save_video_keypoints)
        self.name_video_keypoints=save_video_keypoints
        if save_video_keypoints:
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.fps = 3.5
            self.writer = None
        self.init_keypoints()
        if self.activated3D:
            self.init_3Dkeypoints()
        
    def init_keypoints(self):
        print("in init keypoints")
        pose_detector_checkpoint=self.path_weights
        pose_detector_config=self.path_config
        #pose_detector_config=os.path.join(path_mmpose,"configs/body/2d_kpt_sview_rgb_img/deeppose/coco/res152_coco_384x288_rle.py")
        #pose_detector_config=os.path.join(path_mmpose,"configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py")
        
        #pose_detector_checkpoint="https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_coco_384x288_rle-b77c4c37_20220624.pth"
        #pose_detector_checkpoint="https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"
        self.pose_det_model = init_pose_model(
            pose_detector_config,
            pose_detector_checkpoint,
            device=self.device)

        self.pose_det_dataset = self.pose_det_model.cfg.data['test']['type']

        self.dataset_info = self.pose_det_model.cfg.data['test'].get('dataset_info', None)
        if self.dataset_info is None:
            warnings.warn(
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning)
        else:
            self.dataset_info = DatasetInfo(self.dataset_info)
            #l.319
        
        self.pose_det_results_list = []
        self.next_id = 0
        self.pose_det_results = []

    def inference_keypoints(self, frame, bbox):
        pose_det_results_last = self.pose_det_results
        bbox=Utils.bbox_xcentycentwh_to_xtlytlwh(bbox)
        self.pose_det_results, _ = inference_top_down_pose_model(
            self.pose_det_model,
            frame,
            [{"bbox":bbox}],
            bbox_thr=None,
            format='xywh',
            dataset=self.pose_det_dataset,
            dataset_info=self.dataset_info,
            return_heatmap=None,
            outputs=None)
        # get track id for each person instance
        self.pose_det_results, self.next_id = get_track_id(
            self.pose_det_results,
            pose_det_results_last,
            self.next_id,
            use_oks=False,
            tracking_thr=0.3)

        if self.activated3D:
        #only needed if 3D keypoints activated
        # convert keypoint definition
            for res in self.pose_det_results:
                keypoints = res['keypoints']
                res['keypoints'] = Utils.convert_keypoint_definition(
                    keypoints, self.pose_det_dataset, self.pose_lift_dataset)
            
        self.pose_det_results_list.append(copy.deepcopy(self.pose_det_results))
        
        img_vis=vis_pose_result(
            self.pose_det_model,
            frame,
            self.pose_det_results,
            dataset=self.pose_det_dataset,
            dataset_info=self.dataset_info,
            kpt_score_thr=0.3,
            radius=3,
            thickness=1,
            show=False,
            out_file=None)
        
        
        return img_vis, self.pose_det_results
    
        # cv2.imshow('Camera Loomo',img_vis)
        # cv2.waitKey(1)

    def init_3Dkeypoints(self):
            pass