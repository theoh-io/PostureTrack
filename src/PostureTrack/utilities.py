from easydict import EasyDict
import yaml
import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple
import itertools
import logging
from PIL import Image
import torch
from importlib import import_module
from pathlib import Path
import pandas as pd
import os

class Utils():
    @staticmethod
    def get_scaled_img(img: np.ndarray, scale_percent: float=20) -> np.ndarray:
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    @staticmethod
    def get_resized_img(img: np.ndarray, new_dim: Tuple[int,int]=(200,140)):
        return cv2.resize(img, new_dim, interpolation = cv2.INTER_AREA)
    
    # FIXME Do it consistent! Always use one representation for bboxes... -> Maybe x1,y1,x2,y2
    @staticmethod
    def bounding_box_from_points(points):
        x_coordinates, y_coordinates = zip(*points)
        return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]

    @staticmethod
    def get_bbox_xyxy_from_xy_wh(bbox):
        return [bbox[0][0],bbox[0][1],bbox[0][0] + bbox[1][0],bbox[0][1] + bbox[1][1]]
    
    @staticmethod
    def get_bbox_xyxy_from_xy_xy(bbox):
        return [bbox[0][0],bbox[0][1],bbox[1][0],bbox[1][1]]
    

    @staticmethod
    def get_bbox_as_union_from_two_bboxes_xyxy(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        return xA,yA,xB,yB

    @staticmethod
    def bb_intersection_over_union(boxA, boxB):
        boxA = Utils.get_bbox_xyxy_from_xy_wh(boxA)
        boxB = Utils.get_bbox_xyxy_from_xy_wh(boxB)
        xA,yA,xB,yB = Utils.get_bbox_as_union_from_two_bboxes_xyxy(boxA, boxB)
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    @staticmethod
    def get_detection_with_max_IoU(bboxesA: list, bboxesB: list) -> Tuple[np.ndarray, float]:
        
        all_combinations = list(itertools.product(bboxesA, bboxesB))
        
        #candidates.sort(key=lambda x: (x[0] - ref[0]) ** 2 + (x[1] - ref[1]) ** 2)  
        # FIXME Do that nice!
        best_iou_metric = 0
        ic_best = -1
        for ic in range(len(all_combinations)):
            comb = all_combinations[ic]
            iou_metric = Utils.bb_intersection_over_union(comb[0],comb[1])
            if iou_metric >= best_iou_metric: #FIXME Not true
                ic_best = ic
                best_iou_metric = iou_metric
    
        # print("ic_best",ic_best)
        det_bbox_A = Utils.get_bbox_xyxy_from_xy_xy(all_combinations[ic_best][0])
        det_bbox_B = Utils.get_bbox_xyxy_from_xy_xy(all_combinations[ic_best][1])
        detected_bbox = list(Utils.get_bbox_as_union_from_two_bboxes_xyxy(det_bbox_A,det_bbox_B)) # FIXME This is in some way wrong... -> use utils.BBox class for all bbox operations
        
        return detected_bbox, best_iou_metric, det_bbox_A, det_bbox_B
    
    @staticmethod
    def get_bbox_xcent_ycent_w_h_from_xy_xy(bbox):
        #original format: [(top_left), (bot-right)]
        bbox=[bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]
        x_center=(bbox[0]+bbox[2])/2
        y_center=(bbox[1]+bbox[3])/2
        width=bbox[2]-bbox[0]
        height=bbox[3]-bbox[1]
        bbox=[x_center, y_center, width, height]
        bbox=np.array(bbox).astype(int)
        return bbox

    @staticmethod
    def get_bbox_tlwh_from_xcent_ycent_w_h(img_xc_yc_w_h):
        x_cent, y_cent, w, h = img_xc_yc_w_h
        x_top_left = x_cent - w//2
        y_top_left = y_cent - h//2
        return np.array([x_top_left, y_top_left, w, h])

    @staticmethod
    def get_xyah_from_tlwh(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
    @staticmethod
    def get_xyah_from_xc_yc_wh(xywh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = xywh.copy()
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def bbox_xcentycentwh_to_x1y1x2y2(bbox):
        #convert from (xcenter,y_center, width, height) to (x1,y1,x2,y2)
        offset_x=int(bbox[2]/2)
        offset_y=int(bbox[3]/2)
        new_bbox=[0, 0, 0, 0]
        new_bbox[0]=bbox[0]-offset_x
        new_bbox[1]=bbox[1]-offset_y
        new_bbox[2]=bbox[0]+offset_x
        new_bbox[3]=bbox[1]+offset_y
        return new_bbox

    @staticmethod
    def bbox_x1y1x2y2_to_xcentycentwh(bbox):
        #convert from (x1,y1,x2,y2) to (xcenter,y_center, width, height)
        offset_x=int((bbox[2]-bbox[0])/2)
        offset_y=int((bbox[3]-bbox[1])/2)
        bbox[0]=int(bbox[0]+offset_x)
        bbox[1]=int(bbox[1]+offset_y)
        bbox[2]=offset_x*2
        bbox[3]=offset_y*2
        return bbox

    @staticmethod
    def bbox_x1y1wh_to_xcentycentwh(bbox):
        #convert from (x1,y1,x2,y2) to (xcenter,y_center, width, height)
        offset_x=int(bbox[2]/2)
        offset_y=int(bbox[3]/2)
        bbox[0]=bbox[0]+offset_x
        bbox[1]=bbox[1]+offset_y
        return bbox

    @staticmethod
    def bbox_xcentycentwh_to_xtlytlwh(bbox):
        #convert from (xcenter,ycenter,width,height) to (x_topleft,y_topleft, width, height)
        x_tl= bbox[0] - bbox[2]/2
        y_tl= bbox[1] - bbox[3]/2
        w= bbox[2]
        h= bbox[3]
        new_bbox=[x_tl, y_tl, w, h]
        return new_bbox

    @staticmethod
    def bbox_xtlytlwh_to_xcentycentwh(bbox):
        #convert from (x_topleft,y_topleft, width, height) to (xcenter,ycenter,width,height)
        x= bbox[0] + bbox[2]/2
        y= bbox[1] + bbox[3]/2
        w= bbox[2]
        h= bbox[3]
        new_bbox=[x, y, w, h]
        return new_bbox

    @staticmethod
    def average_bbox(bbox_list):
        #for bbox in bbox_list:
        bbox=np.mean(bbox_list, axis=0)
        return bbox

            

    @staticmethod
    def crop_img_parts_from_bboxes(bbox_list: list, img: np.ndarray, image_processing: Callable):
        img_list=[]
        if bbox_list is not None and bbox_list[0] is not None:
            # if bbox_list[0] is not None:
            for i in range(bbox_list.shape[0]):
                bbox_indiv=bbox_list[i]
                crop_img=np.array(img[int((bbox_indiv[1]-bbox_indiv[3]/2)):int((bbox_indiv[1]+bbox_indiv[3]/2)), int((bbox_indiv[0]-bbox_indiv[2]/2)):int((bbox_indiv[0]+bbox_indiv[2]/2))])
                #to apply the normalization need a PIL image
                # PIL RGB while CV is BGR.
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                crop_img = Image.fromarray(crop_img)
                tensor_img_indiv=image_processing(crop_img)
                tensor_img_indiv=torch.unsqueeze(tensor_img_indiv, 0)
                img_list.append(tensor_img_indiv)
            tensor_img=torch.cat(img_list)
            return tensor_img
        else:
            return None

    @staticmethod
    def import_from_string(class_str: str) -> object:
        try:
            module_path, class_name = class_str.rsplit('.', 1)
            module = import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(class_str)

    @staticmethod
    def convert_strtoint(text):
        if isinstance(text, str):
            if text.isnumeric():
                value=int(text)
                return value
        return text

    @staticmethod 
    def visualization(img, bbox_list: list, color=(0,0,255), thickness=2):
        if bbox_list is not None:
            if not isinstance(bbox_list[0], list):
                #handle the case of only 1 detection
                bbox_list=[bbox_list]
            for bbox in bbox_list:  
                start=(int(bbox[0]-bbox[2]/2), int(bbox[1]+bbox[3]/2)) #top-left corner
                stop= (int(bbox[0]+bbox[2]/2), int(bbox[1]-bbox[3]/2)) #bottom right corner
                cv2.rectangle(img, start, stop, color, thickness)
        cv2.imshow('Camera Loomo',img)
        cv2.waitKey(1)

    @staticmethod 
    def resized_visualization(img, dim, bbox_list: list, color=(0,0,255), thickness=2):
        if bbox_list is not None:
            if not isinstance(bbox_list[0], list):
                #handle the case of only 1 detection
                bbox_list=[bbox_list]
            for bbox in bbox_list:  
                start=(int(bbox[0]-bbox[2]/2), int(bbox[1]+bbox[3]/2)) #top-left corner
                stop= (int(bbox[0]+bbox[2]/2), int(bbox[1]-bbox[3]/2)) #bottom right corner
                cv2.rectangle(img, start, stop, color, thickness)
        img=cv2.resize(img, dim)
        cv2.imshow('Camera Loomo',img)
        cv2.waitKey(1)

    @staticmethod 
    def save_results(detector, bboxes_to_save):
        if detector.cfg.RECORDING.SAVE_RESULTS:
            bboxes_to_save = [b if b is not None else np.zeros(4) for b in bboxes_to_save]
            bboxes_to_save = np.array(bboxes_to_save, dtype=np.int16)

            folder_str = detector.cfg.RECORDING.FOLDER_FOR_PREDICTION
            # save_str = f"{folder_str}/{detector.cfg.PERCEPTION.BENCHMARK_FILE.replace('.','').replace('/','').replace('_','')}_tracker_{detector.cfg.TRACKER.TRACKER_CLASS[-11:]}.txt"
            save_str = f"{folder_str}/ID_{detector.cfg.RECORDING.EXPID:04d}_prediction"
            path = Path(f"{save_str}.txt")
            if path.is_file():
                print("File already exists. Did not store it.")
            else:
                print(f"Saving predicted bboxes to {save_str}.txt.")
                np.savetxt(f"{save_str}.txt", bboxes_to_save, fmt='%.i',delimiter=' , ')

                # FIXME Specify all parameters that are varied for a specif configuration
                #config_dict = {"EXP_ID": detector.cfg.RECORDING.EXPID, "DS_MAX_DIST": detector.cfg.DEEPSORT.MAX_DIST}

                #file = open(f"{save_str}.yaml", "w")
                #yaml.dump(config_dict,file)
                #file.close()
                detector.store_elapsed_time()

    @staticmethod
    def load_groundtruth(path_ground_truth, verbose=False):
        try:
            path_current=os.getcwd()
            path_txt=os.path.join(path_current, path_ground_truth)   
            data = pd.read_csv(path_txt, header=None, names= ["x_center", "y_center", "width", "height"], index_col=None)  
            data.index = np.arange(1, len(data) + 1)  #start frame index at 1
            #if verbose is True: print(data)
            data=data.to_numpy()
            #only if needed bbox format
            #for i in data.shape[0]:
            #    bbox=data[i]
            #    data[i]=[(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2, bbox[2]-bbox[0], bbox[1]-bbox[3]]
            
            if verbose is True: print("in load gt :", data)
            print("successfully loaded gt")
        except:
            print("path provided to ground_truth is not Working.")
            return None
        return data
    
    @staticmethod
    def convert_keypoint_definition(keypoints, pose_det_dataset,
                                pose_lift_dataset):
        """Convert pose det dataset keypoints definition to pose lifter dataset
        keypoints definition, so that they are compatible with the definitions
        required for 3D pose lifting.
        Args:
            keypoints (ndarray[K, 2 or 3]): 2D keypoints to be transformed.
            pose_det_dataset, (str): Name of the dataset for 2D pose detector.
            pose_lift_dataset (str): Name of the dataset for pose lifter model.
        Returns:
            ndarray[K, 2 or 3]: the transformed 2D keypoints.
        """
        assert pose_lift_dataset in [
            'Body3DH36MDataset', 'Body3DMpiInf3dhpDataset'
            ], '`pose_lift_dataset` should be `Body3DH36MDataset` ' \
            f'or `Body3DMpiInf3dhpDataset`, but got {pose_lift_dataset}.'

        coco_style_datasets = [
            'TopDownCocoDataset', 'TopDownPoseTrack18Dataset',
            'TopDownPoseTrack18VideoDataset'
        ]
        keypoints_new = np.zeros((17, keypoints.shape[1]), dtype=keypoints.dtype)
        if pose_lift_dataset == 'Body3DH36MDataset':
            if pose_det_dataset in ['TopDownH36MDataset']:
                keypoints_new = keypoints
            elif pose_det_dataset in coco_style_datasets:
                # pelvis (root) is in the middle of l_hip and r_hip
                keypoints_new[0] = (keypoints[11] + keypoints[12]) / 2
                # thorax is in the middle of l_shoulder and r_shoulder
                keypoints_new[8] = (keypoints[5] + keypoints[6]) / 2
                # spine is in the middle of thorax and pelvis
                keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
                # in COCO, head is in the middle of l_eye and r_eye
                # in PoseTrack18, head is in the middle of head_bottom and head_top
                keypoints_new[10] = (keypoints[1] + keypoints[2]) / 2
                # rearrange other keypoints
                keypoints_new[[1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = \
                    keypoints[[12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]]
            elif pose_det_dataset in ['TopDownAicDataset']:
                # pelvis (root) is in the middle of l_hip and r_hip
                keypoints_new[0] = (keypoints[9] + keypoints[6]) / 2
                # thorax is in the middle of l_shoulder and r_shoulder
                keypoints_new[8] = (keypoints[3] + keypoints[0]) / 2
                # spine is in the middle of thorax and pelvis
                keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
                # neck base (top end of neck) is 1/4 the way from
                # neck (bottom end of neck) to head top
                keypoints_new[9] = (3 * keypoints[13] + keypoints[12]) / 4
                # head (spherical centre of head) is 7/12 the way from
                # neck (bottom end of neck) to head top
                keypoints_new[10] = (5 * keypoints[13] + 7 * keypoints[12]) / 12

                keypoints_new[[1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]] = \
                    keypoints[[6, 7, 8, 9, 10, 11, 3, 4, 5, 0, 1, 2]]
            elif pose_det_dataset in ['TopDownCrowdPoseDataset']:
                # pelvis (root) is in the middle of l_hip and r_hip
                keypoints_new[0] = (keypoints[6] + keypoints[7]) / 2
                # thorax is in the middle of l_shoulder and r_shoulder
                keypoints_new[8] = (keypoints[0] + keypoints[1]) / 2
                # spine is in the middle of thorax and pelvis
                keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
                # neck base (top end of neck) is 1/4 the way from
                # neck (bottom end of neck) to head top
                keypoints_new[9] = (3 * keypoints[13] + keypoints[12]) / 4
                # head (spherical centre of head) is 7/12 the way from
                # neck (bottom end of neck) to head top
                keypoints_new[10] = (5 * keypoints[13] + 7 * keypoints[12]) / 12

                keypoints_new[[1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]] = \
                    keypoints[[7, 9, 11, 6, 8, 10, 0, 2, 4, 1, 3, 5]]
            else:
                raise NotImplementedError(
                    f'unsupported conversion between {pose_lift_dataset} and '
                    f'{pose_det_dataset}')

        elif pose_lift_dataset == 'Body3DMpiInf3dhpDataset':
            if pose_det_dataset in coco_style_datasets:
                # pelvis (root) is in the middle of l_hip and r_hip
                keypoints_new[14] = (keypoints[11] + keypoints[12]) / 2
                # neck (bottom end of neck) is in the middle of
                # l_shoulder and r_shoulder
                keypoints_new[1] = (keypoints[5] + keypoints[6]) / 2
                # spine (centre of torso) is in the middle of neck and root
                keypoints_new[15] = (keypoints_new[1] + keypoints_new[14]) / 2

                # in COCO, head is in the middle of l_eye and r_eye
                # in PoseTrack18, head is in the middle of head_bottom and head_top
                keypoints_new[16] = (keypoints[1] + keypoints[2]) / 2

                if 'PoseTrack18' in pose_det_dataset:
                    keypoints_new[0] = keypoints[1]
                    # don't extrapolate the head top confidence score
                    keypoints_new[16, 2] = keypoints_new[0, 2]
                else:
                    # head top is extrapolated from neck and head
                    keypoints_new[0] = (4 * keypoints_new[16] -
                                        keypoints_new[1]) / 3
                    # don't extrapolate the head top confidence score
                    keypoints_new[0, 2] = keypoints_new[16, 2]
                # arms and legs
                keypoints_new[2:14] = keypoints[[
                    6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15
                ]]
            elif pose_det_dataset in ['TopDownAicDataset']:
                # head top is head top
                keypoints_new[0] = keypoints[12]
                # neck (bottom end of neck) is neck
                keypoints_new[1] = keypoints[13]
                # pelvis (root) is in the middle of l_hip and r_hip
                keypoints_new[14] = (keypoints[9] + keypoints[6]) / 2
                # spine (centre of torso) is in the middle of neck and root
                keypoints_new[15] = (keypoints_new[1] + keypoints_new[14]) / 2
                # head (spherical centre of head) is 7/12 the way from
                # neck (bottom end of neck) to head top
                keypoints_new[16] = (5 * keypoints[13] + 7 * keypoints[12]) / 12
                # arms and legs
                keypoints_new[2:14] = keypoints[0:12]
            elif pose_det_dataset in ['TopDownCrowdPoseDataset']:
                # head top is top_head
                keypoints_new[0] = keypoints[12]
                # neck (bottom end of neck) is in the middle of
                # l_shoulder and r_shoulder
                keypoints_new[1] = (keypoints[0] + keypoints[1]) / 2
                # pelvis (root) is in the middle of l_hip and r_hip
                keypoints_new[14] = (keypoints[7] + keypoints[6]) / 2
                # spine (centre of torso) is in the middle of neck and root
                keypoints_new[15] = (keypoints_new[1] + keypoints_new[14]) / 2
                # head (spherical centre of head) is 7/12 the way from
                # neck (bottom end of neck) to head top
                keypoints_new[16] = (5 * keypoints[13] + 7 * keypoints[12]) / 12
                # arms and legs
                keypoints_new[2:14] = keypoints[[
                    1, 3, 5, 0, 2, 4, 7, 9, 11, 6, 8, 10
                ]]

            else:
                raise NotImplementedError(
                    f'unsupported conversion between {pose_lift_dataset} and '
                    f'{pose_det_dataset}')

        return keypoints_new


def img_seq2vid(path_source, fps=15, verbose=False):
    #global  path_seq, path_source, seq_vid_fps, path_vid
    path_seq=path_source + '/img'
    sequences = os.listdir(path_seq)
    sequences=sorted(sequences)
    #if the video from the sequence of images doesn't exist => create it
    path_vid=os.path.join(path_source, "video.avi")
    exists=os.path.exists(path_vid)
    if exists is False:
        if verbose is True: print("creating video from img sequence")
        if verbose is True: print("first img name", sequences[0])
        if verbose is True: print("init image", os.path.join(path_seq, sequences[0]))
        init_img=cv2.imread(os.path.join(path_seq, sequences[0]))
        height, width, layers =init_img.shape
        size=(width, height)
        if verbose is True: print("Size of the input seq:", size)
        seq_vid=cv2.VideoWriter(path_vid, cv2.VideoWriter_fourcc(*'MJPG'), fps , size)
        for sequence in sequences:
            if verbose is True: print("Running sequence %s" % sequence)
            sequence_dir = os.path.join(path_seq, sequence)
            cvimg=cv2.imread(sequence_dir)
            seq_vid.write(cvimg)
        seq_vid.release()
    return path_vid



class FrameGrab:
    def __init__(self, mode: str ="webcam", path: str = None) -> None:
        self.cap = None
        if mode == "webcam": #FIXME Do it as enum
            self.cap = cv2.VideoCapture(0) 
        elif mode == "video":
            # self.cap = cv2.VideoCapture(FILDER + "Loomo/video.avi")
            path_vid=path
            self.cap = cv2.VideoCapture(path_vid)
        elif mode == "img":
            path_vid=img_seq2vid(path)
            self.cap = cv2.VideoCapture(path_vid)
        else:
            print(f"input file not working : {path_vid}")
        
    def read_cap(self) -> np.ndarray:
        success, image =  self.cap.read()
        if not success:
            logging.warning("Reading was not successful.")
            image = None
        return image
    
    def __del__(self):
        self.cap.release()
        print('Released cap.')

class BBox:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def bottom(self):
        return self.y + self.h
    
    def right(self):
        return self.x + self.w
    
    def area(self):
        return self.w * self.h

    def union(self, b):
        posX = min(self.x, b.x)
        posY = min(self.y, b.y)
        
        return BBox(posX, posY, max(self.right(), b.right()) - posX, max(self.bottom(), b.bottom()) - posY)
    
    def intersection(self, b):
        posX = max(self.x, b.x)
        posY = max(self.y, b.y)
        
        candidate = BBox(posX, posY, min(self.right(), b.right()) - posX, min(self.bottom(), b.bottom()) - posY)
        if candidate.w > 0 and candidate.h > 0:
            return candidate
        return BBox(0, 0, 0, 0)
    
    def ratio(self, b):
        return self.intersection(b).area() / self.union(b).area()


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class YamlInteract():
    def __init__(self):
        pass

    @staticmethod
    def load_dict_cfg_from_yaml_file(file_path: str) -> dict:
        file = open(file_path, "r")
        dictionary = yaml.safe_load(file)
        file.close()
        return dictionary

    @staticmethod
    def save_dict_cfg_from_yaml_file(cfg: dict, file_path: str):
        file = open(file_path, "w")
        yaml.dump(cfg, file)
        file.close()

    @staticmethod
    def load_easy_dict_cfg_from_yaml_file(file_path: str ) -> EasyDict:
        dictionary = YamlInteract.load_dict_cfg_from_yaml_file(file_path)
        easy_dict = EasyDict(dictionary)
        return easy_dict

    @staticmethod
    def merge_cfgs(cfg_1: dict, cfg_2: dict) -> EasyDict:
        return EasyDict({**cfg_1, **cfg_2})

    @staticmethod
    def save_cfgs():
        pass


class Interval(ABC):
    def __init__(self, min, max):
        self._min = min
        self._max = max

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    def asdict(self):
        return {'min': self.min, 'max': self.max}

    def astuple(self):
        return self.min, self.max

    @abstractmethod
    def __contains__(self, item):
        pass

    @abstractmethod
    def __str__(self):
        pass

class OpenInterval(Interval):
    def __init__(self, min, max):
        super().__init__(min, max)

    def __contains__(self, item):
        return self.min < item < self.max

    def __str__(self):
        return f"({self.min}, {self.max})"

class ClosedInterval(Interval):
    def __init__(self, min, max):
        super().__init__(min, max)

    def __contains__(self, item):
        return self.min <= item <= self.max

    def __str__(self):
        return f"[{self.min}, {self.max}]"

class ClosedIntervalNotDirected(Interval):
    def __init__(self, min, max):
        super().__init__(min, max)

    def __contains__(self, item):
        return self.min <= item <= self.max or self.max <= item <= self.min

    def __str__(self):
        return f"[{self.min}, {self.max}]"