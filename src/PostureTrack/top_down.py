#!/usr/bin/env python3

##################
#IMPORTS
##################
#import matplotlib.pyplot as plt
import time
from utilities import Utils, FrameGrab
#import glob
#from datetime import datetime

import os
import sys
import cv2
import numpy as np
from perceptors import sot_perceptor, mot_perceptor
from trackers import mmtracking_sot, mmtracking_mot
import hydra
from omegaconf import DictConfig, OmegaConf

global tracker_type, detector_model, tracking_conf, verbose, source, gt, path_output_vid, output_json

# @hydra.main(version_base=None, config_path="config", config_name="cfg_topdown")
# def hydra_topdown(cfg: DictConfig) -> None:
#     global tracker_type, detector_model, tracking_conf, verbose, source, gt, path_output_vid, output_json
#     #print(OmegaConf.to_yaml(cfg))
#     tracker_type=cfg.settings.tracker_type
#     detector_model=cfg.settings.detector_model
#     tracking_conf=cfg.settings.tracking_conf
#     verbose=cfg.settings.verbose

#     source=cfg.settings.source
#     gt=cfg.settings.gt
#     #path_output_vid=cfg.settings.output_vid
#     #output_json=cfg.settings.output_json


#def main():
#if __name__ == "__main__":
def TopDown(detector_cfg, tracker_cfg, verbose,
                    source, gt, path_output_vid, path_output_json):

    ###################################
    # Load ConfigFile Arguments
    ####################################
    #hydra_topdown()
    detector_type=detector_cfg["type"]
    detector_model=detector_cfg["size"]
    detector_thresh=detector_cfg["thresh"]
    tracker_type= tracker_cfg["type"]
    tracking_conf= tracker_cfg["conf"]
    path_cfg= tracker_cfg["cfg"]
    path_weights= tracker_cfg["weights"]
    print(f"tracker_type {tracker_type}")
    ###################################
    # Initialize Full detector
    ###################################

    # Initialize Detector Configuration 
    if tracker_type == "Yolov7":
        from detectors import yolov7_detector
        perceptor = sot_perceptor.SotPerceptor(width = 640, height = 480, channels = 3, downscale = 1,
                                                detector = yolov7_detector.Yolov7Detector, detector_size=detector_model, 
                                                tracker=None, tracker_model=None, tracking_conf=None,
                                                type_input = "opencv", verbose=verbose)
    elif tracker_type =="Stark":
        from detectors import yolov7_detector
        perceptor = sot_perceptor.SotPerceptor(width = 640, height = 480, channels = 3, downscale = 1,
                                                detector = yolov7_detector.Yolov7Detector, detector_size=detector_model, 
                                                tracker=mmtracking_sot.SotaTracker, tracker_model="Stark", tracking_conf=tracking_conf,
                                                type_input = "opencv", verbose=verbose)
    elif tracker_type =="ByteTrack":
        from detectors import yolov5_detector
        perceptor = mot_perceptor.MotPerceptor(width = 640, height = 480, channels = 3, downscale = 1,
                                                detector = yolov5_detector.Yolov5Detector, detector_size=detector_model, 
                                                tracker=mmtracking_mot.MotTracker, tracker_model="ByteTrack", tracking_conf=tracking_conf,
                                                type_input = "opencv", verbose=verbose)
    
    else:
        print(f"tracker type {tracker_type} is not implemented.")


    ##################################
    # Load Input Source/Video Writer #
    ##################################
    # start video stream

    #df_gt=Utils.load_groundtruth(path_groundtruth)
    print(f"Using: {source} as input")
    grab = FrameGrab(mode="video", path=source)

    if path_output_vid:
        # get vcap property 
        width  = int(grab.cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
        height = int(grab.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
        fps = int(grab.cap.get(cv2.CAP_PROP_FPS))
        print(f"width: {width}, height: {height}, fps: {fps}")
        output_vid = cv2.VideoWriter(path_output_vid, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
   
    ##############################
    # Main Loop                  #
    ##############################

    bboxes_to_save = []
    elapsed_time_list=[]
    frame_number=0

    while(True):
        img = grab.read_cap()
        if img is None:
            print("Stop reading.")
            break
        tic = time.perf_counter()
        # INFERENCE
        bbox_list = perceptor.forward(img)
        toc = time.perf_counter()
        if verbose:
            print(f"Elapsed time for whole fordward pass: {(toc-tic)*1e3:.1f}ms")
        #record bbox and elapsed time
        bboxes_to_save.append(bbox_list)
        elapsed_time_list.append((toc-tic)*1e3)

        # VISUALIZATION

        Utils.visualization(img, bbox_list, (255, 0, 0), 2)
        if gt is not None:
            truth=df_gt[frame_number]
            truth=Utils.bbox_x1y1wh_to_xcentycentwh(truth)
            Utils.visualization(img, truth, (0, 255, 0), 1)

        if verbose:
            print("bbox:", bbox_list)

        if path_output_vid is not None:
            output_vid.write(img)

        #To get result before the end quit the program with q instead of Ctrl+C
        k = cv2.waitKey(10) & 0xFF
        if k == ord('q'):
            break
        frame_number+=1

    average_forward_time=np.mean(elapsed_time_list)
    if verbose:
        print(f"Average time for a forward pass is {average_forward_time:.1f}ms")
    #Utils.save_results(detector, bboxes_to_save)

    cv2.destroyAllWindows()
    del grab