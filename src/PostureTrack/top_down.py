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
from keypoints import keypoints3D, keypoints2D
import hydra
from omegaconf import DictConfig, OmegaConf

#global tracker_type, detector_model, tracking_conf, verbose, device, source, gt, path_output_vid, output_json

def TopDown(detector_cfg, tracker_cfg, pose_cfg, verbose, device, 
                    source, gt, path_output_vid, path_output_json):

    ###################################
    # Load ConfigFile Arguments
    ####################################
    #hydra_topdown()
    device=Utils.convert_strtoint(device)
    verbose=Utils.convert_strtoint(verbose)
    detector_type=detector_cfg["type"]
    # detector_size=detector_cfg["size"]
    # detector_thresh=detector_cfg["thresh"]
    # tracker_name=tracker_cfg["name"]
    tracker_type= tracker_cfg["type"]
    # tracking_conf= tracker_cfg["conf"]
    # path_cfg_tracker= tracker_cfg["cfg"]
    # path_weights_tracker= tracker_cfg["weights"]
    if pose_cfg:
        keypoints_name=pose_cfg["name"]
    else:
        keypoints_name=None
    #     if pose_cfg["3D"]:
    #         keypoints3D_activ=True
    #         keypoints3D_name=pose_cfg["3Dname"]
    #         for key in pose_cfg:
    #             print(f"the key name is {key} and its value is {pose_cfg[key]}")
    #         path_output_3D=pose_cfg["path_output_3D"]
    #     else:
    #         keypoints3D_activ=False
    #         keypoints3D_name=None
    #         path_output_3D=None
    # else:
    #     keypoints_name=None
    #     keypoints3D_activ=False
    #     keypoints3D_model=None
    #     path_output_3D=None
            

    ###################################
    # Initialize Full detector
    ###################################

    # Initialize Detector Configuration 
    
    #Problem when loading both detector at the same time
    if detector_type=="yolov5":
        from detectors import yolov5_detector
        detector_object=yolov5_detector.Yolov5Detector
    elif detector_type=="yolov7":
        from detectors import yolov7_detector
        detector_object=yolov7_detector.Yolov7Detector

    #Tracker object MOT/SOT
    if tracker_type=="SOT":
        tracker_object=mmtracking_sot.SotaTracker
        perceptor_object=sot_perceptor.SotPerceptor
    elif tracker_type=="MOT":
        tracker_object=mmtracking_mot.MotTracker
        perceptor_object=mot_perceptor.MotPerceptor

    #Pose Estimation object
    if keypoints_name:
        if pose_cfg["3D"]:
            pose_est_object=keypoints3D.Keypoints3D
        else:
            pose_est_object=keypoints2D.Keypoints2D

    else:
        pose_est_object=None

    perceptor=perceptor_object(width = 640, height = 480, channels = 3, downscale = 1,
                                detector=detector_object, detector_cfg=detector_cfg, 
                                tracker=tracker_object, tracker_cfg=tracker_cfg,
                                keypoints=pose_est_object, keypoints_cfg=pose_cfg,
                                type_input = "opencv", verbose=verbose, device=device)


    # perceptor=perceptor_object(width = 640, height = 480, channels = 3, downscale = 1,
    #                             detector = detector_object, detector_size=detector_size, detector_thresh=detector_thresh, 
    #                             tracker=tracker_object, tracker_model=tracker_name, tracking_conf=tracking_conf,
    #                             path_cfg_tracker=path_cfg_tracker, path_weights_tracker=path_weights_tracker,
    #                             keypoints=pose_est_object, keypoints3D_activ= keypoints3D_activ, keypoints3D_model= keypoints3D_name, path_output_3D=path_output_3D,
    #                             type_input = "opencv", verbose=verbose, device=device)


    # if tracker_type == "Yolov7":
    #     from detectors import yolov7_detector
    #     perceptor = sot_perceptor.SotPerceptor(width = 640, height = 480, channels = 3, downscale = 1,
    #                                             detector = yolov7_detector.Yolov7Detector, detector_size=detector_model, 
    #                                             tracker=None, tracker_model=None, tracking_conf=None,
    #                                             type_input = "opencv", verbose=verbose)
    # elif tracker_type =="Stark":
    #     from detectors import yolov7_detector
    #     perceptor = sot_perceptor.SotPerceptor(width = 640, height = 480, channels = 3, downscale = 1,
    #                                             detector = yolov7_detector.Yolov7Detector, detector_size=detector_model, 
    #                                             tracker=mmtracking_sot.SotaTracker, tracker_model="Stark", tracking_conf=tracking_conf,
    #                                             type_input = "opencv", verbose=verbose)
    # elif tracker_type =="ByteTrack":
    #     from detectors import yolov5_detector
    #     perceptor = mot_perceptor.MotPerceptor(width = 640, height = 480, channels = 3, downscale = 1,
    #                                             detector = yolov5_detector.Yolov5Detector, detector_size=detector_model, 
    #                                             tracker=mmtracking_mot.MotTracker, tracker_model="ByteTrack", tracking_conf=tracking_conf,
    #                                             type_input = "opencv", verbose=verbose)
    
    # else:
    #     print(f"tracker type {tracker_type} is not implemented.")


    ##################################
    # Load Input Source/Video Writer #
    ##################################
    # start video stream
    #df_gt=Utils.load_groundtruth(path_groundtruth)
    if verbose:
        print(f"Using: {source} as input")
    grab = FrameGrab(mode="video", path=source)

    if path_output_vid:
        # get vcap property 
        width  = int(grab.cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
        height = int(grab.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
        fps = int(grab.cap.get(cv2.CAP_PROP_FPS))
        if verbose:
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
        if verbose >= 2:
            print(f"Elapsed time for whole fordward pass: {(toc-tic)*1e3:.1f}ms")
        #record bbox and elapsed time
        bboxes_to_save.append(bbox_list)
        elapsed_time_list.append((toc-tic)*1e3)

        # VISUALIZATION
        if hasattr(perceptor,"img_kpts"):
            cv2.imshow("Camera Loomo", perceptor.img_kpts)
            cv2.waitKey(1)
        else:
            Utils.visualization(img, bbox_list, (255, 0, 0), 2)
        if gt is not None:
            truth=df_gt[frame_number]
            truth=Utils.bbox_x1y1wh_to_xcentycentwh(truth)
            Utils.visualization(img, truth, (0, 255, 0), 1)

        if verbose:
            print("final bbox:", bbox_list)

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