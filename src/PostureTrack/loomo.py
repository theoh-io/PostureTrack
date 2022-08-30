# VITA, EPFL 
# from xmlrpc.client import boolean
# import cv2
# import socket
# import sys
# import numpy as np
# import struct
# import time
# import torch
# import argparse
# from PIL import Image
# import os

# from perceptionloomo.detectors.pose_detectors import PoseColorGuidedDetector
# from perceptionloomo.perception.perception import DetectorG16
# from perceptionloomo.utils.utils import Utils

# if str(os.getcwd())[-7:] == "scripts":
#     os.chdir("..")


#new imports
import os
import sys
import socket
import time
from PIL import Image
import cv2
import numpy as np
import torch
from perceptors import sot_perceptor, mot_perceptor
from trackers import mmtracking_sot, mmtracking_mot
from utilities import Utils
import struct
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="cfg")
def get_config(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    #loading IO cfg
    ip=cfg.loomo.ip
    downscale=cfg.loomo.downscale
    path_output_vid=cfg.loomo.path_video
    verbose=cfg.loomo.verbose
    loomo_cfg={'ip':ip, 'downscale': downscale, 'path_output_vid': path_output_vid, 'verbose': verbose}
    if cfg.arch.arch_type=="topdown":
        print("top-down architecture")
        if cfg.tracker.tracker_type=="SOT":
            print("using SOT Tracker")           
        elif cfg.tracker.tracker_type=="MOT":
            print("using MOT Tracker")
        detector_name=cfg.detector.detector_name
        detector_size=cfg.detector.detector_size
        detector_thresh=cfg.detector.bbox_thresh
        detector_cfg={"type": detector_name , "size": detector_size,    
                    "thresh": detector_thresh}
        tracker_name= cfg.tracker.tracker_name
        tracker_type=cfg.tracker.tracker_type
        tracking_conf= cfg.tracker.tracking_conf
        path_cfg= cfg.tracker.path_cfg
        path_weights= cfg.tracker.path_weights
        tracker_cfg={"name":tracker_name, "type":tracker_type ,"conf":tracking_conf,
                    "cfg":path_cfg, "weights":path_weights}
        Loomo(loomo_cfg, detector_cfg, tracker_cfg)
    else:
        print("error bottom_up not impleented yet")
        # TopDown(detector_cfg, tracker_cfg, verbose,
        #         source, gt, path_output_vid, path_output_json)
    # elif cfg.arch.arch_type=="bottomup":
    #     print("bottom-up architecture")
    #     if cfg.arch.tracker=="pifpaf":
    #         print("using pifpaf Tracker")
    #         checkpoint=cfg.tracker.checkpoint
    #         decoder=cfg.tracker.decoder
    #         long_edge=cfg.tracker.long_edge
    #         show_result=cfg.tracker.show_result
    #         disable_cuda=cfg.tracker.disable_cuda
    #         PifPaf(checkpoint, decoder, long_edge, show_result, disable_cuda,
    #                 source, path_output_vid, path_output_json)

def Loomo(loomo_cfg, detector_cfg, tracker_cfg):
    ###################################
    # Load ConfigFile Arguments
    ####################################
    ip=loomo_cfg["ip"]
    downscale=loomo_cfg["downscale"]
    path_output_vid=loomo_cfg["path_output_vid"]
    verbose=loomo_cfg["verbose"]
    detector_type=detector_cfg["type"]
    detector_size=detector_cfg["size"]
    detector_thresh=detector_cfg["thresh"]
    tracker_name=tracker_cfg["name"]
    tracker_type= tracker_cfg["type"]
    tracking_conf= tracker_cfg["conf"]
    path_cfg_tracker= tracker_cfg["cfg"]
    path_weights_tracker= tracker_cfg["weights"]
    print(f"tracker_type {tracker_type}")

    ####################################
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

    perceptor=perceptor_object(width = 640, height = 480, channels = 3, downscale = downscale,
                            detector = detector_object, detector_size=detector_size, 
                            tracker=tracker_object, tracker_model=tracker_name, tracking_conf=tracking_conf,
                            path_cfg_tracker=path_cfg_tracker, path_weights_tracker=path_weights_tracker,
                            type_input = "opencv", verbose=verbose)

    ##################################
    # Connect to Loomo/Video Writer #
    ##################################
    host = ip 
    port = 8081        # The port used by the server
    # create socket  FIX can be added to utils
    print('# Creating socket')
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error:
        print('Failed to create socket')
        sys.exit()
    print('# Getting remote IP address') 
    try:
        remote_ip = socket.gethostbyname( host )
    except socket.gaierror:
        print('Hostname could not be resolved. Exiting')
        sys.exit()

    # Connect to remote server
    print('# Connecting to server, ' + host + ' (' + remote_ip + ')')
    s.connect((remote_ip , port))

    if path_output_vid:
        # get vcap property 
        width  = perceptor.width
        height = perceptor.height
        fps = 3.5
        print(f"width: {width}, height: {height}, fps: {fps}")
        output_vid = cv2.VideoWriter(path_output_vid, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

    #Image Receiver 
    net_recvd_length = 0
    recvd_image = b''
    print("cuda : ", torch.cuda.is_available())
    while True:
        # Receive data
        reply = s.recv(perceptor.data_size)
        recvd_image += reply
        net_recvd_length += len(reply)
        if net_recvd_length == perceptor.data_size:
            pil_image = Image.frombytes('RGB', (width, height), recvd_image)
            opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            opencvImage = cv2.cvtColor(opencvImage,cv2.COLOR_BGR2RGB)

            net_recvd_length = 0
            recvd_image = b''

            # INFERENCE
            bbox=perceptor.forward(opencvImage)

            # VISUALIZATION
            #Video recording adding the bounding boxes
            if path_output_vid is not None:
                output_vid.write(opencvImage)
                
            Utils.visualization(opencvImage, bbox, (255, 0, 0), 2)

            # TRANSMISSION
            # https://pymotw.com/3/socket/binary.html
            if bbox is None:
                bbox=[0,0,0,0]
                bbox_label=False
            else:
                bbox_label=True

            values = (bbox[0], bbox[1], bbox[2], bbox[3], float(bbox_label))
            packer = struct.Struct('f f f f f')
            packed_data = packer.pack(*values)
            # Send data
            send_info = s.send(packed_data)

            k=cv2.waitKey(10) & 0xFF
            if k == ord('q'):
                break


if __name__ == "__main__":
    get_config()












# detector=DetectorG16()
# #General Configs
# print(detector.cfg)
# verbose=detector.cfg.PERCEPTION.VERBOSE
# ip=detector.cfg.LOOMO.IP
# downscale=detector.cfg.LOOMO.DOWNSCALE
# rec= detector.cfg.LOOMO.RECORDING

# print("verbose :", verbose)
# print("value of downscale parameter :", downscale)
# print("ip adress of Loomo: ", ip)
# if rec is not None:
#     print("path for the recorded video :", rec)
# #Detector configs

# #Tracker configs
# print(detector.cfg.TRACKER.TRACKER_CLASS)

# ##### IP Address of server #########
# host = ip #local : 127.0.0.1  # The server's hostname or IP address
# ####################################
# port = 8081        # The port used by the server


# # create socket  FIXME can be added to utils
# print('# Creating socket')
# try:
#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# except socket.error:
#     print('Failed to create socket')
#     sys.exit()
# print('# Getting remote IP address') 
# try:
#     remote_ip = socket.gethostbyname( host )
# except socket.gaierror:
#     print('Hostname could not be resolved. Exiting')
#     sys.exit()

# # Connect to remote server
# print('# Connecting to server, ' + host + ' (' + remote_ip + ')')
# s.connect((remote_ip , port))


# # image data
# width = int(640/downscale)
# height = int(480/downscale)
# channels = 3
# sz_image = width*height*channels

# #Video writersize
# framerate=3.5 #3.3 for downscale 2 and 5 for downscale 4
# if rec is not None:
#     path_vid=rec
#     output_vid = cv2.VideoWriter(path_vid, cv2.VideoWriter_fourcc(*'MJPG'), framerate, (width, height)) #try without specifying width and height
    

# #function to Warn the user in case of wrong downscale factor
# def size_adjust():
#     if(mismatch==1 and time.time()>timeout):
#         print("Warning: Image Size Mismatch: ", sz_image, "  ",net_recvd_length)

# mismatch=1 #FSM for avoiding checking size once it has been verified one time
# timeout=time.time()+0.2 #variable used to avoid printing warning on size mismatch for initialization

# #Image Receiver 
# net_recvd_length = 0
# recvd_image = b''
# print("cuda : ", torch.cuda.is_available())
# while True:
#     # Receive data
#     reply = s.recv(sz_image)
#     recvd_image += reply
#     net_recvd_length += len(reply)
#     # if verbose is True:
#     #     print("Size info: ", sz_image, "  ",net_recvd_length)
#     if net_recvd_length == sz_image:
#         pil_image = Image.frombytes('RGB', (width, height), recvd_image)
#         opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
#         opencvImage = cv2.cvtColor(opencvImage,cv2.COLOR_BGR2RGB)

#         net_recvd_length = 0
#         recvd_image = b''
#         mismatch=0

#         bbox=detector.forward(opencvImage)

#         #######################
#         # Visualization
#         #######################
#         #Video recording adding the bounding boxes
#         if rec is not None:
#             output_vid.write(opencvImage)
            
#         Utils.visualization(opencvImage, bbox)

        



#         #######################
#         # Socket
#         #######################
#         # https://pymotw.com/3/socket/binary.html
#         if bbox is None:
#             bbox=[0,0,0,0]
#             bbox_label=False
#         else:
#             bbox_label=True

#         values = (bbox[0], bbox[1], bbox[2], bbox[3], float(bbox_label))
#         packer = struct.Struct('f f f f f')
#         packed_data = packer.pack(*values)
#         # Send data
#         send_info = s.send(packed_data)

#         k=cv2.waitKey(10) & 0xFF
#         if k == ord('q'):
#             break
#     else:
#         size_adjust()