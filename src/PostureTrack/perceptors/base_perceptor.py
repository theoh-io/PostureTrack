#from detectors.yolov7_detector import Yolov7Detector
from detectors.yolov5_detector import Yolov5Detector
#from trackers.mmtracking_sot import SotaTracker


class BasePerceptor():

    def __init__(self, width=640, height=480, channels=3, downscale=1, 
                    detector=Yolov5Detector, detector_size="default", detector_thresh=0.1,
                    tracker=None, tracker_model=None, tracking_conf=0.5,
                    path_cfg_tracker="", path_weights_tracker="",
                    keypoints=False, keypoints3D_activ=False, keypoints3D_model=None, path_output_3D=None,
                    type_input="opencv", verbose=1, device=0):
        # perceptor expected input image dimensions
        self.width = int(width/downscale)
        self.height = int(height/downscale)
        self.downscale = downscale
        self.verbose=verbose
        self.device=device

        # Image received size data.
        self.data_size = int(self.width * self.height * channels)

        self.detector=detector(model=detector_size, thresh_bbox=detector_thresh, device=self.device, verbose=self.verbose)
        if tracker:
            self.tracker=tracker(tracker_model, tracking_conf, path_cfg_tracker, path_weights_tracker,device = self.device, verbose=self.verbose)
        else:
            self.tracker=None
        self.type_input=type_input

        #Pose estimation
        self.keypoints_activ=bool(keypoints)
        self.keypoints3D_activ=bool(keypoints3D_activ)
        if self.keypoints_activ:
            self.keypoints=keypoints(self.device, (self.width, self.height), 
                    show3D=True, save_video_keypoints=path_output_3D, smooth=False)


        if self.verbose:
            print("Initializing Perceptor")
            print(f"-> Input image of type {self.type_input} and downscale {self.downscale}")


    def forward(self, image):
        raise NotImplementedError("perceptor Base Class does not provide a forward method.")


