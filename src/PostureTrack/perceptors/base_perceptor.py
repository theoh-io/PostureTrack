#from detectors.yolov7_detector import Yolov7Detector
from detectors.yolov5_detector import Yolov5Detector
#from trackers.mmtracking_sot import SotaTracker


class BasePerceptor():

    def __init__(self, width=640, height=480, channels=3, downscale=1, 
                detector=Yolov5Detector, detector_cfg=None, 
                tracker=None, tracker_cfg=None,
                keypoints=None, keypoints_cfg=None,
                type_input = "opencv", verbose=1, device=0):

        # detector=Yolov5Detector, detector_size="default", detector_thresh=0.1,
        # tracker=None, tracker_model=None, tracking_conf=0.5,
        # path_cfg_tracker="", path_weights_tracker="",
        # keypoints=False, keypoints3D_activ=False, keypoints3D_model=None, path_output_3D=None,
        # type_input="opencv", verbose=1, device=0):
        # perceptor expected input image dimensions
        self.width = int(width/downscale)
        self.height = int(height/downscale)
        self.downscale = downscale
        self.verbose=verbose
        self.device=device

        # Image received size data.
        self.data_size = int(self.width * self.height * channels)

        self.detector=detector(model=detector_cfg["size"], thresh_bbox=detector_cfg["thresh"], device=self.device, verbose=self.verbose)
        if tracker:
            self.tracker=tracker(tracker_cfg["name"], tracker_cfg["conf"], tracker_cfg["cfg"], tracker_cfg["weights"],device = self.device, verbose=self.verbose)
        else:
            self.tracker=None
        self.type_input=type_input

        #Pose estimation
        
        if keypoints:
            self.keypoints=keypoints(keypoints_cfg["weights"], keypoints_cfg["config"],
                    keypoints_cfg["3D"], keypoints_cfg["weights3D"], keypoints_cfg["config3D"],
                    self.device, (self.width, self.height), show3D=True, 
                    save_video_keypoints=keypoints_cfg["path_output_3D"], smooth=False)
            self.keypoints3D_activ=bool(keypoints_cfg["3D"])
            self.keypoints3D=keypoints_cfg["3Dname"]
        else:
            self.keypoints=None
            self.keypoints3D_activ=False
            self.keypoints3D=None


        if self.verbose:
            print("Initializing Perceptor")
            print(f"-> Input image of type {self.type_input} and downscale {self.downscale}")


    def forward(self, image):
        raise NotImplementedError("perceptor Base Class does not provide a forward method.")


