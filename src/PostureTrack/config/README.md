# Working with Config Files

## Intro

Our goal is to implement a highly modular video perception toolbox. Our architecture is multi-stage meaning that we can easily switch any stage of the algorithm without modifying the rest. We provide pre-made combinations of methods as basic configurations and allow the user to create new ones.

## Quick Start
### Default Configuration

The Default configuration implements Single Object Tracking (SOT) so that it can be easily tested on Loomo with the Follow App. First Stage is the Yolov5 Detector returning every person Detection and then on top of that we use STARK for SOT.

If you want to use this toolbox with Loomo please change the IP first in loomo.cfg


## Creating Custom Configurations

### Structure of the Config

    config
    │─── arch ───> Folder containing the specified architecture configs
    │       │─── provided configs ───> bottom up / only detection/ top down tracking + pose estimation / top down tracking
    │ 
    │─── detector ───> Folder containing the specified config for the detectors
    │       │─── provided configs ───> yolov5/yolov7
    │ 
    │─── io ───> Folder with configs specifying: input (when using benchmark), Output destination, ground_truth, device to use and verbose level
    │ 
    │─── keypoints ───> Folder with configs specifying: 2D and 3D Pose estimation model
    │       │─── provided configs ───> 2D Pose estimation: Hrnet, DeepPose / 3D Pose estimation: VideoPose3D
    │ 
    │─── loomo ───> Folder with config to connect to loomo robot
    │ 
    │─── tracker ───> 
    │       │─── provided configs ───> stark/siamese (SOT), ByteTrack (MOT), pifpaf_tracker (Bottom-up Pose Tracking)
    │ 
    ├── cfg.yaml ───> High Level config File to compose custom  configuration
    └── README.md

### Composing with existing configurations

To compose a new combination of configurations, the user should only modify the top-level [config file](./cfg.yaml) and specify the configuration he wants to use at each stage.

### Loomo Config

In order to be able to run the algorithms on Loomo its really important to have set the right IP in [loomo.cfg](./loomo/loomo_cfg.yaml). Furthermore the downscale parameter should match the one in the robot sdcard. To check it use: ` adb shell cat sdcard/follow.cfg`