# **PostureTrack**


Associated Paper: [](./others/paper.pdf)

<p float="center">
  <img src="./others/GIF1.gif" width="400" />
  <img src="./others/GIF2.gif" width="400" /> 
</p>



## Intro
---
This repository shows the implementation of a highly modular Video Perception Toolbox. It provides capabilities for the following tasks:
* Single Person Tracking (SOT) 
* Multiple Person Tracking (MOT)
* Top-Down Pose Tracking
* Bottom-Up Pose Tracking

### Major Features
* Real-Time:

    This algorithm was designed to be implemented for real-time applications. We tested it on [Loomo](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjlq5vL9pP6AhUThP0HHavnB1EQFnoECAgQAQ&url=https%3A%2F%2Fstore.segway.com%2Fsegway-loomo-mini-transporter-robot-sidekick&usg=AOvVaw1cqNMjnrmQMHPheATHr6sN) Segway Robot.

* Modularity

    We decompose the video perception framework into different components and one can easily construct a customized multi-stage method by combining different modules.
    
* Benchmarking

   this Repository can also serve as a benchmark to test and compare the performance of different perception algorithms on Single Person Tracking Videos/Img Sequences.

## Methods Implemented
---

### Detection

- [x] [Yolov5](https://github.com/ultralytics/yolov5)
- [x] [Yolov7](https://github.com/WongKinYiu/yolov7)
### Single Object Tracking

- [x] [Stark](https://github.com/open-mmlab/mmtracking/tree/master/configs/sot/stark)
- [x] [SiameseRPN++](https://github.com/open-mmlab/mmtracking/tree/master/configs/sot/siamese_rpn)

### Multiple Object Tracking

- [x] [ByteTrack](https://github.com/open-mmlab/mmtracking/tree/master/configs/mot/bytetrack)

### Pose Estimation
- [x] [DeepPose](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#deeppose-cvpr-2014)
- [x] [HrNet](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#hrnet-cvpr-2019) 
- [ ] [OpenPifPaf](https://arxiv.org/abs/2103.02440)

### 3D Pose Estimation
- [ ] [VideoPose3D](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#videopose3d-cvpr-2019)
### Bottom-Up Pose Tracking
- [x] [OpenPifPaf Tracker](https://arxiv.org/abs/2103.02440)

Every component of the perception pipeline can be easily interchanged using the config files. See [config.md](./src/PostureTrack/config/README.md) for more information.


## Prerequisites
 - Clone the repository: `git clone git@github.com:theoh-io/PostureTrack.git`

  
- Look at the Install Procedure Described in [Install.md](./Install.md)
- Requirements:
    - Python3
    - OpenCV
    - Pytorch
    - torchvision
    - Openpifpaf
    - mmcv-full

## Repo structure

    │─── Benchmark ───> Videos/Img sequences + groundtruth bbox
    │ 
    │─── libs ───> Folder for the installation of external libraries
    │ 
    │─── src/PostureTrack ───> Python files containing perception toolbox
    │       │─── config ───> Folder containing the config files to build customized method
    │       │─── detectors ───> Classes for implemented detectors
    │       │─── trackers ───> Classes for implemented trackers SOT/MOT
    │       │─── keypoints ───> Classes for implemented pose estimators 2D/3D
    │       │─── perceptors ───> High level class combining every module for Top-Down solutions
    │       │─── outputs ───> Folder for the results of the inference
    │       │─── run.py: file to run inference on benchmark files
    │       │─── loomo.py: file to connect and run the program on Loomo
    ├── others         # Other materials / Android app APK
    └── README.md

Check [config.md](./src/PostureTrack/config/README.md) for more information on how to build a customized method.

## Running the Pipeline on Video/Img sequence
---

### Donwload Benchmark Data
Check [Dataset.md](./Dataset.md) for more information on how to download benchmarking data.

### Inference on Benchmark Data
Launch the `run.py` script (src/PostureTrack/) to try the default perception module configuration on default benchmark.

To get more details on how to change the configurations and input file check [config.md](./src/PostureTrack/config/README.md)

---
## Running the Pipeline on Loomo
### Connection with Loomo
Make sure to be connected to the same WiFi, get the ip adress of Loomo from settings. Then connect using `adb connect <ip Loomo>`  and check connection is working using  `adb devices`.

Run the AlgoApp on Loomo and press on button **Start Algo** to allow for socket connection, you should now see the camera on Loomo's screen.

Before trying to launch the app on loomo make sure to have the same downscale parameter on Loomo and in the config file: `config/loomo/loomo_cfg.yaml`. To see the config on loomo use the command: `adb shell cat /sdcard/follow.cfg`

### QuickStart with Loomo
Easiest way to start is to change the ip-adress of loomo in the config file (`config/loomo/loomo_cfg.yaml`)  and launch the script to run the algorithm on loomo
`python3 src/PostureTrack/loomo.py`

---


---
## Other resources
### Introductory documents

Please find the following documents for an introduction to the Loomo robot and a socket protocol.

* [Getting_Started_with_Loomo.pdf](./others/Getting_Started_with_Loomo.pdf)
* [Environment_Setup_Robots.pdf](./others/Environment_Setup_Robots.pdf)
* [Loomo_Deployment_Instruction.pdf](./others/Loomo_Deployment_Instruction.pdf)
* [Tutorial_Loomo_Race.pdf](./others/Tutorial_Loomo_Race.pdf)

* [Loomo with Ros](https://github.com/cconejob/Autonomous_driving_pipeline)
* [Loomo Follower App](https://github.com/segway-robotics/loomo-algodev/blob/master/algo_app/src/main/jni/app_follow/AlgoFollow.cpp)

---
## To add
- ReadMe config folder
- Yolov5 in libs so that we can load without torch hub and specify GPU in map location (currently yolov5 automatically loaded on cuda:0)
- Keypoints for top down inference
- 3D keypoints
- Android App APK
- link from old Perception Pipeline
- Automatic download yolov7 weights
- Multiple GPUs support
- Modify android app to take infrared data into account and emergency brake
- Long Term Tracking by Integrating ReID.

---
### Acknowledgements
We want to thanks OpenMMLab for the provided utils and model-zoo.

    @misc{mmtrack2020,
        title={{MMTracking: OpenMMLab} video perception toolbox and benchmark},
        author={MMTracking Contributors},
        howpublished = {\url{https://github.com/open-mmlab/mmtracking}},
        year={2020}
    }
    @misc{mmpose2020,
        title={OpenMMLab Pose Estimation Toolbox and Benchmark},
        author={MMPose Contributors},
        howpublished = {\url{https://github.com/open-mmlab/mmpose}},
        year={2020}
    }



