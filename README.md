# **PostureTrack**


Associated Paper: [](./others/paper.pdf)

<p float="center">
  <img src="./others/GIF1.gif" width="400" />
  <img src="./others/GIF2.gif" width="400" /> 
</p>


---
## Intro
This repository shows the implementation of a robust and modular Perception-Pipeline. It provides Single Person Tracking (SOT) and Multiple Person Tracking (MOT) state-of-the-art capabilities. This Perception module was developed to be implemented for real-time on Loomo Segway Robot. Furthermore, this Repository can also serve as a benchmark to test the performance of different perception algorithms on Single Person Tracking Videos/Img Sequences.

This pipeline propose a modular implementation combining **Yolov5**, **Yolov7** and **OpenPifPaf** for the Detection module. **ByteTrack** (MOT), **SiamRPN++** and **Stark** (SOT) for the Tracking module. Every component of the perception pipeline can be easily interchanged using the config files.

Furthermore Bottom-up keypoints tracking method using **OpenPifPaf Tracker** is also implemented. 


## Prerequisites
 - Clone the repository: `git clone git@github.com:theoh-io/PostureTrack.git`

  
- Look at the Install Procedure Described at the bottom
- Requirements:
    - Python3
    - OpenCV
    - Pytorch
    - torchvision
    - Openpifpaf
    - mmcv-full

## Repo structure
    ├── Benchmark      # Videos/Img sequences + groundtruth bbox
    ├── libs     # Folder for the installation of external libraries (mmtracking, deep-person-reid)
    ├── python         # Python files containing perception module
    ├── Results        # Folder to store the videos and bbox resulting from inference
    ├── others         # Other materials / Android app APK
    └── README.md

Check the other ReadMe file (add link) to get more info about the perception module and configurations

## Downloading pretrained models

### Detection
[Yolov5](https://github.com/ultralytics/yolov5):
    Automatic Download implemented

[Yolov7](https://github.com/WongKinYiu/yolov7): Trainable bag-of-freebies, State-of-the-art object detector
    cd src/PostureTrack/detectors/weights
    wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
    wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
    wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt

### Tracking
#### SOT
[Stark](https://github.com/open-mmlab/mmtracking/tree/master/configs/sot/stark): Learning Spatio-Temporal Transformer for Visual Tracking

    cd src/PostureTrack/trackers/weights
    wget https://download.openmmlab.com/mmtracking/sot/stark/stark_st2_r50_50e_lasot/stark_st2_r50_50e_lasot_20220416_170201-b1484149.pth

[SiameseRPN++](https://github.com/open-mmlab/mmtracking/tree/master/configs/sot/siamese_rpn):  Evolution of Siamese Visual Tracking With Very Deep Networks

    wget https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_lasot/siamese_rpn_r50_20e_lasot_20220420_181845-dd0f151e.pth
#### MOT
[ByteTrack](https://github.com/open-mmlab/mmtracking/tree/master/configs/mot/bytetrack)

    wget https://download.openmmlab.com/mmtracking/mot/bytetrack/bytetrack_yolox_x/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth

### Pose 2D
[DeepPose]()

    cd src/PostureTrack/keypoints/weights
    wget https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_coco_384x288_rle-b77c4c37_20220624.pth

[HrNet]()

    wget https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth


### Pose 3D

[VideoPose Human3.6M]()

    cd src/PostureTrack/keypoints/weights
    wget "https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth"


---
## Running the Pipeline on Video/Img sequence

### Downloading Benchmark Data
To download data to the Benchmark folder use the command:  `wget <link_data>`  in the Benchmark folder.
- [**LaSOT**](http://vision.cs.stonybrook.edu/~lasot/): Large scale dataset for Single Image Tracking (SOT) as our detection algorithm is only trained on humans we are only interested on the person category available for download [here](https://drive.google.com/drive/folders/1v09JELSXM_v7u3dF7akuqqkVG8T1EK2_?usp=sharing) 
- [**OTB100**](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html): Img sequences from various different sources with provided categories for tracking challenges (Illumination variations, Body deformation...). Only some sequences are relevant for out perception module (Human, Skater, BlurBody, Basketball...)
- **Loomo Dataset** provides 8 + 16 Videos with provided ground truth from real-life experiments recordings. The Dataset is available for download at this [link](https://drive.google.com/drive/folders/1r9-GRIsfojvwlljnHovZ5SvlmbUzsZtZ?usp=sharing)

### Inference on Benchmark Data
Launch the `run.py` script (src/PostureTrack/) to try the default perception module configuration

(To get more details on how to change the configurations and input file check the ReadMe.md inside config directory)

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
## Install

### Virtual environment Setup
First set up a python >= 3.7 virtual environment:

    cd <desired-dir>
    python3 -m venv <desired venv name>
    source <name_venv>/bin/activate

Make sure to install python 3 in case you do not have it.

### Dependencies Install

In the root directory: after creating a virtual environment run the following commands to install pip packages
 `pip install -r requirements.txt`

--> If you encounter problem at this step try to run each line of the requirements individually with pip install ... and not specifying the version of the package.

install packages from source in libs folder
    pip install openmim
    mim install mmdet
    mim install mmcv-full
    mmpose: (official instructions: https://mmpose.readthedocs.io/en/v0.28.0/install.html)
        clone in libs
        cd mmpose
        pip install -r requirements.txt
        pip install -v -e .
    mmtracking
    Follow the install instructions from the [official documentation](https://mmtracking.readthedocs.io/en/latest/install.html)
        git clone https://github.com/open-mmlab/mmtracking.git
        cd mmtracking
        pip install -r requirements/build.txt
        pip install -v -e .  # or "python setup.py develop"
        
    yolov7: https://github.com/WongKinYiu/yolov7
        pip install seaborn thop
        cd yolov7
        pip install -r requirements

(Optional) In case you work with specific GPU and CUDA version
-   More info on specific torch and torchvision versions:

    https://pypi.org/project/torchvision/ (compatibility list)
    https://download.pytorch.org/whl/torch_stable.html (torch)
    http://download.pytorch.org/whl/torchvision/ (torchvision)



In the `Perception-Pipeline/python/` directory run the following command to install our custom perceptionloomo package `pip install -e .`

### OpenCV install from source (Optional)

In case you have problems with the installation of opencv using pip you can try to build it directly from source using the following procedure.

    mkdir depencency 
    cd dependency
    wget https://github.com/opencv/opencv/archive/3.4.5.zip
    unzip 3.4.5.zip
    mkdir opencv 
    cd opencv-3.4.5
    mkdir build
    cd build
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=../../opencv -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_V4L=ON -DWITH_OPENGL=ON -DWITH_CUBLAS=ON -DCUDA_NVCC_FLAGS="-D_FORCE_INLINES" ..
    make -j4
    make install

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
We want to thanks OpenMMLab for the provided utils and model-zoo

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



