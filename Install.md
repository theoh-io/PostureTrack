## Install

:warning: to be able to run the program we must be sure to have installed all the dependencies and at least have downloaded the weights for the method in the used multi-stage method.

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


In the `Perception-Pipeline/python/` directory run the following command to install our custom perceptionloomo package ` pip install -e .`

(Optional) In case you work with specific GPU and CUDA version
-   More info on specific torch and torchvision versions:

    https://pypi.org/project/torchvision/ (compatibility list)
    https://download.pytorch.org/whl/torch_stable.html (torch)
    http://download.pytorch.org/whl/torchvision/ (torchvision)

## Downloading pretrained models
---

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
[DeepPose](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#deeppose-cvpr-2014)

    cd src/PostureTrack/keypoints/weights
    wget https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_coco_384x288_rle-b77c4c37_20220624.pth

[HrNet](https://mmpose.readthedocs.io/en/latest/papers/backbones.html#hrnet-cvpr-2019)

    wget https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth


### Pose 3D

[VideoPose3D](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#videopose3d-cvpr-2019)

    cd src/PostureTrack/keypoints/weights
    wget "https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth"


---
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