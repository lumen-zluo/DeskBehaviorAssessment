# Desk Behavior Assessment System

This template should help get you collecting [eyetracker](https://docs.pupil-labs.com/core/developer/), [realsense camera](https://github.com/IntelRealSense/librealsense) data.

## Enviroment

+ python >= 3.9

```
conda create --name DeskBehaviorAssessment python=3.9
```

### Install CUDA

use ` nvidia-smi` to check which CUDA version you need to download:

https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html

### Install cuDNN

https://developer.nvidia.com/rdp/cudnn-archive

### Install Pytorh

you can find your version according to this [reference](https://pytorch.org/), eg:

```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

after installation, you can run `python test_cuda.py` to check if you install all the staff above successfully

### Install MMAction2

https://mmaction2.readthedocs.io/zh-cn/latest/get_started/installation.html

## Project Setup

```sh
pip install ultralytics
pip install msgpack pyzmq pyqt5 pyrealsense2
```

### Compile Pyqt5

```sh
 pyuic5 -o .\main.py .\main.ui
```

### COCO Pose Keypoints index [[Reference](https://mmpose.readthedocs.io/zh-cn/latest/dataset_zoo/2d_body_keypoint.html)]

![https://user-images.githubusercontent.com/100993824/227864552-489d03de-e1b8-4ca2-8ac1-80dd99826cb7.png](https://user-images.githubusercontent.com/100993824/227864552-489d03de-e1b8-4ca2-8ac1-80dd99826cb7.png)

正常小孩：26，27，29，31，38(ASD)，40，45，46，48，58