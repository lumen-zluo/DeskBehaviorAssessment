# Desk Behavior Assessment platform

This template should help get you collecting [eyetracker](https://docs.pupil-labs.com/core/developer/), [realsense camera](https://github.com/IntelRealSense/librealsense) data.

### Enviroment

+ python >= 3.9

```
conda create --name DeskBehavior python=3.9
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

after installation, you can run `python ./scripts/test_cuda.py` to check if you install all the staff above successfully

### Submodule 

```
git submodule init		# 初始化子模块
git submodule update	# 更新子模块
```

### Project Setup

```sh
pip install ultralytics
pip install pyinstaller 
```

### Install MMAction2

```
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet
mim install mmpose

cd mmaction2
pip install -v -e .
# "-v" 表示输出更多安装相关的信息
# "-e" 表示以可编辑形式安装，这样可以在不重新安装的情况下，让本地修改直接生效。
```

### Pack exe

```
pyinstaller .\main.spec
```

### COCO Pose Keypoints index [[Reference](https://mmpose.readthedocs.io/zh-cn/latest/dataset_zoo/2d_body_keypoint.html)]

![https://user-images.githubusercontent.com/100993824/227864552-489d03de-e1b8-4ca2-8ac1-80dd99826cb7.png](https://user-images.githubusercontent.com/100993824/227864552-489d03de-e1b8-4ca2-8ac1-80dd99826cb7.png)
