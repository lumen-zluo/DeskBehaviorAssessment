import random

import numpy as np
import torch

from ultralytics import YOLO


# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # setup_seed(2024)
    # # Load a model
    # model = YOLO('yolov8n.yaml')  # build a new model from YAML
    # model = YOLO('yolov8n-lixiao.yaml')  # build a new model from YAML
    # model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
    model = YOLO('yolov8s.yaml').load('yolov8s.pt')  # build a new model from YAML
    # Train the model
    # model.train(data='coco128.yaml', epochs=100, imgsz=640)

    # 使用“.yaml”数据集对模型进行3个时期的训练
    # dataset_path = os.path.join("..", "ori_datasets", "dataset_internet_depart_v1", "datasets.yaml")
    # results = model.train(data='datasets_v2_random.yaml', epochs=1000, batch=8, patience=50, lr0=0.00001, optimizer='SGD')
    results = model.train(data='datasets_v2_random.yaml', epochs=1000, batch=1, patience=300)

    # results = model.train(data='datasets.yaml', epochs=2000, batch=4, patience=50, device=['0', '1', '2', '3'])

    # 评估模型在验证集上的性能
    # results = model.val()
