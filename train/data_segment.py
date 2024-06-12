import os
import random
import shutil
import warnings

import yaml
from tqdm import tqdm


class Dataset:
    images_path: list[str] = []
    labels_path: list[str] = []

    file_sum: int

    train_file_num: int
    val_file_num: int
    test_file_num: int

    def __init__(self, _image_folder, _label_folder, _rate):

        self.image_folder = _image_folder
        self.label_folder = _label_folder

        self.images_list = os.listdir(self.image_folder)
        for image_name in self.images_list:
            image_path = os.path.join(self.image_folder, image_name)
            name = image_name.split('.')[0]
            label_path = os.path.join(self.label_folder, name + '.txt')
            if os.path.exists(label_path):
                self.images_path.append(image_path)
                self.labels_path.append(label_path)
            else:
                warnings.warn(f'{label_path} 不存在')

        self.file_sum = len(self.images_path)
        rate_sum = rate[0] + rate[1] + rate[2]
        self.train_file_num = (rate[0] * self.file_sum) // rate_sum
        self.val_file_num = (rate[1] * self.file_sum) // rate_sum
        self.test_file_num = self.file_sum - self.train_file_num - self.val_file_num

    def depart_datasets_v1(self, _shuffle: bool = False):

        self.output_folder = os.path.join(os.path.split(self.image_folder)[0]+'_depart_v2_random')
        self.train_images_folder = os.path.join(self.output_folder, 'train', 'images')
        self.train_labels_folder = os.path.join(self.output_folder, 'train', 'labels')
        self.val_images_folder = os.path.join(self.output_folder, 'val', 'images')
        self.val_labels_folder = os.path.join(self.output_folder, 'val', 'labels')
        self.test_images_folder = os.path.join(self.output_folder, 'test', 'images')
        self.test_labels_folder = os.path.join(self.output_folder, 'test', 'labels')
        os.makedirs(self.train_images_folder, exist_ok=True)
        os.makedirs(self.train_labels_folder, exist_ok=True)
        os.makedirs(self.val_images_folder, exist_ok=True)
        os.makedirs(self.val_labels_folder, exist_ok=True)
        os.makedirs(self.test_images_folder, exist_ok=True)
        os.makedirs(self.test_labels_folder, exist_ok=True)

        fun = list(range(self.file_sum))
        if _shuffle:
            # 名字打乱
            # 创建一个index数组，打乱这个数组，然后同时打乱但对应数据集和标签
            random.shuffle(fun)

        # 提取train的名字，找到图片，找到label
        for i in tqdm(range(self.file_sum)):
            index = fun[i]
            image_path = self.images_path[index]
            label_path = self.labels_path[index]

            image_name = os.path.split(image_path)[-1]
            label_name = os.path.split(label_path)[-1]
            if i < self.train_file_num:
                image_target_path = os.path.join(self.train_images_folder, image_name)
                label_target_path = os.path.join(self.train_labels_folder, label_name)
            elif i < self.train_file_num + self.val_file_num:
                image_target_path = os.path.join(self.val_images_folder, image_name)
                label_target_path = os.path.join(self.val_labels_folder, label_name)
            else:
                image_target_path = os.path.join(self.test_images_folder, image_name)
                label_target_path = os.path.join(self.test_labels_folder, label_name)
            a=1
            # copy文件
            try:
                shutil.copy(image_path, image_target_path)
                shutil.copy(label_path , label_target_path)
            except FileNotFoundError:
                pass

    # 创建yaml文件
    def build_detect_yaml(self, _namelist):
        self.yaml_file_path = 'datasets_v2_random.yaml'
        local_content = {
            "train": self.train_images_folder,
            "val": self.val_images_folder,
            "nc": len(_namelist),
            "names": _namelist,
        }

        with open(self.yaml_file_path, 'w', encoding='utf-8') as file:
            yaml.dump(local_content, file)

    # 创建yaml文件
    def build_pose_yaml(self, _nameList, _nc):

        local_content = {
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "kpt_shape": [17, 3],  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
            "names": {0: "person"}
        }

        with open(self.yaml_file_path, 'w', encoding='utf-8') as file:
            yaml.dump(local_content, file)


if __name__ == "__main__":
    # 图片文件夹路径
    images_folder = r"C:\Users\Owner\Downloads\dataset\imgs"
    # label文件夹路径
    labels_folder = r"C:\Users\Owner\Downloads\dataset\yolo"
    # 分的比例
    rate = (8, 1, 1)

    myDataset = Dataset(images_folder, labels_folder, rate)
    myDataset.depart_datasets_v1(_shuffle=True)
    myDataset.build_detect_yaml(['ruler', 'worksheet', 'eraser', 'pen'])
