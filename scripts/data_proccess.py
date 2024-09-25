import os
import shutil
import re
from random import sample


def extra_dirs():
    # 這是一個將所有剪輯過的視頻從文件夾取出放到上一級目錄的脚本
    for directory in directories:
        curr_dir = os.path.join(root_path, directory)
        if os.path.isdir(curr_dir):
            filenames = os.listdir(curr_dir)
            filename = filenames[0]
            source_file = os.path.join(curr_dir, filename)
            print(source_file)
            if os.path.isfile(source_file):
                destination_file = os.path.join(root_path, filename)
                shutil.move(source_file, destination_file)


def rename():
    # 這是一個將視頻文件重命名的脚本
    filenames = os.listdir(root_path)
    for filename in filenames:
        filepath = os.path.join(root_path, filename)
        if os.path.isfile(filepath) and filename.endswith('-1.mp4'):
            new_filename = re.sub(r'-1\.mp4$', '.mp4', filename)
            print(new_filename)
            new_filepath = os.path.join(root_path, new_filename)
            os.rename(filepath, new_filepath)


def generate_setting():
    # 這是一個生成訓練集和測試集劃分配置的脚本
    filenames = os.listdir(root_path)

    os.makedirs('./custom_dataset', exist_ok=True)
    class1 = []
    class2 = []
    class3 = []
    class4 = []
    class5 = []
    class6 = []
    for filename in filenames:
        data = filename.split('_')
        classname = data[1]
        if classname == 'class1':
            class1.append(filename)
        elif classname == 'class2':
            class2.append(filename)
        elif classname == 'class3':
            class3.append(filename)
        elif classname == 'class4':
            class4.append(filename)
        elif classname == 'class5':
            class5.append(filename)
        elif classname == 'class6':
            class6.append(filename)

    class1_val_cnt = len(class1) // 4
    class2_val_cnt = len(class2) // 4
    class3_val_cnt = len(class3) // 4
    class4_val_cnt = len(class4) // 4
    class5_val_cnt = len(class5) // 4
    class6_val_cnt = len(class6) // 4

    class1_val = sample(class1, class1_val_cnt)
    class2_val = sample(class2, class2_val_cnt)
    class3_val = sample(class3, class3_val_cnt)
    class4_val = sample(class4, class4_val_cnt)
    class5_val = sample(class5, class5_val_cnt)
    class6_val = sample(class6, class6_val_cnt)

    for item in class1_val:
        class1.remove(item)
    for item in class2_val:
        class2.remove(item)
    for item in class3_val:
        class3.remove(item)
    for item in class4_val:
        class4.remove(item)
    for item in class5_val:
        class5.remove(item)
    for item in class6_val:
        class6.remove(item)

    val_set = class1_val + class2_val + class3_val + class4_val + class5_val + class6_val
    train_set = class1 + class2 + class3 + class4 + class5 + class6

    print(len(val_set), len(train_set))

    with open('./custom_dataset/custom_dataset_train_video.txt', 'w') as f:
        for item in class1:
            f.write(f"{item} 0\n")
        for item in class2:
            f.write(f"{item} 1\n")
        for item in class3:
            f.write(f"{item} 2\n")
        for item in class4:
            f.write(f"{item} 3\n")
        for item in class5:
            f.write(f"{item} 4\n")
        for item in class6:
            f.write(f"{item} 5\n")

    with open('./custom_dataset/custom_dataset_val_video.txt', 'w') as f:
        for item in class1_val:
            f.write(f"{item} 0\n")
        for item in class2_val:
            f.write(f"{item} 1\n")
        for item in class3_val:
            f.write(f"{item} 2\n")
        for item in class4_val:
            f.write(f"{item} 3\n")
        for item in class5_val:
            f.write(f"{item} 4\n")
        for item in class6_val:
            f.write(f"{item} 5\n")


def move_video_files():
    # 這是一個將視頻文件分別按照配置移動到對應目錄的脚本
    train_file_name = './custom_dataset/custom_dataset_train_video.txt'
    val_file_name = './custom_dataset/custom_dataset_val_video.txt'
    os.makedirs('./custom_dataset/train', exist_ok=True)
    os.makedirs('./custom_dataset/val', exist_ok=True)

    with open(train_file_name, 'r') as file:
        for line in file:
            filename = line.split()[0]
            filepath = os.path.join(root_path, filename)
            shutil.copy2(filepath, './custom_dataset/train/')
    with open(val_file_name, 'r') as file:
        for line in file:
            filename = line.split()[0]
            filepath = os.path.join(root_path, filename)
            shutil.copy2(filepath, './custom_dataset/val/')


if __name__ == '__main__':
    root_path = 'f:/ActionRecognition'
    directories = os.listdir(root_path)
    # extra_dirs()
    # rename()
    # generate_setting()
    move_video_files()
