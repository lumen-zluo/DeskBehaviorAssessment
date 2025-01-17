import os
import pandas as pd
import matplotlib.pyplot as plt

# 定义包含所有 subject 文件夹的主目录路径
base_path = '/Users/gxm/Desktop/desk behavior assessment/plugin/Analysis_Footage/output'
output_path = '/Users/gxm/Desktop/fixation_histograms'

# 创建输出文件夹
if not os.path.exists(output_path):
    os.makedirs(output_path)

all_objects = set()

# 第一次遍历：数据聚合并收集objects
for subject_folder in os.listdir(base_path):
    subject_path = os.path.join(base_path, subject_folder)
    fixation_file_path = os.path.join(subject_path, 'fixations.csv')

    # 检查 fixations.csv 文件是否存在
    if os.path.exists(fixation_file_path):
        # 读取 CSV 文件
        fixation_data = pd.read_csv(fixation_file_path)

        # 聚合数据
        aggregated_data = fixation_data.drop_duplicates()

        # 保存
        aggregated_file_path = os.path.join(subject_path, 'fixations_aggregated.csv')
        aggregated_data.to_csv(aggregated_file_path, index=False)

        print(f"{subject_folder} 的数据已聚合并保存。")

        # 收集objects
        all_objects.update(aggregated_data['object'].dropna().unique())

# 将空值也加入到 all_objects，代表 "others"
all_objects.add("others")

# 排序objects
all_objects = sorted(all_objects)

# 第二次遍历：绘制直方图
for subject_folder in os.listdir(base_path):
    subject_path = os.path.join(base_path, subject_folder)
    fixation_file = os.path.join(subject_path, 'fixations_aggregated.csv')

    if os.path.isfile(fixation_file):
        # 读取聚合后的数据
        data = pd.read_csv(fixation_file)

        # 替换空值为 "others"
        data['object'].fillna("others", inplace=True)

        # 统计object的数量
        counts = data['object'].value_counts().to_dict()
        counts = {obj: counts.get(obj, 0) for obj in all_objects}

        # 绘制直方图
        plt.figure(figsize=(8, 8)) 
        bars = plt.bar(counts.keys(), counts.values(), color='skyblue', edgecolor='black', width=0.9)

        # 设置柱体样式
        for bar in bars:
            bar.set_linewidth(1.5)

        plt.xticks(rotation=45)
        plt.xlabel("Objects")
        plt.ylabel("Fixation Count")
        plt.title(f"{subject_folder} - Fixation Count Distribution")

        # 保存图像
        output_file = os.path.join(output_path, f"{subject_folder}_fixation_histogram.png")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

print(f"所有文件已聚合并生成直方图，保存到：{output_path}")
