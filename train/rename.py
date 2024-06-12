import os
import re

if __name__ == '__main__':
    folder_path = r'C:\Users\Owner\Downloads\subject7\subject7\pupil\imgs\\'
    files = os.listdir(folder_path)
    file_prefix = 'jpgs_'
    file_index = 969

    for file in files:
        match = re.search(r"\d+", file)
        current_index = int(match.group())
        replacement = str(current_index + file_index)
        new_file = re.sub(r"\d+", replacement, file)

        # 构建原文件路径和新文件路径
        old_path = os.path.join(folder_path, file)
        new_path = os.path.join(folder_path, new_file)

        # 重命名文件
        os.rename(old_path, new_path)

        print(f"重命名文件: {file} -> {new_file}")
