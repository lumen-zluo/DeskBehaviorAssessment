import os
import shutil

if __name__ == '__main__':
    root_path = r'D:\DeskBehaviorData'

    subjects = os.listdir(root_path)

    for subject in subjects:
        subject_path = os.path.join(root_path, subject)

        source_folder = os.path.join(subject_path, 'pupil')

        destination_folder = fr'D:\DesktopEyetrackerData\{subject}'

        shutil.copytree(source_folder, destination_folder)
