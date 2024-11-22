import os
from scripts.alignment import get_video_path
import shutil

if __name__ == '__main__':
    root_path = r'D:\DeskBehaviorData'

    subjects = os.listdir(root_path)

    for subject in subjects:
        subject_path = os.path.join(root_path, subject)

        pupil_path = os.path.join(subject_path, 'pupil')
        camera_path = os.path.join(subject_path, r'3camera\front')
        pupil_files = os.listdir(pupil_path)
        # Find pupil video timestamp started at
        pupil_timestamp = 0
        for filename in pupil_files:
            if filename.endswith('.mp4') and not filename.startswith('.'):
                pupil_timestamp = filename.split('.')[0]
                pupil_timestamp = int(pupil_timestamp)
                break

        source_video_path = get_video_path(camera_path, pupil_timestamp)

        destination_video_path = fr'D:\UploadtoA100\{subject}.mp4'

        shutil.copy2(source_video_path, destination_video_path)