import os

import cv2
import numpy as np


def get_video_capture(file_path, target_timestamp):
    timestamps = []
    for filename in os.listdir(file_path):
        if filename.endswith('.mp4') and not filename.startswith('.'):
            temp = filename.split('.')[0]
            temp = int(temp)
            timestamps.append(temp)

    closest_timestamp = min(timestamps, key=lambda x: abs(x - target_timestamp))
    filename = str(closest_timestamp) + '.mp4'

    video = os.path.join(file_path, filename)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video}")
    return cap


def get_frame_time(cap):
    # Assuming the video has frame time metadata
    return cap.get(cv2.CAP_PROP_POS_MSEC)


def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None, None
    time = get_frame_time(cap)
    return frame, time


def sync_videos(subject_path):
    pupil_path = os.path.join(subject_path, 'pupil')
    camera_path = os.path.join(subject_path, '3camera')
    pupil_files = os.listdir(pupil_path)
    # Find pupil video timestamp started at
    pupil_timestamp = 0
    for filename in pupil_files:
        if filename.endswith('.mp4') and not filename.startswith('.'):
            pupil_timestamp = filename.split('.')[0]
            pupil_timestamp = int(pupil_timestamp)
            break

    video_perspect = [
        os.path.join(camera_path, 'front'),
        os.path.join(camera_path, 'left'),
        os.path.join(camera_path, 'side'),
    ]

    caps = [get_video_capture(path, pupil_timestamp) for path in video_perspect]
    frames = [None] * len(caps)
    times = [None] * len(caps)
    print(frames)

    while True:
        for i, cap in enumerate(caps):
            frames[i], times[i] = read_frame(cap)

        if any(frame is None for frame in frames):
            break

        min_time = min(times)
        max_time = max(times)

        if max_time - min_time > 100:  # Example threshold of 100 ms
            for i, time in enumerate(times):
                if time == max_time:
                    frames[i], times[i] = read_frame(caps[i])
                    if frames[i] is not None:
                        frames[i] = cv2.resize(frames[i], (frames[i].shape[1] // 4, frames[i].shape[0] // 4))

        stacked_frame = np.hstack(frames)
        # img = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))

        cv2.imshow('Synced Videos', stacked_frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    root_path = r'E:\data'
    subjects = os.listdir(root_path)
    for subject in subjects:
        subject_path = os.path.join(root_path, subject)
        sync_videos(subject_path)
