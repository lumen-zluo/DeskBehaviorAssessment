import cv2
import numpy as np
import time
from skimage.metrics import structural_similarity as ssim


def are_images_similar(img1, img2, threshold=0.70):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score = ssim(gray1, gray2)
    # print(f"SSIM: {score}")
    return score > threshold


def filter_flow_by_direction(good_p0, good_p1, threshold=30):
    # Calculate motion vectors
    motion_vectors = good_p1 - good_p0
    angles = np.arctan2(motion_vectors[:, 1], motion_vectors[:, 0]) * 180 / np.pi

    # Calculate the main direction
    main_angle = np.median(angles)

    # Filter points by direction difference
    condition = np.abs(angles - main_angle) <= threshold
    return good_p0[condition], good_p1[condition]


def CalOptFlow(img1, img2, show=False):
    if not are_images_similar(img1, img2):
        print("Images are too different.")
        return None, None
    else:
        try:

            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=5, blockSize=7)
            p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)

            lk_params = dict(winSize=(35, 35), maxLevel=3)
            p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

            good_p0 = p0[st == 1]
            good_p1 = p1[st == 1]

            # Filter points
            filtered_p0, filtered_p1 = filter_flow_by_direction(good_p0, good_p1)
            dx = np.mean(filtered_p1[:, 0] - filtered_p0[:, 0])
            dy = np.mean(filtered_p1[:, 1] - filtered_p0[:, 1])



            if show:

                # Visualize filtered points
                for i, (new, old) in enumerate(zip(filtered_p1, filtered_p0)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    cv2.circle(img2, (int(a), int(b)), 5, (0, 255, 0), -1)
                    cv2.line(img2, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)

                img2 = cv2.resize(img2, (int(img2.shape[1] / 2), int(img2.shape[0] / 2)))
                cv2.imshow('Tracked Features', img2)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return dx, dy

        except:

            return None, None

        # print(f"Estimated translation: dx = {dx}, dy = {dy}")
        # print("Process time:", time.time() - start_time)

if __name__ == '__main__':

    img1 = cv2.imread(r"D:\Bryant_Python\Tools\EyetrackerCore\Analysis_Footage\temp_img\img_36.jpg")
    img2 = cv2.imread(r"D:\Bryant_Python\Tools\EyetrackerCore\Analysis_Footage\temp_img\img_37.jpg")
    dx, dy = CalOptFlow(img1, img2, show=True)
    print("dx, dy:", dx, dy)

