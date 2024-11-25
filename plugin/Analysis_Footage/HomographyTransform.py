import cv2
import numpy as np

def compute_homography_and_transform_gaze(img1, img2, gaze_point,show=False):

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)


    good = []

    try:
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
    except:
        return None, None, None

    MIN_MATCH_COUNT = 10  # 设定最少匹配点数

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

        # 计算单应性矩阵
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 转换 gaze 点
        gaze_point = np.array([[gaze_point]], dtype='float32')  # gaze_point 为 [x, y]
        try:
            transformed_point = cv2.perspectiveTransform(gaze_point, M)
            new_gaze_x, new_gaze_y = transformed_point[0][0]
        except:
            return None, None, None


        if show :

            print("M", M)

            img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good, None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            img_matches = cv2.resize(img_matches, (int(img_matches.shape[1] * 0.5), int(img_matches.shape[0] * 0.5)))

            cv2.imshow('img_matches', img_matches)

            cv2.waitKey(0)

        return new_gaze_x, new_gaze_y, M
    else:
        return None, None, None

if __name__ == '__main__':

    img1 = cv2.imread(r"D:\Bryant_Python\Tools\EyetrackerCore\Analysis_Footage\gaze_recording\732.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(r"D:\Bryant_Python\Tools\EyetrackerCore\Analysis_Footage\gaze_recording\863.jpg", cv2.IMREAD_GRAYSCALE)
    gaze_point = [100, 150]  # Example gaze point

    compute_homography_and_transform_gaze(img1, img2, gaze_point,show=True)