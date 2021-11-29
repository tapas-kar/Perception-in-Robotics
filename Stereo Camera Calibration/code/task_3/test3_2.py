import cv2
import numpy as np
import pathlib
import math
import matplotlib.pyplot as plt

workspace = pathlib.Path().cwd().parents[1]

task_3_and_4_image_directory = workspace.joinpath('images').joinpath('task_3_and_4')

left_images = []
right_images = []

for image in task_3_and_4_image_directory.iterdir():
    if image.name.startswith('left'):
        left_images.append(str(image.resolve()))
    else:
        right_images.append(str(image.resolve()))

left_images.sort()
right_images.sort()

# READING IN THE STEREO CALIBRATION AND STEREO RECTIFICATION PARAMETERS

# READ STEREO CALIBRATION
stereo_calibration_path = workspace.joinpath('parameters').joinpath('stereo_calibration.xml')

# print(stereo_calibration_path)
stereo_calibration_parameters = cv2.FileStorage(str(stereo_calibration_path.resolve()), cv2.FILE_STORAGE_READ)

left_camera_K = stereo_calibration_parameters.getNode("Left_camera_K").mat()
left_camera_D = stereo_calibration_parameters.getNode("Left_camera_D").mat()

right_camera_K = stereo_calibration_parameters.getNode("Right_camera_K").mat()
right_camera_D = stereo_calibration_parameters.getNode("Right_camera_D").mat()

rotation_matrix = stereo_calibration_parameters.getNode("Rotation_matrix").mat()
translation_matrix = stereo_calibration_parameters.getNode("Translation_matrix").mat()

essential_matrix = stereo_calibration_parameters.getNode("Essential_matrix").mat()
fundamental_matrix = stereo_calibration_parameters.getNode("Fundamental_matrix").mat()


# READ STEREO RECTIFICATION
stereo_rectification_path = workspace.joinpath('parameters').joinpath('stereo_rectification.xml')

stereo_rectification_parameters = cv2.FileStorage(str(stereo_rectification_path.resolve()), cv2.FILE_STORAGE_READ)

rectified_pose1 = stereo_rectification_parameters.getNode("Rectified_Pose1").mat()
rectified_pose2 = stereo_rectification_parameters.getNode("Rectified_Pose2").mat()

projection_matrix1 = stereo_rectification_parameters.getNode("Projection_matrix1").mat()
projection_matrix2 = stereo_rectification_parameters.getNode("Projection_matrix2").mat()

depth_matrix = stereo_rectification_parameters.getNode("Depth_matrix").mat()

orb = cv2.ORB_create(nfeatures=1000)
matcher = cv2.BFMatcher_create(normType=cv2.NORM_HAMMING, crossCheck=True)


def within_range(point1, point2, radius):
    return (point1[0] - point2[0]) * (point1[0] - point2[0]) + (point1[1] - point2[1]) * (point1[1] - point2[1]) \
           < radius * radius


def non_maxima_suppression(kp, des, radius=6):
    ret_kp = []
    ret_des = []
    n = len(kp)

    for index_1 in range(n):
        kp_index_1 = kp[index_1]
        des_index_1 = des[index_1]
        maxima = True

        for index_2 in range(n):
            if index_1 == index_2:
                continue

            kp_index_2 = kp[index_2]

            if within_range(kp_index_1.pt, kp_index_2.pt,
                                  radius) and kp_index_2.response > kp_index_1.response:
                maxima = False
                break

        if maxima:
            ret_kp.append(kp_index_1)
            ret_des.append(des_index_1)

    return np.array(ret_kp), np.array(ret_des)

def verify_epipolar_constraints(matches, F, left_kp, right_kp, match_filtering_ratio=20,
                                 epipolar_error_tolerance=2):
    threshold_dist = matches[0].distance * match_filtering_ratio
    good_matches = []

    for index, match in enumerate(matches):
        if match.distance > threshold_dist:
            break

        left_good_index = np.asarray(left_kp[match.queryIdx].pt)
        right_good_index = np.asarray(right_kp[match.trainIdx].pt)

        left_h_good_index = np.ones((3, 1))
        left_h_good_index[0:2, :] = left_good_index.reshape((2, 1))

        right_h_good_index = np.ones((3, 1))
        right_h_good_index[0:2, :] = right_good_index.reshape((2, 1))

        ec = np.asscalar(np.matmul(np.matmul(left_h_good_index.T, F), right_h_good_index))

        if math.fabs(ec) < epipolar_error_tolerance:
            good_matches.append(match)
    return good_matches


def plot_3d_scatter(points):
    x = []
    y = []
    z = []

    for point in points:
        x.append(point[0])
        y.append(point[1])
        z.append(point[2])

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z, linewidth=0.5)
    plt.show()


for left_image_path, right_image_path in zip(left_images, right_images):

    left_imgs = cv2.imread(left_image_path)
    right_imgs = cv2.imread(right_image_path)

    left_gray = cv2.cvtColor(left_imgs, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_imgs, cv2.COLOR_BGR2GRAY)

    left_undistorted_images = cv2.undistort(np.array(left_imgs), left_camera_K,
                                            left_camera_D, rectified_pose1, projection_matrix1)
    right_undistorted_images = cv2.undistort(np.array(right_imgs), right_camera_K,
                                             right_camera_D, rectified_pose2, projection_matrix2)

    cv2.imshow('left undistorted images', left_undistorted_images)
    cv2.imshow('right undistorted images', right_undistorted_images)
    cv2.waitKey(0)

    left_kp, left_descriptors = orb.detectAndCompute(left_undistorted_images, None)
    right_kp, right_descriptors = orb.detectAndCompute(right_undistorted_images, None)

    left_kp_img = cv2.drawKeypoints(left_undistorted_images, left_kp, color=(0, 255, 0), outImage=None)
    right_kp_img = cv2.drawKeypoints(right_undistorted_images, right_kp, color=(0, 255, 0), outImage=None)

    # cv2.imshow('left KP images', left_kp_img)
    # cv2.imshow('right KP images', right_kp_img)
    # cv2.waitKey(0)

    left_kp, left_descriptors = non_maxima_suppression(left_kp, left_descriptors)
    right_kp, right_descriptors = non_maxima_suppression(right_kp, right_descriptors)

    matches = matcher.match(left_descriptors, right_descriptors)

    matches = verify_epipolar_constraints(matches, fundamental_matrix, left_kp, right_kp)

    matches_img = cv2.drawMatches(left_undistorted_images, left_kp, right_undistorted_images, right_kp, matches,
                                  outImg=None)

    # cv2.imshow('matches', matches_img)
    # cv2.waitKey(0)

    points_3d = []

    for match in matches:
        kp1 = left_kp[match.queryIdx]
        kp2 = right_kp[match.trainIdx]

        triangulated_points = cv2.triangulatePoints(projection_matrix1, projection_matrix2,
                                                    np.array(kp1.pt).T, np.array(kp2.pt).T)
        triangulated_points /= triangulated_points[3]
        points_3d.append(triangulated_points)

    plot_3d_scatter(points_3d)
