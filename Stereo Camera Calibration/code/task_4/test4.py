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

win_size = 9
min_disp = 0

max_disp = 64
num_disp = max_disp - min_disp
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=win_size,
                               preFilterCap=63,
                               uniquenessRatio=15,
                               speckleWindowSize=10,
                               speckleRange=1,
                               disp12MaxDiff=20,
                               P1=8*3*win_size**2,
                               P2=32*3*win_size**2,
                               mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

left_matcher = stereo
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

l = 70000
s = 1.2

disparity_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
disparity_filter.setLambda(l)
disparity_filter.setSigmaColor(s)

for left_image_path, right_image_path in zip(left_images, right_images):

    left_imgs = cv2.imread(left_image_path)
    right_imgs = cv2.imread(right_image_path)

    left_gray = cv2.cvtColor(left_imgs, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_imgs, cv2.COLOR_BGR2GRAY)

    left_undistorted_images = cv2.undistort(np.array(left_imgs), left_camera_K,
                                            left_camera_D, rectified_pose1, projection_matrix1)
    right_undistorted_images = cv2.undistort(np.array(right_imgs), right_camera_K,
                                             right_camera_D, rectified_pose2, projection_matrix2)

    d_l = left_matcher.compute(left_undistorted_images, right_undistorted_images)
    d_r = right_matcher.compute(right_undistorted_images, left_undistorted_images)

    d_l = np.int16(d_l)
    d_r = np.int16(d_r)

    d_filter = disparity_filter.filter(d_l, left_undistorted_images, None, d_r)

    print("|n Computing the disparity map ...")

    disparity_map = stereo.compute(left_undistorted_images, right_undistorted_images)

    disp = cv2.normalize(d_filter, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imshow('Disparity', disp)
    cv2.waitKey(0)

