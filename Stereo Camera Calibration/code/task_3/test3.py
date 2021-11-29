import cv2
import numpy as np
import pathlib

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

# Read left camera K1 and D1 and right camera K2 and D2
# Left camera parameters
left_parameter_paths = workspace.joinpath('parameters').joinpath('left_camera_intrinsics.xml')
# print(left_parameter_paths)

left_parameters = cv2.FileStorage(str(left_parameter_paths.resolve()), cv2.FILE_STORAGE_READ)
left_camera_matrix = left_parameters.getNode("K").mat()
left_camera_distortion_coeff = left_parameters.getNode("D").mat()

left_parameters.release()


# Right camera parameters
right_parameter_paths = workspace.joinpath('parameters').joinpath('right_camera_intrinsics.xml')
# print(right_parameter_paths)

right_parameters = cv2.FileStorage(str(right_parameter_paths.resolve()), cv2.FILE_STORAGE_READ)
right_camera_matrix = right_parameters.getNode("K").mat()
right_camera_distortion_coeff = right_parameters.getNode("D").mat()

right_parameters.release()

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints = []
left_image_points = []
right_image_points = []

for left_image_path, right_image_path in zip(left_images, right_images):

    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)

    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    left_ret, left_corners = cv2.findChessboardCorners(left_gray, (9, 6), None)
    right_ret, right_corners = cv2.findChessboardCorners(right_gray, (9, 6), None)

    if left_ret and right_ret:
        objpoints.append(objp)

        left_corners2 = cv2.cornerSubPix(left_gray, left_corners, (11,11), (-1,-1), criteria)
        right_corners2 = cv2.cornerSubPix(right_gray, right_corners, (11, 11), (-1, -1), criteria)

        left_image_points.append(left_corners)
        right_image_points.append(right_corners)

        cv2.drawChessboardCorners(left_image, (9, 6), left_corners2, left_ret)
        # cv2.imshow('refined points on chessboard', left_image)
        # cv2.waitKey(0)

        cv2.drawChessboardCorners(right_image, (9, 6), right_corners2, right_ret)
        # cv2.imshow('refined points on chessboard', right_image)
        # cv2.waitKey(0)



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


# Undistort the images
# Undistort the left images
img = cv2.imread(left_image_points)
h, w = img.shape[:2]

dst = cv2.undistort(img, left_camera_K, left_camera_D, rectified_pose1, projection_matrix1)

cv2.imwrite('undistorted_left_image_points.png', dst)


