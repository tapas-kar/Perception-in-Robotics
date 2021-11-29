import cv2
import numpy as np
import pathlib
import matplotlib.pyplot as plt

workspace = pathlib.Path().cwd().parents[1]

task_2_image_directory = workspace.joinpath('images').joinpath('task_2')
task_2_output_task_2_path = workspace.joinpath('output').joinpath('task_2')

left_imgs = []
right_imgs = []

for file in task_2_image_directory.iterdir():
    if file.name.startswith('left'):
        left_imgs.append(file)
    else:
        right_imgs.append(file)

left_imgs.sort()
right_imgs.sort()

# l_r = [left_images[0], right_images[0]]
# r_l = [left_images[1], right_images[1]]
#
# left_images = l_r
# right_images = r_l

def plot(points, R, T):
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

    f = 1
    tan_x = 1
    tan_y = 1

    R_prime = np.identity(3)
    t_prime = np.zeros((3, 1))

    cam_1_center_local = np.asarray([
        [0, 0, 0], [tan_x, tan_y, 1],
        [tan_x, -tan_y, 1], [0, 0, 0], [tan_x, -tan_y, 1],
        [-tan_x, -tan_y, 1], [0, 0, 0], [-tan_x, -tan_y, 1],
        [-tan_x, tan_y, 1], [0, 0, 0], [-tan_x, tan_y, 1],
        [tan_x, tan_y, 1], [0, 0, 0]
    ]).T

    cam_1_center_local *= f
    cam_1_center = np.matmul(R_prime, cam_1_center_local) + t_prime

    cam_2_center_local = np.asarray([
        [0, 0, 0], [tan_x, tan_y, 1],
        [tan_x, -tan_y, 1], [0, 0, 0], [tan_x, -tan_y, 1],
        [-tan_x, -tan_y, 1], [0, 0, 0], [-tan_x, -tan_y, 1],
        [-tan_x, tan_y, 1], [0, 0, 0], [-tan_x, tan_y, 1],
        [tan_x, tan_y, 1], [0, 0, 0]
    ]).T

    cam_2_center_local *= f
    cam_2_center = np.matmul(R, cam_2_center_local) + T

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.plot(cam_1_center[0, :], cam_1_center[1, :], cam_1_center[2, :],
            color='k', linewidth=2)
    ax.plot(cam_2_center[0, :], cam_2_center[1, :], cam_2_center[2, :],
            color='k', linewidth=2)
    ax.scatter(x, y, z)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Read left camera K1 and D1 and right camera K2 and D2
# Left camera parameters
left_parameter_paths = workspace.joinpath('parameters').joinpath('left_camera_intrinsics.xml')

left_parameters = cv2.FileStorage(str(left_parameter_paths.resolve()), cv2.FILE_STORAGE_READ)
left_camera_matrix = left_parameters.getNode("K").mat()
left_camera_distortion_coeff = left_parameters.getNode("D").mat()

left_parameters.release()


# Right camera parameters
right_parameter_paths = workspace.joinpath('parameters').joinpath('right_camera_intrinsics.xml')

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

for left_image_path, right_image_path in zip(left_imgs, right_imgs):

    left_image = cv2.imread(str(left_image_path.resolve()))
    right_image = cv2.imread(str(right_image_path.resolve()))

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
        cv2.imshow('refined points on chessboard', left_image)
        cv2.waitKey(0)
        output_file = task_2_output_task_2_path.joinpath('refined_points_{}'.format(left_image_path.name))
        cv2.imwrite(str(output_file.resolve()), left_image)

        cv2.drawChessboardCorners(right_image, (9, 6), right_corners2, right_ret)
        cv2.imshow('refined points on chessboard', right_image)
        cv2.waitKey(0)
        output_file = task_2_output_task_2_path.joinpath('refined_points_{}'.format(right_image_path.name))
        cv2.imwrite(str(output_file.resolve()), right_image)

        undistorted_left = cv2.undistort(left_image, left_camera_matrix, left_camera_distortion_coeff)
        undistorted_right = cv2.undistort(right_image, right_camera_matrix, right_camera_distortion_coeff)

        output_file_undistorted1 = task_2_output_task_2_path.joinpath('undistorted_image_{}'.format(left_image_path.name))
        cv2.imwrite(str(output_file_undistorted1.resolve()), undistorted_left)

        output_file_undistorted2 = task_2_output_task_2_path.joinpath('undistorted_image_{}'.format(right_image_path.name))
        cv2.imwrite(str(output_file_undistorted2.resolve()), undistorted_left)


ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objpoints, left_image_points, right_image_points,
                                                      left_camera_matrix, left_camera_distortion_coeff,
                                                      right_camera_matrix, right_camera_distortion_coeff, (9,6),
                                                      flags=cv2.CALIB_FIX_INTRINSIC)

identity_3 = np.identity(3)
zeros_3 = np.zeros((3, 1))

P1 = np.append(identity_3, zeros_3, 1)
P2 = np.append(R, T, 1)

pts_3D = []

# Undistorting left and right points before triangulating
left_undistorted_points = cv2.undistortPoints(np.array(left_image_points[0]), left_camera_matrix,
                                              left_camera_distortion_coeff)

right_undistorted_points = cv2.undistortPoints(np.array(right_image_points[0]), right_camera_matrix,
                                               right_camera_distortion_coeff)



for left_point, right_point in zip(left_undistorted_points, right_undistorted_points):

    triangulated_pts = cv2.triangulatePoints(P1, P2, np.array(left_point).T, np.array(right_point).T)

    triangulated_pts /= (triangulated_pts[3] + 1e-9)

    pts_3D.append(triangulated_pts)

plot(pts_3D, R, T)


R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K1, D1, K2, D2, (9, 6),
                                                           R, T, flags=cv2.CALIB_FIX_INTRINSIC)

for left_image_path, right_image_path in zip(left_imgs, right_imgs):

    left_image = cv2.imread(str(left_image_path.resolve()))
    right_image = cv2.imread(str(right_image_path.resolve()))

    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    left_ret, left_corners = cv2.findChessboardCorners(left_gray, (9, 6), None)
    right_ret, right_corners = cv2.findChessboardCorners(right_gray, (9, 6), None)

    if left_ret and right_ret:
        objpoints.append(objp)

        left_corners2 = cv2.cornerSubPix(left_gray, left_corners, (11, 11), (-1, -1), criteria)
        right_corners2 = cv2.cornerSubPix(right_gray, right_corners, (11, 11), (-1, -1), criteria)

        left_image_points.append(left_corners)
        right_image_points.append(right_corners)

        cv2.drawChessboardCorners(left_image, (9, 6), left_corners2, left_ret)
        cv2.imshow('refined points on chessboard', left_image)
        cv2.waitKey(0)
        output_file = task_2_output_task_2_path.joinpath('refined_points_{}'.format(left_image_path.name))
        cv2.imwrite(str(output_file.resolve()), left_image)

        cv2.drawChessboardCorners(right_image, (9, 6), right_corners2, right_ret)
        cv2.imshow('refined points on chessboard', right_image)
        cv2.waitKey(0)
        output_file = task_2_output_task_2_path.joinpath('refined_points_{}'.format(right_image_path.name))
        cv2.imwrite(str(output_file.resolve()), right_image)

        undistorted_left = cv2.undistort(left_image, left_camera_matrix, left_camera_distortion_coeff)
        undistorted_right = cv2.undistort(right_image, right_camera_matrix, right_camera_distortion_coeff)

        undistorted_rectified_left = cv2.undistort(left_image, left_camera_matrix, left_camera_distortion_coeff,
                                                   R1, P1)

        undistorted_rectified_right = cv2.undistort(right_image, right_camera_matrix, right_camera_distortion_coeff,
                                                   R2, P2)

        output_file_rectified1 = task_2_output_task_2_path.joinpath('rectified_{}'.format(left_image_path.name))
        cv2.imwrite(str(output_file_rectified1.resolve()), undistorted_rectified_left)

        output_file_rectified2 = task_2_output_task_2_path.joinpath('rectified_{}'.format(right_image_path.name))
        cv2.imwrite(str(output_file_rectified2.resolve()), undistorted_rectified_right)




# STEP - 7 STORING THE STEREO CALIBRATION PARAMETERS AND STEREO RECTIFICATION PARAMETERS INTO XML FILES
# ------------------------------------------------------------------------------------------------------

# STEREO CALIBRATION RESULTS
cv_file = cv2.FileStorage(str(workspace.joinpath('parameters').joinpath('stereo_calibration.xml').resolve()),
                          cv2.FILE_STORAGE_WRITE)
cv_file.write("Left_camera_K", K1)
cv_file.write("Left_camera_D", D1)

cv_file.write("Right_camera_K", K2)
cv_file.write("Right_camera_D", D2)

cv_file.write("Rotation_matrix", R)
cv_file.write("Translation_matrix", T)

cv_file.write("Essential_matrix", E)
cv_file.write("Fundamental_matrix", F)

# note you *release* you don't close() a FileStorage object
cv_file.release()


# STEREO RECTIFICATION RESULTS
cv_file = cv2.FileStorage(str(workspace.joinpath('parameters').joinpath('stereo_rectification.xml').resolve()),
                          cv2.FILE_STORAGE_WRITE)
cv_file.write("Rectified_Pose1", R1)
cv_file.write("Rectified_Pose2", R2)

cv_file.write("Projection_matrix1", P1)
cv_file.write("Projection_matrix2", P2)

cv_file.write("Depth_matrix", Q)

# note you *release* you don't close() a FileStorage object
cv_file.release()
