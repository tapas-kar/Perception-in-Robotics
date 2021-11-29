import pathlib
import numpy as np
import cv2

workspace = pathlib.Path().cwd().parents[1]

left_imgs = []
right_imgs = []

task_1_images_directory = workspace.joinpath('images').joinpath('task_1')
task_1_output_task_1_path = workspace.joinpath('output').joinpath('task_1')
# print(task_1_output_task_1_path)

for file in task_1_images_directory.iterdir():
    if file.name.startswith('left'):
        left_imgs.append(file)
    else:
        right_imgs.append(file)

left_imgs.sort()
right_imgs.sort()

print(left_imgs)
print(right_imgs)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points like (0,0,0), (1,0,0), (2,0,0), ...., (6,5,0)
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9,0:6].T.reshape(-1,2)


# Arrays to store object points and image points from all the images
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

for index, fname in enumerate(right_imgs):
        img = cv2.imread(str(fname.resolve()))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points (after refining them)
        if ret:

            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners2, ret)
            cv2.imshow('refined points on chessboard', img)
            cv2.waitKey(0)
            output_file = task_1_output_task_1_path.joinpath('chessboard_{}'.format(fname.name))
            cv2.imwrite(str(output_file.resolve()), img)

cv2.destroyAllWindows()

# CALIBRATION
# -------------

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


# UNDISTORTION
# -------------
img = cv2.imread(str(task_1_images_directory.joinpath('right_2.png').resolve()))
h, w = img.shape[:2]


#undistort
dst = cv2.undistort(img, mtx, dist)

output_file = task_1_output_task_1_path.joinpath('right_undistorted_image2.png')

cv2.imwrite(str(output_file.resolve()), dst)


# STEP - 5 STORING THE CAMERA INTRINSIC MATRIX AND DISTORTION COEFFICIENTS INTO XML FILES
# ----------------------------------------------------------------------------------------
cv_file = cv2.FileStorage(str(workspace.joinpath('parameters').joinpath('right_camera_intrinsics.xml').resolve()),
                          cv2.FILE_STORAGE_WRITE)
cv_file.write("K", mtx)
cv_file.write("D", dist)

# note you *release* you don't close() a FileStorage object
cv_file.release()


