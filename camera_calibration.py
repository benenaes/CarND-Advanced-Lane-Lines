import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import glob2
import pickle


def calculate_camera_matrices(images, nb_corners=(9,6), draw_corners=False):
    """
    Calculates the intrinsic (focal distance, principal points, distortion coefficients) parameters of the camera
    based on camera images containing a single chessboard pattern viewed from different perspectives
    :param images: Camera images containing the same chessboard pattern
    :param nb_corners: Tuple containing the number of (internal) chessboard corners in each row and each column
    :param draw_corners: Plots an image with the discovered corners for each image (for testing purposes)
    :return: (success, intrinsic parameter matrix, distortion coefficient vector)
    """
    row_corners = nb_corners[0]
    col_corners = nb_corners[1]
    image_points = []
    object_points = []
    total_corners = row_corners * col_corners
    grid_points = np.zeros((total_corners,3), np.float32)
    # Grid with points (0,0,0), (1,0,0) ... (row_corners - 1, 0, 0), (0, 1, 0) ... (row_corners - 1, col_corners - 1, 0)
    grid_points[:,:2] = np.mgrid[0:row_corners, 0: col_corners].T.reshape(-1,2) # infer 1st dim automatically

    if draw_corners:
        nb_images = len(images)
        if nb_images < 4:
            fig, axes = plt.subplots(1, nb_images, squeeze=False)
        else:
            fig, axes = plt.subplots(math.ceil(nb_images / 4), 4, squeeze=False)
        fig.tight_layout()
        plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1, wspace=0.025, hspace=0.05)
        ax1d = axes.flatten()
        for subplot in ax1d:
            subplot.axis('off')

    for idx, image in enumerate(images):
        gray_img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)
        shape = gray_img.shape[::-1]
        pattern_found, corners = cv2.findChessboardCorners(gray_img, patternSize=nb_corners)
        if draw_corners:
            corner_img = cv2.drawChessboardCorners(
                gray_img, patternSize=nb_corners, corners=corners, patternWasFound=pattern_found)
            ax1d[idx].imshow(corner_img, cmap='gray')
        if pattern_found:
            image_points.append(corners)
            object_points.append(grid_points)

    if draw_corners:
        _ = plt.show(block=True)

    if len(image_points):
        success, intrinsic_params, dist_coeffs, _, _ = cv2.calibrateCamera(
            object_points, image_points, shape, None, None)
        return success, intrinsic_params, dist_coeffs
    else:
        return False, None, None


def save_calibration_params(filename, intrinsic_params, dist_coeffs):
    """
    Save the camera calibration parameters
    :param filename: The filename where the calibration parameters need to be stored
    :param intrinsic_params: The intrinsic parameters matrix
    :param dist_coeffs: The vector with distortion coefficients
    :return: Nothing
    """
    calibration_data = {}
    calibration_data["intrinsics"] = intrinsic_params
    calibration_data["dist"] = dist_coeffs
    pickle.dump(calibration_data, open(filename, "wb"))


def load_calibration_params(filename):
    """
    Load the camera calibration parameters
    :param filename: The filename where the calibration parameters were stored
    :return: If the filename exists,
             returns a tuple with the intrinsic parameters matrix and the vector with distortion coefficients.
             Else None, None
    """
    calibration_data = pickle.load(open(filename, "rb"))
    if calibration_data:
        intrinsic_params = calibration_data["intrinsics"]
        dist_coeffs = calibration_data["dist"]
        return intrinsic_params, dist_coeffs
    else:
        return None, None


if __name__ == "__main__":
    image_files = glob2.glob("./camera_cal/*.jpg")
    success, intrinsic_params, dist_coeffs = calculate_camera_matrices(image_files, draw_corners=False)
    if success:
        save_calibration_params("camera_cal/calibration_data.p", intrinsic_params, dist_coeffs)
        gray_img1 = cv2.cvtColor(cv2.imread(image_files[0]), cv2.COLOR_BGR2GRAY)
        undistorted_image = cv2.undistort(
            gray_img1,
            intrinsic_params, dist_coeffs)
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(gray_img1, cmap='gray')
        axes[0].axis('off')
        axes[0].set_title("Original 1st image")
        axes[1].imshow(undistorted_image, cmap='gray')
        axes[1].axis('off')
        axes[1].set_title("Undistorted 1st image")
        plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1)
        _ = plt.show(block=True)






