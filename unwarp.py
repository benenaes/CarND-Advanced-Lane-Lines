import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from camera_calibration import load_calibration_params


def show_transformation(orig_img, transformation_matrix):
    """
    Displays an original image and an image warped by the given transformation matrix next to each other
    :param orig_img: The original (RGB) image
    :param transformation_matrix: The transformation matrix that warps the original image
    :return: Nothing
    """
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(orig_img)
    axes[0].axis('off')
    axes[0].set_title("Original image")
    warped_image = cv2.warpPerspective(
        orig_img, transformation_matrix, orig_img.shape[1::-1], flags=cv2.INTER_LINEAR)
    axes[1].imshow(warped_image)
    axes[1].axis('off')
    axes[1].set_title("Warped image")
    plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1)
    _ = plt.show(block=True)


def calculate_perspective_transform(img, trapezoid, offset = 100, show_transform=False):
    """
    Calculates a transformation matrix based that transforms a given trapezoid to a rectangle
    :param img: The source (RGB) image
    :param trapezoid: 1D array of corner points that defines the trapezoid: [top_left, top_right, bottom_right, bottom_left]
    :param offset: Defines the rectangle (destination shape). The rectangle is defined by the following points:
                    [(offset, 0), (img_width - offset, 0), (img_width - offset, img_height), (offset, img_height)]
    :param show_transform: Applies the transformation matrix on the given image and shows the result
    :return: The transformation matrix and the inverse transformation matrix
    """
    (img_width, img_height) = img.shape[1::-1]
    src = np.float32(trapezoid)
    dst = np.float32([
        [offset, 0],
        [img_width - offset, 0],
        [img_width - offset, img_height],
        [offset, img_height]])
    perspective_matrix = cv2.getPerspectiveTransform(src, dst)
    inverse_perspective_matrix = cv2.getPerspectiveTransform(dst,src)

    if show_transform:
        pts = src.copy()
        pts = pts.reshape((-1, 1, 2))
        polygon_img = np.zeros_like(img)
        cv2.polylines(polygon_img, np.int32([pts]), True, (255, 0, 0), thickness=5)
        blended_image = cv2.addWeighted(img, 1, polygon_img, 0.3, 0)
        show_transformation(blended_image, perspective_matrix)

    return perspective_matrix, inverse_perspective_matrix


def save_perspective_matrices(filename, matrix, inverse_matrix):
    """
    Save the perspective matrix and its inverse
    :param filename: The filename where the matrices need to be stored
    :param matrix: The perspective matrix
    :param inverse_matrix: The inverse perspective matrix
    :return: Nothing
    """
    perspective_data = {}
    perspective_data["perspective_matrix"] = matrix
    perspective_data["inverse_perspective_matrix"] = inverse_matrix
    pickle.dump(perspective_data, open(filename, "wb"))


def load_perspective_matrices(filename):
    """
    Load the perspective matrix and its inverse
    :param filename: The filename where the perspective matrix and its inverse were stored
    :return: If the filename exists,
             returns a tuple with the perspective matrix and its inverse
             Else None, None
    """
    perspective_data = pickle.load(open(filename, "rb"))
    if perspective_data:
        matrix = perspective_data["perspective_matrix"]
        inverse_matrix = perspective_data["inverse_perspective_matrix"]
        return matrix, inverse_matrix
    else:
        return None, None


def find_perspective_matrix_on_straight_lanes(intrinsic_params, dist_coeffs, offset=250):
    straight_lane_image_path = 'test_images/straight_lines2.jpg'
    straight_lane_image = cv2.cvtColor(cv2.imread(straight_lane_image_path), cv2.COLOR_BGR2RGB)
    undistorted_straight_lane_image = cv2.undistort(
        straight_lane_image,
        intrinsic_params, dist_coeffs)
    # Handpicked coordinates
    trapezoid = [(585, 456), (699, 456), (1055, 685), (266, 685)]
    perspective_matrix, inverse_perspective_matrix = calculate_perspective_transform(
        undistorted_straight_lane_image, trapezoid=trapezoid, offset=offset, show_transform=False)
    return perspective_matrix, inverse_perspective_matrix


if __name__ == "__main__":
    # Load the intrinsic parameters of the camera
    intrinsic_params, dist_coeffs = load_calibration_params("camera_cal/calibration_data.p")

    # Test image with a straight line on a flat surface: perfect to use this to calculate the camera's perspective matrix
    straight_lane_image_path = 'test_images/straight_lines2.jpg'
    straight_lane_image = cv2.cvtColor(cv2.imread(straight_lane_image_path), cv2.COLOR_BGR2RGB)
    undistorted_straight_lane_image = cv2.undistort(
        straight_lane_image,
        intrinsic_params, dist_coeffs)
    # Handpicked coordinates
    trapezoid = [(585, 456), (699, 456), (1055, 685), (266, 685)]
    offset = 300
    perspective_matrix, inverse_perspective_matrix = calculate_perspective_transform(
        undistorted_straight_lane_image, trapezoid=trapezoid, offset=offset, show_transform=True)
    save_perspective_matrices("camera_cal/perspective_matrix.p", perspective_matrix, inverse_perspective_matrix)

    cross_check_image_path = 'test_images/straight_lines1.jpg'
    cross_check_image = cv2.cvtColor(cv2.imread(cross_check_image_path), cv2.COLOR_BGR2RGB)
    undistorted_cross_check_image = cv2.undistort(
        cross_check_image,
        intrinsic_params, dist_coeffs)
    show_transformation(undistorted_cross_check_image, perspective_matrix)