import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

from image_filters import filter_image
from sliding_windows_fit import fit_polynomial_sliding_window
from lane_line_history import LaneHistory

from camera_calibration import load_calibration_params
from unwarp import load_perspective_matrices
from camera_parameters import CameraParameters


def process_lanes(left_lane_history, right_lane_history):
    """
    Process the fits of the left and right lanes and check if the found lane lines are good candidates
    :param left_lane_history: The statistics of the left lane line detection (containing the left lane line candidate)
    :param right_lane_history: The statistics of the right lane line detection (containing the right lane line candidate)
    :return: The adapted statistics of the left and right lane line containing smoothened or replaced fits
    """
    # Maximum number of subsequent bad fits before the lanes are not displayed anymore
    max_bad_fits_for_non_detection = 20
    # Maximum number of subsequent bad fits before a new histogram is made at the beginning of the lane line detection
    # algorithm to find the peaks that are candidates to start the sliding window algorithm
    max_bad_fits_for_lane_histogram = 3
    # Maximum difference in pixels between subsequent detections of the same lane line
    # to be accepted as a good candidate
    max_diff_on_bottom_line = 100
    # Maximum distance in between the lane lines on 2/3 of the warped view
    max_lane_distance = 800
    # Minimum distance in between the lane lines on 2/3 of the warped view
    min_lane_distance = 400
    # Minimum pixel height where the lane lines may intersect (outside the image) for the two fits to be good candidates
    min_lane_line_intersection = -200
    # Maximum pixel height where the lane lines can intersect (outside the image) for the two fits to be good candidates
    max_lane_line_intersection = 920
    # Maximum slope difference of the two lane lines to be accepted as good candidates
    max_slope_difference = 1.0

    good_left_fit = True
    if left_lane_history.detected:
        candidate_bottom_line_intersection = int(left_lane_history.current_fit[0] * 719 ** 2
                                                         + left_lane_history.current_fit[1] * 719
                                                         + left_lane_history.current_fit[2])
        # The difference in between the current and the previous fit on the bottom line is too large
        if abs(candidate_bottom_line_intersection - left_lane_history.bottom_line_intersection) > max_diff_on_bottom_line:
            good_left_fit = False
    else:
        good_left_fit = False

    good_right_fit = True
    if right_lane_history.detected:
        candidate_bottom_line_intersection = int(right_lane_history.current_fit[0] * 719 ** 2
                                                          + right_lane_history.current_fit[1] * 719
                                                          + right_lane_history.current_fit[2])
        # The difference in between the current and the previous fit on the bottom line is too large
        if abs(candidate_bottom_line_intersection - right_lane_history.bottom_line_intersection) > max_diff_on_bottom_line:
            good_right_fit = False
    else:
        good_right_fit = False

    candidate_left_fit = left_lane_history.current_fit
    if not good_left_fit:
        candidate_left_fit = left_lane_history.best_fit

    candidate_right_fit = right_lane_history.current_fit
    if not good_right_fit:
        candidate_right_fit = right_lane_history.best_fit

    if candidate_left_fit is None or candidate_right_fit is None:
        good_left_fit = False
        good_right_fit = False
    # Slope must be more or less equal than max_slope_difference
    elif abs(candidate_left_fit[1] - candidate_right_fit[1]) > max_slope_difference:
        good_left_fit = False
        good_right_fit = False
    else:
        # Check distance on the 2/3 of the screen height
        left_lane_x_pos = int(candidate_left_fit[0] * 480 ** 2 + candidate_left_fit[1] * 480 + candidate_left_fit[2])
        right_lane_x_pos = int(candidate_right_fit[0] * 480 ** 2 + candidate_right_fit[1] * 480 + candidate_right_fit[2])
        if right_lane_x_pos - left_lane_x_pos > max_lane_distance or \
                                right_lane_x_pos - left_lane_x_pos < min_lane_distance:
            good_left_fit = False
            good_right_fit = False
        else:
            # Calculate intersections of the two lines, so where left - right = 0
            # For a * x^2 + b * x + c = 0, the solution is: (-b +- sqrt(b ^ 2 - 4 * a * c) / ( 2 * a)
            a = candidate_left_fit[0] - candidate_right_fit[0]
            b = candidate_left_fit[1] - candidate_right_fit[1]
            c = candidate_left_fit[2] - candidate_right_fit[2]
            tmp = np.sqrt(b ** 2 - 4 * a * c)
            intersection_y_1 = (-b + tmp) / (2 * a)
            intersection_y_2 = (-b - tmp) / (2 * a)

            # If the crossings of the two lines are in the image or too close to the borders, then
            # the lines aren't parallel enough and thus the found solutions are probably bogus
            if min_lane_line_intersection < intersection_y_1 < max_lane_line_intersection or \
                                    min_lane_line_intersection < intersection_y_2 < max_lane_line_intersection:
                good_left_fit = False
                good_right_fit = False

    if good_left_fit:
        # Smoothening
        if left_lane_history.best_fit is not None:
            left_lane_history.current_fit = 0.8 * left_lane_history.current_fit + 0.2 * left_lane_history.best_fit
        left_lane_history.best_fit = left_lane_history.current_fit
        left_lane_history.bad_fits = 0
        left_lane_history.bottom_line_intersection = int(left_lane_history.current_fit[0] * 719 ** 2
                                                         + left_lane_history.current_fit[1] * 719
                                                         + left_lane_history.current_fit[2])
    else:
        left_lane_history.current_fit = left_lane_history.best_fit
        left_lane_history.bad_fits += 1
        # If the number of bad or no fits exceeds this number, then reset the bottom line intersection
        if left_lane_history.bad_fits >= max_bad_fits_for_lane_histogram:
            left_lane_history.bottom_line_intersection = None
        # If the number of bad or no fits exceeds this number, then the best fit is not applicable anymore
        if left_lane_history.bad_fits >= max_bad_fits_for_non_detection:
            left_lane_history.best_fit = None

    if good_right_fit:
        # Smoothening
        if right_lane_history.best_fit is not None:
            right_lane_history.current_fit = 0.8 * right_lane_history.current_fit + 0.2 * right_lane_history.best_fit
        right_lane_history.best_fit = right_lane_history.current_fit
        right_lane_history.bad_fits = 0
        right_lane_history.bottom_line_intersection = int(right_lane_history.current_fit[0] * 719 ** 2
                                                         + right_lane_history.current_fit[1] * 719
                                                         + right_lane_history.current_fit[2])
    else:
        right_lane_history.current_fit = right_lane_history.best_fit
        right_lane_history.bad_fits += 1
        # If the number of bad or no fits exceeds this number, then reset the bottom line intersection
        if right_lane_history.bad_fits >= max_bad_fits_for_lane_histogram:
            right_lane_history.bottom_line_intersection = None
        # If the number of bad or no fits exceeds this number, then the best fit is not applicable anymore
        if right_lane_history.bad_fits >= max_bad_fits_for_non_detection:
            right_lane_history.best_fit = None

    return left_lane_history, right_lane_history


def draw_lanes_on_image(
        warped_image,
        undistorted_image,
        inverse_perspective_matrix,
        left_lane_history,
        right_lane_history):
    """
    Draws the lane in a transparent green color on the undistorted image using the fitted lane lines in the warped image
    :param warped_image: The warped image
    :param undistorted_image: The original, but undistorted image
    :param inverse_perspective_matrix: The inverse perspective matrix to transform the warped image into the same coordinate system as the undistorted image
    :param left_lane_history: The statistics of the left lane detection
    :param right_lane_history: The statistics of the right lane detection
    :return: The result image
    """
    # Create an image to draw the lines on
    color_warp = np.zeros_like(warped_image).astype(np.uint8)

    if left_lane_history.current_fit is not None and right_lane_history.current_fit is not None:
        ploty = np.linspace(0, 719, num=720)
        left_fitx = np.int_(left_lane_history.current_fit[0] * ploty ** 2 + left_lane_history.current_fit[1] * ploty + left_lane_history.current_fit[2])
        right_fitx = np.int_(right_lane_history.current_fit[0] * ploty ** 2 + right_lane_history.current_fit[1] * ploty + right_lane_history.current_fit[2])

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, np.int_(ploty)]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, np.int_(ploty)])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    new_warp = cv2.warpPerspective(color_warp, inverse_perspective_matrix, (undistorted_image.shape[1], undistorted_image.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_image, 1, new_warp, 0.3, 0)
    return result


def calculate_radius_of_curvature_and_lane_offset(left_lane_history, right_lane_history):
    """
    Calculates the radius of curvature of the left and right lane lines and
    the offset of the image towards the lane center
    :param left_lane_history: The statistics of the left lane detection
    :param right_lane_history: The statistics of the right lane detection
    :return: The radius of curvature of the left and right lane lines and the lane offset
    """
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    plot_y = np.linspace(0, 719, num=720)
    left_fit_x = left_lane_history.current_fit[0] * plot_y ** 2 \
                 + left_lane_history.current_fit[1] * plot_y \
                 + left_lane_history.current_fit[2]
    right_fit_x = right_lane_history.current_fit[0] * plot_y ** 2 \
                  + right_lane_history.current_fit[1] * plot_y \
                  + right_lane_history.current_fit[2]
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(plot_y * ym_per_pix, left_fit_x * xm_per_pix, 2)
    right_fit_cr = np.polyfit(plot_y * ym_per_pix, right_fit_x * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * 720 * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * 720 * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # Calculate the lane offset
    middle_lane = (left_fit_x[719] + right_fit_x[719]) * xm_per_pix / 2
    middle_image = 640 * xm_per_pix
    lane_offset = middle_lane - middle_image

    return left_curverad, right_curverad, lane_offset


def process_frame(image, camera_parameters, filter_output_folder=None, lane_detection_folder=None):
    """
    Processes a single image
    :param image: The original image
    :param camera_parameters: CameraParameters instance containing the perspective matrices and intrinsic parameters
    :param filter_output_folder: Path to write results of the filter stage
    :param lane_detection_folder: Path to write results of the lane line detection stage
    :return: The image with the detected lane projected on it, together with added statistics
    """
    filter_output_name = None
    if filter_output_folder is not None:
        filter_output_name = os.path.join(filter_output_folder, "filter" + str(process_frame.counter) + ".jpg")

    lane_detection_output_name = None
    if lane_detection_folder is not None:
        lane_detection_output_name = os.path.join(lane_detection_folder, "lane" + str(process_frame.counter) + ".jpg")
    process_frame.counter += 1

    undistorted_image = cv2.undistort(
        image,
        camera_parameters.intrinsic_params,
        camera_parameters.dist_coeffs)
    # http://tanbakuchi.com/posts/comparison-of-openv-interpolation-algorithms/ : sharpest edges with INTER_LANCZOS4
    warped_image = cv2.warpPerspective(
        src=undistorted_image,
        M=camera_parameters.perspective_matrix,
        dsize=image.shape[1::-1],
        flags=cv2.INTER_LANCZOS4)
    filtered_image = filter_image(
        orig_img=undistorted_image,
        warped_image=warped_image,
        show_results=False,
        results_output_file=filter_output_name)
    left_lane_history, right_lane_history, fig = \
        fit_polynomial_sliding_window(
            binary_warped=filtered_image,
            left_lane_history=process_frame.lane_history.left_lane,
            right_lane_history=process_frame.lane_history.right_lane)
    if lane_detection_output_name is not None:
        fig.savefig(lane_detection_output_name)
    left_lane_history, right_lane_history = process_lanes(
        left_lane_history=left_lane_history,
        right_lane_history=right_lane_history)
    result_img = draw_lanes_on_image(
        warped_image=warped_image,
        undistorted_image=undistorted_image,
        inverse_perspective_matrix=camera_parameters.inverse_perspective_matrix,
        left_lane_history=left_lane_history,
        right_lane_history=right_lane_history)
    if left_lane_history.current_fit is not None and right_lane_history.current_fit is not None:
        left_curverad, right_curverad, lane_offset = \
            calculate_radius_of_curvature_and_lane_offset(left_lane_history, right_lane_history)
        average_curverad = (left_curverad + right_curverad) / 2.0
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result_img,
                    'Radius of curvature:  %.4f' % average_curverad,
                    (40, 40), font, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        if lane_offset < 0:
            car_position = "%.2f m left" % -lane_offset
        else:
            car_position = "%.2f m right" % lane_offset
        cv2.putText(result_img,
                    'Vehicle is %s of center:' % car_position,
                    (40, 70), font, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    process_frame.lane_history.left_lane = left_lane_history
    process_frame.lane_history.right_lane = right_lane_history
    return result_img


process_frame.counter = 0
process_frame.lane_history = LaneHistory()

if __name__ == "__main__":
    # Load the intrinsic parameters of the camera
    intrinsic_params, dist_coeffs = load_calibration_params("camera_cal/calibration_data.p")

    # Load the pre-calibrated perspective matrix
    perspective_matrix, inverse_perspective_matrix = load_perspective_matrices(
        "camera_cal/perspective_matrix.p")

    cam_params = CameraParameters(intrinsic_params, dist_coeffs, perspective_matrix, inverse_perspective_matrix)

    left_lane_history, right_lane_history = None, None
    # for test_img in glob.glob('video_images/challenge/challenge-*.jpg'):
    for test_img in glob.glob('test_images/test*.jpg*'):
        orig_img = cv2.cvtColor(cv2.imread(test_img), cv2.COLOR_BGR2RGB)
        process_frame(orig_img, camera_parameters=cam_params)
        _ = plt.show(block=True)
