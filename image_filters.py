import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from camera_calibration import load_calibration_params
from unwarp import load_perspective_matrices
from sliding_windows_fit import fit_polynomial_sliding_window

def color_filter(gray_img, lower_limit, upper_limit):
    """
    Filter a single color channel image according to its color values
    :param gray_img:  A single color channel image
    :param lower_limit: The lower limit for the single color channel
    :param upper_limit: The upper limit for the single color channel
    :return: Binary masked image
    """
    mask = np.zeros_like(gray_img)
    mask[(gray_img > lower_limit) & (gray_img <= upper_limit)] = 1
    return mask


def color_filter3(img, lower_limits, upper_limits):
    """
    Filter a three color channel image according to its color values
    :param img:  A three color channel image
    :param lower_limits: A tuple containing the lower limits for the respective color channels
    :param upper_limits: A tuple containing the upper limits for the respective color channels
    :return: Binary masked image
    """
    channel1 = img[:,:,0]
    channel2 = img[:,:,1]
    channel3 = img[:,:,2]
    mask = np.zeros_like(channel1)
    mask[(channel1 > lower_limits[0]) & (channel1 <= upper_limits[0]) &
           (channel2 > lower_limits[1]) & (channel2 <= upper_limits[1]) &
           (channel3 > lower_limits[2]) & (channel3 <= upper_limits[2])] = 1
    return mask


def gradient_filter(
        gray_img, sobel_kernel=9, gradient_thresh=(0, np.pi / 2), norm_threshold=1000, morph_kernel_size=None):
    """
    Filter the grayscale image according to its gradients
    :param gray_img: The grayscale image
    :param sobel_kernel: Size of the Sobel kernel to perform edge detection
    :param gradient_thresh: Tuple containing the min and max. angles for the gradients
    :param norm_threshold: Minimum gradient norm
    :param morph_kernel_size: Kernel size of the morphological opening operation. If None, no morphological operation is applied.
    :return: Binary masked image
    """
    # Absolute value of the x and y gradients
    x_gradient = np.abs(cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    y_gradient = np.abs(cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Calculate the direction of the gradient
    angles = np.arctan2(y_gradient, x_gradient)
    norm = np.sqrt(y_gradient ** 2 + x_gradient ** 2)
    scale_factor = np.max(norm)/255
    norm = (norm/scale_factor).astype(np.uint8)
    # Create a binary mask where direction and norm thresholds are met
    mask = np.zeros_like(gray_img)
    mask[(angles > gradient_thresh[0]) & (angles < gradient_thresh[1]) & (norm > norm_threshold)] = 1
    # Apply morphological operations dilation and erosion if requested
    if morph_kernel_size:
        open_kernel = np.ones((3, 3), np.uint8)
        close_kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=close_kernel)
        # mask = cv2.dilate(mask, kernel=kernel)
        # mask = cv2.erode(mask, kernel=kernel)
    return mask


def filter_image(orig_img, warped_image, show_results = False, results_output_file = None):
    """
    Applies gradient and color filters on the warped image to find good candidate points for lane lines
    :param orig_img: The original image
    :param warped_image: The warped image
    :param show_results: Display the result if True.
    :param results_output_file: The file to write all the intermediate results and the final result image
                                of the filtering to
    :return: Returns the filtered image
    """
    hsv_img = cv2.cvtColor(warped_image, cv2.COLOR_RGB2HSV)
    saturation_channel = hsv_img[:, :, 1]

    hls_img = cv2.cvtColor(warped_image, cv2.COLOR_RGB2HLS)
    saturation_channel2 = hls_img[:, :, 2]

    red_channel = warped_image[:, :, 0]

    hue_channel = hls_img[:, :, 0]

    lightness_channel = hls_img[:, :, 1]

    # https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    equalized_channel = clahe.apply(saturation_channel)

    equalized_channel2 = clahe.apply(saturation_channel2)

    equalized_red_channel = clahe.apply(red_channel)

    mask_img = gradient_filter(
        gray_img=equalized_channel,
        sobel_kernel=15,
        gradient_thresh=(0.0, 1.0),
        norm_threshold=30,
        morph_kernel_size=11)

    mask_img2 = gradient_filter(
        gray_img=equalized_channel2,
        sobel_kernel=15,
        gradient_thresh=(0.0, 1.0),
        norm_threshold=30,
        morph_kernel_size=None)

    mask_img3 = gradient_filter(
        gray_img=equalized_red_channel,
        sobel_kernel=15,
        gradient_thresh=(0.0, 1.0),
        norm_threshold=30,
        morph_kernel_size=11)

    hls_img2 = hls_img
    hls_img2[:, :, 2] = equalized_channel2

    mask_img4 = color_filter3(
        hls_img2, lower_limits=(0, 80, 150), upper_limits=(40, 255, 255))

    mask_img5 = color_filter(
        equalized_red_channel, lower_limit=210, upper_limit=255)

    mask_img6 = color_filter(
        red_channel, lower_limit=210, upper_limit=255)

    gradient_combined = np.zeros_like(mask_img)
    gradient_combined[(mask_img == 1) | (mask_img2 == 1) | (mask_img3 == 1)] = 1

    color_combined = np.zeros_like(mask_img5)
    color_combined[(mask_img4 == 1) | (mask_img5 == 1)] = 1

    if len(color_combined.nonzero()[0]) < 20000:
        all_combined = color_combined
    else:
        all_combined = np.zeros_like(mask_img5)
        all_combined[(color_combined == 1) & (gradient_combined == 1)] = 1

    if show_results or results_output_file is not None:
        fig, axes = plt.subplots(6, 5)
        for ax in axes.flatten():
            ax.axis('off')
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        axes[0][2].imshow(orig_img)
        axes[0][2].set_title("Original image")

        axes[1][2].imshow(warped_image)
        axes[1][2].set_title("Warped channel")

        axes[2][0].imshow(saturation_channel, cmap='gray')
        axes[2][0].set_title("S channel (from HSV)")

        axes[2][1].imshow(saturation_channel2, cmap='gray')
        axes[2][1].set_title("S channel (from HLS)")

        axes[2][2].imshow(red_channel, cmap='gray')
        axes[2][2].set_title("Red channel")

        axes[2][3].imshow(hue_channel, cmap='gray')
        axes[2][3].set_title("H channel")

        axes[2][4].imshow(lightness_channel, cmap='gray')
        axes[2][4].set_title("L channel")

        axes[3][0].imshow(equalized_channel, cmap='gray')
        axes[3][0].set_title("Equalized S channe (from HSV)")

        axes[3][1].imshow(equalized_channel2, cmap='gray')
        axes[3][1].set_title("Equalized S channel (from HLS)")

        axes[3][2].imshow(equalized_red_channel, cmap='gray')
        axes[3][2].set_title("Equalized R channel")

        axes[3][4].imshow(mask_img6, cmap='gray')
        axes[3][4].set_title("Binary masked unequalized R channel")

        axes[4][0].imshow(mask_img, cmap='gray')
        axes[4][0].set_title("Binary masked S edges (from HSV)")

        axes[4][1].imshow(mask_img2, cmap='gray')
        axes[4][1].set_title("Binary masked S edges (from HLS)")

        axes[4][2].imshow(mask_img3, cmap='gray')
        axes[4][2].set_title("Binary masked R edges")

        axes[4][3].imshow(mask_img4, cmap='gray')
        axes[4][3].set_title("Binary masked HLS lanes")

        axes[4][4].imshow(mask_img5, cmap='gray')
        axes[4][4].set_title("Binary masked red channel lanes")

        axes[5][1].imshow(gradient_combined, cmap='gray')
        axes[5][1].set_title("Gradient masks combined")

        axes[5][2].imshow(color_combined, cmap='gray')
        axes[5][2].set_title("Color masks combined")

        axes[5][3].imshow(all_combined, cmap='gray')
        axes[5][3].set_title("All masks combined")

        if results_output_file is not None:
            plt.savefig(results_output_file)

        if show_results:
            _ = plt.show(block=False)
        else:
            plt.close(fig)

    return all_combined


def show_images_with_filter(intrinsic_params, dist_coeffs, perspective_matrix):
    """
    Helper function to investigate the filtering algorithm on a series of test images
    :param intrinsic_params: Camera intrinsic parameters
    :param dist_coeffs: Distortion coefficients
    :param perspective_matrix: Perspective matrix
    :return: Nothing
    """
    left_lane_history, right_lane_history = None, None
    # for test_img in glob.glob('video_images/challenge/challenge-*.jpg'):
    for test_img in glob.glob('test_images/test*.jpg*'):
        orig_img = cv2.cvtColor(cv2.imread(test_img), cv2.COLOR_BGR2RGB)
        undistorted_image = cv2.undistort(
            orig_img,
            intrinsic_params, dist_coeffs)
        # http://tanbakuchi.com/posts/comparison-of-openv-interpolation-algorithms/ : sharpest edges with INTER_LANCZOS4
        warped_image = cv2.warpPerspective(
            src=undistorted_image, M=perspective_matrix, dsize=orig_img.shape[1::-1], flags=cv2.INTER_LANCZOS4)
        filtered_image = filter_image(orig_img=undistorted_image, warped_image=warped_image, show_results=True)
        left_lane_history, right_lane_history, fig = \
            fit_polynomial_sliding_window(filtered_image, left_lane_history, right_lane_history)
        _ = plt.show(block=True)


if __name__ == "__main__":
    # Load the intrinsic parameters of the camera
    intrinsic_params, dist_coeffs = load_calibration_params("camera_cal/calibration_data.p")

    # Load the pre-calibrated perspective matrix
    perspective_matrix, inverse_perspective_matrix = load_perspective_matrices(
        "camera_cal/perspective_matrix.p")

    show_images_with_filter(intrinsic_params, dist_coeffs, perspective_matrix)