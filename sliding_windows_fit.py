import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from lane_line_history import LaneLineHistory


# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
def draw_vector(v0, v1, ax=None):
    """
    Draw the vector of the principal component
    :param v0: Center of the point cloud
    :param v1: Direction and relative magnitude of the principal component
    :param ax: Plot to draw to
    :return:
    """
    ax = ax or plt.gca()
    arrow_props = dict(
        arrowstyle='->',
        linewidth=2,
        shrinkA=0,
        shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrow_props)


def find_pca(points, show_results = False):
    """
    Find first principal component of the point cloud
    :param points: An array of points with size n x 2 where n = number of points
    :param show_results: Show the principal component vector on top of the point cloud
    :return: Mean of the principal component, the vector, the ratio of the variance explained by the 1st principal component
    """
    pca = PCA(n_components=2, whiten=True)
    pca.fit(points)
    length1 = pca.explained_variance_[0]
    vector1 = pca.components_[0, :]
    length2 = pca.explained_variance_[1]
    vector2 = pca.components_[1, :]
    if show_results:
        v1 = vector1 * np.sqrt(length1)
        v2 = vector2 * np.sqrt(length2)
        fig = plt.figure()
        plt.scatter(points[:, 0], points[:, 1])
        draw_vector(pca.mean_, pca.mean_ + v1)
        draw_vector(pca.mean_, pca.mean_ + v2)
        plt.axis('equal');
        fig.show()
    return pca.mean_, vector1, pca.explained_variance_ratio_[0]


def fit_polynomial_sliding_window(
        binary_warped,
        left_lane_history=None,
        right_lane_history=None):
    """
    Searches for candidate points that belong to a lane line and then fits a second degree polynomial
    :param binary_warped: Binary warped image (already processed by the filtering algorithm)
    :param left_lane_history: Statistics / history of the previous detection cycles of the left lane line
    :param right_lane_history: Statistics / history of the previous detection cycles of the right lane line
    :return: The updated lane line history instances containing new lane line candidates
    """
    if left_lane_history is None:
        left_lane_history = LaneLineHistory()
    if right_lane_history is None:
        right_lane_history = LaneLineHistory()

    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Choose the number of sliding windows
    n_windows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/n_windows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    # Current positions to be updated for each window
    left_x_current = left_lane_history.bottom_line_intersection
    right_x_current = right_lane_history.bottom_line_intersection

    if left_lane_history.bottom_line_intersection is None or right_lane_history.bottom_line_intersection is None:
        # The histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        # Create an output image to draw on and visualize the result
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        if left_lane_history.bottom_line_intersection is None:
            left_x_current = np.argmax(histogram[:midpoint])
            left_lane_history.bottom_line_intersection = left_x_current
        if right_lane_history.bottom_line_intersection is None:
            right_x_current = np.argmax(histogram[midpoint:]) + midpoint
            right_lane_history.bottom_line_intersection = right_x_current

    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum and maximum number of pixels found to perform PCA analysis
    min_pix = 80
    max_pix = 5000
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Slopes of the principal components in the sliding windows
    # previous_slope_right = 1000000
    # previous_slope_left = 1000000
    # Minimum explained variance ration that should be explained by the 1st principal component
    min_pca_explained_variance_ratio = 0.75
    # Minimum slope of the 1st principal component (otherwise probably the wrong principal component is chosen
    # => probably the one perpendicular to it, is the right one)
    min_principal_component_slope = 0.4

    win_y_low = binary_warped.shape[0] - window_height
    win_y_high = binary_warped.shape[0]
    left_lane_history.reset_for_detection()
    right_lane_history.reset_for_detection()
    # Step through the windows one by one
    for window in range(1, n_windows+1):
        # Number of windows that were accepted to add points for fitting
        win_xleft_low = left_x_current - margin
        if win_xleft_low >= 0:
            win_xleft_high = left_x_current + margin
        else:
            win_xleft_high = 0
        win_xright_high = right_x_current + margin
        if win_xright_high <= 1280:
            win_xright_low = right_x_current - margin
        else:
            win_xright_low = 1280
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low,win_y_low), (win_xleft_high,win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low,win_y_low), (win_xright_high,win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the left window
        good_left_inds = \
            ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
            (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        # Filter out the nonzero pixels that are too far away from the X mean in the left window
        left_x_coords = nonzero_x[good_left_inds]
        left_x_mean = np.mean(left_x_coords)
        left_x_std = np.std(left_x_coords)
        good_left_inds = good_left_inds[abs(nonzero_x[good_left_inds] - left_x_mean) < 2 * left_x_std]

        # Identify the nonzero pixels in x and y within the right window
        good_right_inds = \
            ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
            (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
        # Filter out the nonzero pixels that are too far away from the X mean in the right window
        right_x_coords = nonzero_x[good_right_inds]
        right_x_mean = np.mean(right_x_coords)
        right_x_std = np.std(right_x_coords)
        good_right_inds = good_right_inds[abs(nonzero_x[good_right_inds] - right_x_mean) < 2 * right_x_std]
        # If you found > minpix pixels, recenter next window on their mean position
        pca_mean_left = None
        pca_mean_right = None
        if len(good_left_inds) > min_pix and len(good_left_inds) < max_pix:
            pca_mean_left, pca_vector_left, pca_explained_variance_ratio_left = \
                find_pca(np.vstack((nonzero_x[good_left_inds] / 1280, nonzero_y[good_left_inds] / 720)).T)
        if len(good_right_inds) > min_pix and len(good_right_inds) < max_pix:
            pca_mean_right, pca_vector_right, pca_explained_variance_ratio_right = \
                find_pca(np.vstack((nonzero_x[good_right_inds] / 1280, nonzero_y[good_right_inds] / 720)).T)
        # Adapt X center of the next sliding window
        if pca_mean_left is not None:
            if pca_explained_variance_ratio_left > min_pca_explained_variance_ratio:
                # y = mean.y + (pca_vector_left.y / pca_vector_left.x) * (x - mean.x)
                slope_left = (pca_vector_left[1] * 720) / (pca_vector_left[0] * 1280)
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                left_lane_history.detection_windows += 1
                # Adapt next sliding window horizontal position
                if abs(slope_left) >= min_principal_component_slope:
                    left_x_current = int((win_y_low - pca_mean_left[1] * 720) / slope_left + (pca_mean_left[0] * 1280))
            else:
                left_x_current = np.int(np.mean(nonzero_x[good_left_inds]))
        else:
            pass
            # left_x_current += int((win_y_low - win_y_high) / previous_slope_left)
        if pca_mean_right is not None:
            if pca_explained_variance_ratio_right > min_pca_explained_variance_ratio:
                slope_right = (pca_vector_right[1] * 720) / (pca_vector_right[0] * 1280)
                # Append these indices to the lists
                right_lane_inds.append(good_right_inds)
                right_lane_history.detection_windows += 1
                # Adapt next sliding window horizontal position
                if abs(slope_right) >= min_principal_component_slope:
                    right_x_current = int((win_y_low - pca_mean_right[1] * 720) / slope_right + (pca_mean_right[0] * 1280))
            else:
                right_x_current = np.int(np.mean(nonzero_x[good_right_inds]))
        else:
            pass
            # right_x_current += int((win_y_low - win_y_high) / previous_slope_right)
        # Calculate next windows upper and lower boundaries
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height

    if len(left_lane_inds):
        # Concatenate the arrays of indices and extract left and right line pixel positions
        left_lane_inds = np.concatenate(left_lane_inds)
        # Extract left line pixel positions
        left_lane_history.allx = nonzero_x[left_lane_inds]
        left_lane_history.ally = nonzero_y[left_lane_inds]
        if left_lane_history.detection_windows >= 3:
            # Fit a second order polynomial
            left_lane_history.current_fit = np.polyfit(left_lane_history.ally, left_lane_history.allx, 2)
            left_lane_history.detected = True
    if len(right_lane_inds):
        # Concatenate the arrays of indices
        right_lane_inds = np.concatenate(right_lane_inds)
        # Extract right line pixel positions
        right_lane_history.allx = nonzero_x[right_lane_inds]
        right_lane_history.ally = nonzero_y[right_lane_inds]
        if right_lane_history.detection_windows >= 3:
            # Fit a second order polynomial
            right_lane_history.current_fit = np.polyfit(right_lane_history.ally, right_lane_history.allx, 2)
            right_lane_history.detected = True

    fig = plt.figure()

    # Generate x and y values for plotting
    plot_y = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    if left_lane_history.allx is not None:
        out_img[left_lane_history.ally, left_lane_history.allx] = [255, 0, 0]
    if left_lane_history.detected:
        left_fit_x = left_lane_history.current_fit[0] * plot_y ** 2 \
                     + left_lane_history.current_fit[1] * plot_y \
                     + left_lane_history.current_fit[2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(out_img,
                    'Left:  X = %.4f Y^2 + %.4f Y + %.4f' % (left_lane_history.current_fit[0], left_lane_history.current_fit[1], left_lane_history.current_fit[2]),
                    (50, 50), font, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        plt.plot(left_fit_x, plot_y, color='yellow')
    if right_lane_history.allx is not None:
        out_img[right_lane_history.ally, right_lane_history.allx] = [0, 0, 255]
    if right_lane_history.detected:
        right_fit_x = right_lane_history.current_fit[0] * plot_y ** 2 \
                      + right_lane_history.current_fit[1] * plot_y \
                      + right_lane_history.current_fit[2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(out_img,
                    'Right:  X = %.4f Y^2 + %.4f Y + %.4f' % (right_lane_history.current_fit[0], right_lane_history.current_fit[1], right_lane_history.current_fit[2]),
                    (50, 80), font, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        plt.plot(right_fit_x, plot_y, color='yellow')

    plt.imshow(out_img)
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    return left_lane_history, right_lane_history, fig
