import numpy as np


class LaneLineHistory:
    """
    Statistics/history of the detection algorithm of a single lane line
    """
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        # X position where the fit curve intersects with the bottom line
        self.bottom_line_intersection = None
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # number of subsequent bad or no fits
        self.bad_fits = 0

    def reset_for_detection(self):
        """
        Reset the statistics/history to prepare for a new cycle of the detection algorithm
        :return:
        """
        self.detection_windows = 0
        self.detected = False
        self.recent_xfitted = []
        self.current_fit = [np.array([False])]
        self.radius_of_curvature = None
        self.allx = None
        self.ally = None
        self.line_base_pos = None


class LaneHistory:
    """
    Statistics/history of the detection algorithm of both lane lines
    """
    def __init__(self):
        self.left_lane = LaneLineHistory()
        self.right_lane = LaneLineHistory()
