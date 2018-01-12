class CameraParameters:
    """
    Camera parameters containing the intrinsic camera parameters (focus, field of view, distortion coefficients)
    and the perspective matrix (and its inverse)
    """
    def __init__(self, intrinsic_params, dist_coeffs, perspective_matrix, inverse_perspective_matrix):
        self.intrinsic_params = intrinsic_params
        self.dist_coeffs = dist_coeffs
        self.perspective_matrix = perspective_matrix
        self.inverse_perspective_matrix = inverse_perspective_matrix
