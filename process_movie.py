from moviepy.editor import VideoFileClip

from camera_calibration import load_calibration_params
from unwarp import load_perspective_matrices
from process_frame import process_frame
from camera_parameters import CameraParameters

def process_road_movie(input_file, output_file, camera_parameters, output_folder=None, time_frame=None):
    if time_frame is not None:
        clip = VideoFileClip(input_file).subclip(time_frame[0], time_frame[1])
    else:
        clip = VideoFileClip(input_file)
    lane_clip = clip.fl_image(
        lambda image: process_frame(image, camera_parameters, output_folder, output_folder))
    lane_clip.write_videofile(output_file, audio=False)


if __name__ == "__main__":
    # Load the intrinsic parameters of the camera
    intrinsic_params, dist_coeffs = load_calibration_params("camera_cal/calibration_data.p")

    # Load the pre-calibrated perspective matrix
    perspective_matrix, inverse_perspective_matrix = load_perspective_matrices(
        "camera_cal/perspective_matrix.p")

    cam_params = CameraParameters(intrinsic_params, dist_coeffs, perspective_matrix, inverse_perspective_matrix)
    process_road_movie(
        input_file="project_video.mp4",
        output_file="project_video_with_filtered_lanes.mp4",
        camera_parameters=cam_params,
        output_folder="output_images/project",
        time_frame=None)