from moviepy.editor import VideoFileClip
import cv2


def save_clip(clip, dst_prefix):
    idx = 0
    for frame in clip.iter_frames():
        dst = dst_prefix + str(idx) + ".jpg"
        bgr_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dst, bgr_image)
        idx += 1


clip1 = VideoFileClip("harder_challenge_video.mp4")
save_clip(clip1, "./video_images/harder_challenge/hard-")

clip2 = VideoFileClip("challenge_video.mp4")
save_clip(clip2, "./video_images/challenge/challenge-")

clip3 = VideoFileClip("project_video.mp4")
save_clip(clip3, "./video_images/project_video/project-")

