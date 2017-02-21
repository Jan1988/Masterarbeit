import cv2
import numpy as np
import os


"""Load a video into a numpy array"""


def load_video(video_filename):

    print("Loading " + video_filename)
    if not os.path.isfile(video_filename):
        raise Exception("File Not Found: %s" % video_filename)
    # noinspection PyArgumentList
    capture = cv2.VideoCapture(video_filename)

    # Constant = 7
    frame_count = int(capture.get(7))
    # # OpenCV on Ubuntu not working correctly, have to set it manually:
    # frame_count = int(frame_count/2)

    print("Frame Count: %i" % frame_count)
    width, height = get_capture_dimensions(capture)

    # Constant = 5
    fps = int(capture.get(5))
    # OpenCV on Ubuntu not working correctly, have to set it manually:
    fps = 25
    print("fps: %i" % fps)

    x = 1
    vid_frames = np.zeros((frame_count, height, width, 3), dtype='uint8')

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (width, height))
        vid_frames[x] = resized_frame
        x += 1
        if x >= frame_count:
            break

    # Release everything if job is finished
    capture.release()

    return vid_frames, fps


def get_capture_dimensions(capture):
    """Get the dimensions of a capture"""
    # Constant = 3
    width = int(capture.get(3))
    # Constant = 4
    height = int(capture.get(4))
    print("width % i height % i" % (width, height))
    return width, height


def get_video_dimensions(frame):
    """Get the dimensions of a video"""
    length, height, width = frame.shape[:3]
    return length, width, height


def get_frame_dimensions(frame):
    """Get the dimensions of a frame"""
    height, width = frame.shape[:2]
    return width, height



