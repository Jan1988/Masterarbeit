import cv2
import numpy as np
import os
import scipy.fftpack


def load_video(video_filename):
    """Load a video into a numpy array"""
    print("Loading " + video_filename)
    if not os.path.isfile(video_filename):
        raise Exception("File Not Found: %s" % video_filename)
    # noinspection PyArgumentList
    capture = cv2.VideoCapture(video_filename)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Frame Count: %i" % frame_count)
    width, height = get_capture_dimensions(capture)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    x = 0
    vid_frames = np.zeros((frame_count, height, width, 3), dtype='uint8')

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (width, height))
        vid_frames[x] = resized_frame
        x += 1

    # Release everything if job is finished
    capture.release()

    return vid_frames, fps


def get_capture_dimensions(capture):
    """Get the dimensions of a capture"""
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("width % i height % i" % (width, height))
    return width, height


def get_frame_dimensions(frame):
    """Get the dimensions of a single frame"""
    height, width = frame.shape[:2]
    return width, height


def split_into_rgb_channels(image):
    '''Split the target image into its red, green and blue channels.
    image - a numpy array of shape (rows, columns, 3).
    output - three numpy arrays of shape (rows, columns) and dtype same as
             image, containing the corresponding channels.
    '''

    red = image[:, :, 2]
    green = image[:, :, 1]
    blue = image[:, :, 0]
    return red, green, blue


def load_video_float(video_filename):
    vid_data, fps = load_video(video_filename)
    return uint8_to_float(vid_data), fps


def uint8_to_float(img):
    result = np.ndarray(shape=img.shape, dtype='float')
    result[:] = img * (1. / 255)
    return result


def temporal_bandpass_filter(data, fps, freq_min=0.833, freq_max=1, axis=0, amplification_factor=1):
    print("Applying bandpass between " + str(freq_min) + " and " + str(freq_max) + " Hz")
    fft = scipy.fftpack.rfft(data, axis=axis)
    frequencies = scipy.fftpack.fftfreq(data.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - freq_min)).argmin()
    bound_high = (np.abs(frequencies - freq_max)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0

    result = np.ndarray(shape=data.shape, dtype='float')
    result[:] = scipy.fftpack.ifft(fft, axis=0)
    result *= amplification_factor
    return result

