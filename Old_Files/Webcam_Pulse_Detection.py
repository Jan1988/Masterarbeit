import cv2
import numpy as np
import scipy.fftpack
import scipy.signal

from matplotlib import pyplot
from sympy import Point

from Video_Tools import get_frame_dimensions
from Video_Tools import load_video


def devide_image_into_rois(image, width_divisor, height_divisor):
    width, height = get_frame_dimensions(image)
    roi_width = int(width / width_divisor)
    roi_height = int(height / height_divisor)
    roi_count = width_divisor * height_divisor

    roi_frames = np.zeros((roi_count, roi_height, roi_width, 3), dtype='uint8')

    i = 0
    if height % height_divisor == 0 and width % width_divisor == 0:
        for x in range(0, width, roi_width):
            for y in range(0, height, roi_height):

                cv2.rectangle(image, (x, y), (x + roi_width, y + roi_height), (0, 0, 0), 1)

                roi = image[y:y + roi_height, x:x + roi_width]
                roi_frames[i] = roi
                i += 1
    else:
        print("please use another divisor (%f, %f)" % (roi_width, roi_height))

    return roi_frames, image


def get_roi_means(roi_frame):

    v1 = np.mean(roi_frame[:, :, 0])
    v2 = np.mean(roi_frame[:, :, 1])
    v3 = np.mean(roi_frame[:, :, 2])

    return (v1 + v2 + v3) / 3.


def temporal_bandpass_filter(data, fps, freq_min=0.833, freq_max=2.0, axis=0, amplification_factor=1):
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


if __name__ == '__main__':

    filename = 'assets\\output_1.mp4'
    vid_data, fps = load_video(filename)
    """Graph the average value of the video as well as the frequency strength"""
    averages = []

    height, width = vid_data.shape[1:3]
    for frame in vid_data:

        gray_frames = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blured_frame = cv2.blur(gray_frames, (5, 5))
        center = (int(width/2), int(height/2))
        print(center)
        cv2.circle(blured_frame, (0, 0), 1, (0, 0, 255), -1)
        cv2.imshow("blured_frame", blured_frame)
        bandpassed = temporal_bandpass_filter(blured_frame, fps)
        averages.append(bandpassed[(0, 0)])

    # averages = averages - min(averages)

    charts_x = 1
    charts_y = 2
    pyplot.figure(figsize=(20, 10))
    pyplot.subplots_adjust(hspace=.7)

    pyplot.subplot(charts_y, charts_x, 1)
    pyplot.title("Pixel Average")
    pyplot.xlabel("Time")
    pyplot.ylabel("Brightness")
    pyplot.plot(averages)

    freqs = scipy.fftpack.fftfreq(len(averages), d=1.0 / fps)
    fft = abs(scipy.fftpack.fft(averages))
    idx = np.argsort(freqs)

    pyplot.subplot(charts_y, charts_x, 2)
    pyplot.title("FFT")
    pyplot.xlabel("Freq (Hz)")
    freqs = freqs[idx]
    fft = fft[idx]

    freqs = freqs[len(freqs) / 2 + 1:]
    fft = fft[len(fft) / 2 + 1:]
    pyplot.plot(freqs, abs(fft))

    pyplot.show()
