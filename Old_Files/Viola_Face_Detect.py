import os
import cv2
import numpy as np

from matplotlib import pyplot as plt
from POS_Based_Method import pos_based_method
from Video_Tools import load_video


def plot_results(green_norm, pulse_signal, overlap_signal, raw, fft, heart_rates):
    # plt.axis([0, n, y_lower, y_upper])

    fig = plt.figure(figsize=(17, 9))

    sub1 = fig.add_subplot(331)
    sub1.set_title('Norm. Avg.')
    # sub1.plot(int_frames, red_norm, 'r',
    #           int_frames, green_norm, 'g',
    #           int_frames, blue_norm, 'b')
    sub1.plot(green_norm, 'g')

    sub2 = fig.add_subplot(332)
    sub2.set_title('Pulse Signal')
    sub2.plot(pulse_signal, 'k')

    sub5 = fig.add_subplot(333)
    sub5.set_title('Overlap-added Signal')
    sub5.plot(overlap_signal, 'k')

    sub8 = fig.add_subplot(334)
    sub8.set_title('Hanning Window')
    sub8.plot(raw, 'k')

    sub7 = fig.add_subplot(335)
    sub7.set_title('FFT abs')
    sub7.plot(heart_rates, fft, 'k')

    plt.show()


def viola(frame):
    frame_clone = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in faces:
        x2 = int(x * 1.2)
        y2 = int(y * 1.1)
        w2 = int(w * 0.75)
        h2 = int(h * 0.8)
        cv2.rectangle(frame, (x2, y2), (x + w2, y2 + h2), (255, 0, 0), 2)
        roi_face = frame_clone[y2:y2 + h2, x2:x + w2]
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # roi_face = frame_clone[y:y + h, x:x + w]
        # roi_forehead = roi_face[:120, int(x/3):400]
    return roi_face


if __name__ == '__main__':

    face_cascade_path = os.path.join('C:/', 'Anaconda3', 'pkgs', 'opencv3-3.1.0-py35_0', 'Library', 'etc', 'haarcascades', 'haarcascade_frontalface_default.xml')
    dir_path = os.path.join('assets', 'Vid_Original')
    file = '00112.MTS'
    file_path = os.path.join(dir_path, file)

    window_numbers = 6
    window_size = 40
    frame_count = window_numbers * window_size + 1

    video_frames, fps = load_video(file_path)
    video_frames = video_frames[1:frame_count]
    print('Reduced Frame Count: ' + str(len(video_frames)))

    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Create time series array of the roi means
    viola_roi_sequence = []

    for j, frame in enumerate(video_frames):

        roi_face = viola(frame)
        viola_roi_sequence.append(roi_face)
        # blurred_roi = cv2.blur(roi_face, (5, 5))

        # cv2.imshow('blurred_roi', roi_face)

    bpm, fft, heart_rates, raw_fft, H, pulse_signal, green_avg = pos_based_method(viola_roi_sequence, fps)
    plot_results(green_avg,  pulse_signal, H, raw_fft, fft, heart_rates)

    print(bpm)



