import os
import cv2
import numpy as np

from CHROM_Method_Single_Vid import chrom_based_pulse_signal_estimation
from Video_Tools import devide_frame_into_roi_means, split_vid_into_rgb_channels
from Video_Tools import load_video
from matplotlib import pyplot as plt


dir_path = os.path.join('assets', 'Vid_Original')
file = '00113.MTS'
file_path = os.path.join(dir_path, file)

video_frames, fps = load_video(file_path)
video_frames = video_frames[1:250]

face_cascade_path = os.path.join('C:/', 'Anaconda3', 'pkgs', 'opencv3-3.1.0-py35_0', 'Library', 'etc', 'haarcascades', 'haarcascade_frontalface_default.xml')


face_cascade = cv2.CascadeClassifier(face_cascade_path)

means = np.zeros((video_frames.shape[0], 3), dtype='float64')

for j, frame in enumerate(video_frames):
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

    blurred_roi = cv2.blur(roi_face, (5, 5))

    # Create time series array of the roi means
    means[j] = np.mean(blurred_roi, axis=(0, 1))

    cv2.imshow('frame', blurred_roi)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

blue_channel = means[:, 0]
green_channel = means[:, 1]
red_channel = means[:, 2]

bpm, heart_rates, fft, hann_S, S = chrom_based_pulse_signal_estimation(fps, red_channel, green_channel, blue_channel)
print(bpm)


fig = plt.figure(figsize=(17, 9))
fig.suptitle(file + ' - BPM: ' + str(bpm), fontsize=14, fontweight='bold')

sub1 = fig.add_subplot(221)
sub1.set_title('Norm. Avg.')
sub1.plot(green_channel, 'g')

sub2 = fig.add_subplot(222)
sub2.set_title('S Signal')
sub2.plot(S, 'm',)

sub3 = fig.add_subplot(223)
sub3.set_title('hann_S-Signal')
sub3.plot(hann_S, 'k')

sub4 = fig.add_subplot(224)
sub4.set_title('fft')
sub4.plot(heart_rates, fft, 'k')

plt.show()


