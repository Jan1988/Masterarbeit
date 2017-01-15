# import the necessary packages
# from pyimagesearch import imutils
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os

# construct the argument parse and parse the arguments
from Video_Tools import load_video, get_video_dimensions

dir_path = os.path.join('assets', 'Vid_Original', 'Kuenstliches_Licht')
# dir_path = os.path.join('assets', 'Vid_Original')
file = '00130.MTS'
file_path = os.path.join(dir_path, file)


# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
lower = np.array([0, 120, 100], dtype="uint8")
upper = np.array([20, 255, 255], dtype="uint8")

video_frames, fps = load_video(file_path)
frame_count, width, height = get_video_dimensions(video_frames)

skin_arr = np.ones([frame_count, height, width])

# keep looping over the frames in the video
for i, frame in enumerate(video_frames):
    # grab the current frame
    # (grabbed, frame) = camera.read()

    # if we are viewing a video and we did not grab a
    # frame, then we have reached the end of the video

    # cv2.rectangle(frame, (800, 430), (810, 450), (0, 0, 0), 2)

    # frame[430:450, 800:820])

    # print(np.mean(frame[430:450, 800:820], axis=(0, 1)))


    # resize the frame, convert it to the HSV color space,
    # and determine the HSV pixel intensities that fall into
    # the speicifed upper and lower boundaries
    # frame = imutils.resize(frame, width=400)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)


    # show the skin in the image along with the mask
    # cv2.imshow("images", np.hstack([frame, skin]))
    # cv2.imshow("images", skin)

    # Where values are low
    mean_skin = np.mean(skin, axis=2)
    low_values_indices = mean_skin < 1
    skin_arr[i, low_values_indices] = 0

    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cleanup the camera and close any open windows

final_mean_skin = np.mean(skin_arr, axis=0)

fig = plt.figure(figsize=(17, 9))
sub1 = fig.add_subplot(111)
sub1.set_title('Norm. Avg.')
sub1.imshow(final_mean_skin)
plt.show()

cv2.destroyAllWindows()
