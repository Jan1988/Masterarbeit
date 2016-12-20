# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

from Video_Tools import get_frame_dimensions
from Video_Tools import load_video


def devide_image_into_rois(image, div_width, div_height):
    width, height = get_frame_dimensions(image)
    roi_width = int(width / div_width)
    roi_height = int(height / div_height)
    roi_count = div_width * div_height

    roi_frames = np.zeros((roi_count, roi_height, roi_width, 3), dtype='uint8')

    i = 0
    if height % div_height == 0 and width % div_width == 0:
        for x in range(0, width, roi_width):
            for y in range(0, height, roi_height):

                cv2.rectangle(image, (x, y), (x + roi_width, y + roi_height), (0, 0, 0), 1)

                roi = image[y:y + roi_height, x:x + roi_width]
                roi_frames[i] = roi
                i += 1
    else:
        print("please use another divisor (%f, %f)" % (roi_width, roi_height))

        # Ungeradere Rest wird abgeschnitten
        width = roi_width * div_width
        height = roi_height * div_height
        for x in range(0, width, roi_width):
            for y in range(0, height, roi_height):

                cv2.rectangle(image, (x, y), (x + roi_width, y + roi_height), (0, 0, 0), 1)

                roi = image[y:y + roi_height, x:x + roi_width]
                roi_frames[i] = roi
                i += 1

    return roi_frames, image


# Get region of interest of an image at a desired position
# top left point of roi is (x, y)
def get_roi_frame(frame, y, x, roi_height, roi_width):

    cv2.rectangle(frame, (x, y), (x + roi_width, y + roi_height), (0, 0, 0), 1)
    roi = frame[y:y + roi_height, x:x + roi_width]

    return roi, frame


if __name__ == '__main__':

    filename = '00101.mp4'

    path = os.path.join('assets', 'Videos (original)', filename)
    path = "assets\\ROIs\\Original\\" + filename
    print(path)

    vid_frames, fps = load_video(path)
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    out = cv2.VideoWriter('assets\\ROIs\\ROI_' + filename, fourcc, 30.0, (160, 90))

    # zum Verkürzen des Videos
    # cutted_frames = vid_frames[0:(vid_frames.shape[0]/2)]

    for frame in vid_frames:
        roi, frame_with_rect = get_roi_frame(frame, 170, 520, 90, 160)

        # Show results
        cv2.imshow('Frames', frame_with_rect)
        cv2.imshow('ROI', roi)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # write the frame
        out.write(roi)

    out.release()
    cv2.destroyAllWindows()
