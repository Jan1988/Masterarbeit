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

    roi = frame[y:y + roi_height, x:x + roi_width].copy()
    cv2.rectangle(frame, (x, y), (x + roi_width, y + roi_height), (0, 255, 0), 1)

    return roi, frame




if __name__ == '__main__':

    filename = '00073.MTS'
    out_filename = 'new_00073.avi'

    input_path = os.path.join('assets', filename)
    output_path = os.path.join('assets', out_filename)
    # path = "assets\\ROIs\\Original\\" + filename

    vid_frames, fps = load_video(input_path)
    fourcc = cv2.VideoWriter_fourcc('L', 'A', 'G', 'S')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1080, 1920))

    # zum Verkürzen des Videos
    # [x:] = remove from beginning
    cutted_frames = vid_frames[400:808]



    for frame in cutted_frames:

        # # Show results
        # cv2.imshow('Frames', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # write the frame
        out.write(frame)

    # np.save(output_path, cutted_frames)

    out.release()
    cv2.destroyAllWindows()
    print('Video written to ' + output_path)
