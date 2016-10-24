import cv2
import numpy as np

from Video_Tools import get_frame_dimensions
from Video_Tools import load_video


def devide_image_into_roi(image, div_width, div_height):
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

    return roi_frames, image


if __name__ == '__main__':

    filename = 'assets\\output_1.mp4'
    vid_frames, fps = load_video(filename)
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    out = cv2.VideoWriter('assets\\output_2.mp4', fourcc, 30.0, (80, 90))

    cutted_frames = vid_frames[0:(vid_frames.shape[0]/2)]

    for frame in cutted_frames:

        roi_frames, raster_image = devide_image_into_roi(frame, 2, 1)


        # Show results
        cv2.imshow('Frames', raster_image)
        print(roi_frames[1].shape)
        cv2.imshow('ROI', roi_frames[1])
        # write the frame
        out.write(roi_frames[1])


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break