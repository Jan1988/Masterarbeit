import cv2
import numpy as np

from Video_Tools import get_frame_dimensions
from Video_Tools import load_video
from matplotlib import pyplot


def load_image(img_path):
    img = cv2.imread(img_path)
    return (img)


def devide_image_into_roi(image, divisor):
    width, height = get_frame_dimensions(image)
    roi_width = int(width / divisor)
    roi_height = int(height / divisor)
    roi_count = divisor * divisor

    roi_frames = np.zeros((roi_count, roi_height, roi_width, 3), dtype='uint8')

    i = 0
    if height % divisor == 0 and width % divisor == 0:
        for x in range(0, width, roi_width):
            for y in range(0, height, roi_height):

                cv2.rectangle(image, (x, y), (x + roi_width, y + roi_height), (0, 0, 0), 1)

                roi = image[y:y + roi_height, x:x + roi_width]
                roi_frames[i] = roi
                i += 1
    else:
        print("please use another divisor (%f, %f)" % (roi_width, roi_height))

    return roi_frames, image


def show_video_frames(array):
    i = 0
    while True:
        try:
            for vid in array:
                print(i)
                cv2.imshow('Frames', vid[i])
                i += 1
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        except IndexError:
            break


def calculate_mean_over_interval(vid_frames):
    width, height = get_frame_dimensions(vid_frames[0])
    sum_of_frames = np.zeros((height, width, 3), np.float64)
    mean_of_frames = np.zeros((height, width, 3), np.float)

    i = 0
    for frame in vid_frames:
        sum_of_frames = cv2.accumulate(frame, sum_of_frames)
        i += 1
        # print("Sum: " + str(sum_of_frames[0, 0]))

    mean_of_frames = sum_of_frames / i
    return mean_of_frames


def calculate_std_deviation_over_interval(vid_frames):
    width, height = get_frame_dimensions(vid_frames[0])

    blue_vals = []
    green_vals = []
    red_vals = []
    std_deviation_of_frames = np.zeros((height, width, 3), np.float)

    for x in range(0, width):
        for y in range(0, height):
            for frame in vid_frames:

                blue_vals.append(frame[x, y, 0])
                green_vals.append(frame[x, y, 1])
                red_vals.append(frame[x, y, 2])

            std_deviation_of_frames[x, y, 0] = np.std(blue_vals)
            std_deviation_of_frames[x, y, 1] = np.std(green_vals)
            std_deviation_of_frames[x, y, 2] = np.std(red_vals)
        print(x)

    return std_deviation_of_frames


def normalize_frames(stddev_of_frames, mean, frame):

    frame = frame / mean

    return frame


# def calculate_chrom():


if __name__ == '__main__':

    filename = 'assets\\output_1.1.mp4'

    vid_frames, fps, width, height = load_video(filename)

    blue_vals = []
    green_vals = []
    red_vals = []

    mean_of_frames = calculate_mean_over_interval(vid_frames)
    print("Mean: " + str(mean_of_frames[50, 145]))
    stddev_of_frames = calculate_std_deviation_over_interval(vid_frames)
    print(stddev_of_frames)

    for frame in vid_frames:

        # red, green, blue = split_into_rgb_channels(frame)
        # green_image = np.zeros((green.shape[0], green.shape[1], 3), dtype=green.dtype)
        # green_image[:, :, 1] = green

        normalized_frame = normalize_frames(stddev_of_frames, mean_of_frames, frame)

        roi_frames, raster_image = devide_image_into_roi(frame, 2)
        cv2.imshow('Frames', raster_image)

        blue_vals.append(normalized_frame[50, 145, 0])
        green_vals.append(normalized_frame[50, 145, 1])
        red_vals.append(normalized_frame[50, 145, 2])


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




    charts_x = 1
    charts_y = 2
    pyplot.figure(figsize=(20, 10))
    pyplot.subplots_adjust(hspace=.7)

    pyplot.subplot(charts_y, charts_x, 2)
    pyplot.title("blue")
    pyplot.xlabel("Frames")
    pyplot.ylabel("Pixel Average")
    pyplot.plot(blue_vals)

    pyplot.subplot(charts_y, charts_x, 2)
    pyplot.title("green")
    pyplot.xlabel("Frames")
    pyplot.ylabel("Pixel Average")
    pyplot.plot(green_vals)

    pyplot.subplot(charts_y, charts_x, 2)
    pyplot.title("Red")
    pyplot.xlabel("Frames")
    pyplot.ylabel("Pixel Average")
    pyplot.plot(red_vals)

    pyplot.show()

    # show_video_frames([vid_frames])



