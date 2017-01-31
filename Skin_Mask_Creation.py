import cv2
import os
import numpy as np
import time
from matplotlib import pyplot as plt
from Video_Tools import load_video, get_video_dimensions

start_time = time.time()


def skin_detection_algorithm_multi_video(dir_path, _dest_folder):

    for file in os.listdir(dir_path):
        if file.endswith(".MTS"):
            skin_detection_algorithm_single_video(file, dir_path, _dest_folder)


def skin_detection_algorithm_single_video(_file, _dir_path, _dest_folder, show_figure=False):

    file_path = os.path.join(_dir_path, _file)

    # define the upper and lower boundaries of the HSV pixel
    # intensities to be considered 'skin'

    # # white skin-tone
    # lower = np.array([0, 120, 80], dtype="uint8")
    # upper = np.array([20, 255, 255], dtype="uint8")

    # colored skin-tone 00128
    lower = np.array([0, 80, 30], dtype="uint8")
    upper = np.array([15, 255, 180], dtype="uint8")

    video_frames, fps = load_video(file_path)
    frame_count, width, height = get_video_dimensions(video_frames)

    video_frames = video_frames[22:50]

    skin_arr = np.ones([height, width])

    max_vals = []
    min_vals = []

    # keep looping over the frames in the video
    for i, frame in enumerate(video_frames):

        # To check which HSV values are needed for a certain person
        # cv2.rectangle(frame, (850, 530), (860, 550), (0, 255, 0), 2)
        # cv2.rectangle(frame, (600, 1000), (860, 1080), (0, 255, 0), 2)
        # frame[1000:1080, 600:860]
        rect_hsv = cv2.cvtColor(frame[230:250, 750:760], cv2.COLOR_BGR2HSV)
        # print(rect_hsv)q
        max_in_rect = np.amax(rect_hsv, axis=(0, 1))
        min_in_rect = np.amin(rect_hsv, axis=(0, 1))
        # print(np.mean(rect_hsv, axis=(0, 1)))

        max_vals.append(max_in_rect)
        min_vals.append(min_in_rect)

        # resize the frame, convert it to the HSV color space,
        # and determine the HSV pixel intensities that fall into
        # the speicifed upper and lower boundaries
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(converted, lower, upper)

        # apply a series of erosions and dilations to the mask
        # using an elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

        # blur the mask to help remove noise, then apply the
        # mask to the frame
        skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
        skin = cv2.bitwise_and(frame, frame, mask=skin_mask)

        # show the skin in the image along with the mask
        cv2.imshow("images", np.hstack([frame, skin]))
        # cv2.imshow("images", skin)

        # if the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Reduce by 1 if pixel is not a skin pixel
        mean_skin = np.mean(skin, axis=2)
        low_values_indices = mean_skin < 1
        skin_arr[low_values_indices] -= 1

    # Where values are lower than threshold
    final_mask = np.ones([height, width])
    skin_arr_mean = np.mean(skin_arr)
    skin_arr_min = np.amin(skin_arr)
    print(skin_arr_min)
    print(skin_arr_mean)
    # low_values_indices = skin_arr < 0
    low_values_indices = skin_arr < -2
    final_mask[low_values_indices] = 0

    # # Manuel modyfying
    # final_mask[950:1080, 600:860] = 0

    fig = plt.figure(figsize=(17, 9))
    sub1 = fig.add_subplot(111)
    sub1.set_title('Norm. Avg.')
    sub1.imshow(final_mask)
    fig.savefig(_dest_folder + 'Skin_' + _file[:-4] + '.jpg')
    if show_figure:
        plt.show()

    file_path_out = _dest_folder + 'Skin_' + _file[:-4]

    # Save it as .npy file
    np.save(file_path_out, final_mask)

    # cleanup the camera and close any open windows
    cv2.destroyAllWindows()


def roi_skin_mask_multi_creation(in_dir, out_dir, show_figure=False):

    for file in os.listdir(in_dir):
        if file.endswith(".npy"):
            roi_skin_mask_creation(file, in_dir, out_dir, show_figure=show_figure)


def roi_skin_mask_creation(_file, in_dir, out_dir, show_figure=False):

    full_skin_mask_file_path = os.path.join(in_dir, _file)

    w_div = 16
    h_div = 8
    skin_mask = np.load(full_skin_mask_file_path)
    height, width = skin_mask.shape
    w_steps = int(width / w_div)
    h_steps = int(height / h_div)

    roi_skin_mask = np.zeros([h_div, w_div])

    # Hier wird der ungeradere Rest abgeschnitten
    width = w_steps * w_div
    height = h_steps * h_div
    for x in range(0, width, w_steps):
        for y in range(0, height, h_steps):
            roi_ind_x = int(x / w_steps)
            roi_ind_y = int(y / h_steps)

            roi = skin_mask[y:y + h_steps, x:x + w_steps]

            roi_skin_mask[roi_ind_y, roi_ind_x] = np.mean(roi)

    skin_ind = roi_skin_mask > 0.5
    non_skin_ind = roi_skin_mask <= 0.5

    roi_skin_mask[skin_ind] = 1.0
    roi_skin_mask[non_skin_ind] = 0.0

    roi_dir = os.path.join(out_dir, 'Skin_Masks_' + str(h_div) + 'x' + str(w_div))
    if not os.path.exists(roi_dir):
        os.makedirs(roi_dir)

    roi_skin_mask_file_path = os.path.join(roi_dir, 'Skin_' + str(h_div) + 'x' + str(w_div) + _file[4:])

    fig = plt.figure(figsize=(17, 9))
    sub1 = fig.add_subplot(111)
    sub1.set_title(roi_skin_mask_file_path)
    sub1.imshow(roi_skin_mask)

    fig.savefig(roi_skin_mask_file_path[:-4] + '.png')
    if show_figure:
        plt.show()


    # Save it as .npy file
    np.save(roi_skin_mask_file_path, roi_skin_mask)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    # input_dir_path = os.path.join('assets', 'Vid_Original')
    input_dir_path = os.path.join('assets', 'Vid_Original', 'Kuenstliches_Licht')
    # dest_dir_path = os.path.join('Neural_Net', 'assets', 'Skin_Label_Data')
    dest_dir_path = os.path.join('assets', 'Skin_Label_Data')
    skin_label_dir = os.path.join('assets', 'Skin_Label_Data')
    file = '00149.MTS'
    file_path = os.path.join(input_dir_path, file)
    skin_mask_file = 'Skin_00132.npy'
    skin_mask_file_path = os.path.join(skin_label_dir, skin_mask_file)

    roi_skin_mask_multi_creation(skin_label_dir, dest_dir_path, show_figure=True)
    # roi_skin_mask_creation(skin_mask_file, skin_label_dir, skin_label_dir, show_figure=True)

    # skin_detection_algorithm_multi_video(input_dir_path, dest_skin_dir_path)
    # skin_detection_algorithm_single_video(file, input_dir_path, skin_label_dir)


    print("--- %s seconds ---" % (time.time() - start_time))