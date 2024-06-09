import numpy as np
import nibabel as nib
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import gzip
import cv2
# from mpl_toolkits.mplot3d import Axes3D
# from mayavi import mlab


import os
import matplotlib.pyplot as plt


def get_file_path(file_name, input_path, output_path, axis):
    save_path = output_path + axis + "/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)  # create output folder

    # normalize path name; otherwise replace() would go wrong
    file_name = os.path.normpath(file_name)
    input_path = os.path.normpath(input_path)
    save_path = os.path.normpath(save_path)

    save_name = file_name.replace(input_path, save_path)
    if ".nii.gz" in save_name:
        save_name = save_name.replace(".nii.gz", "_" + axis + ".jpg")
    elif ".nii" in save_name:
        save_name = save_name.replace(".nii", "_" + axis + ".jpg")
    # print("output file name: ", save_name)

    return save_name


def jpg_save(file_data, output_path):
    plt.figure()
    plt.imshow(file_data.T, cmap="gray", origin="lower")
    plt.axis("off")
    plt.savefig(output_path, format='jpg', bbox_inches='tight', pad_inches=0)
    plt.close()


def nii2jpg(input_path, output_path):
    file_path = input_path + "/*.nii" + "*"
    file_path = os.path.normpath(file_path)

    for file in glob.glob(file_path):
        print("nii2jpg processing: ", file)

        # check if file is ok
        # note: processed file errors in this case, but the data is available
        # try:
        #     with gzip.open(file, "rb") as f:
        #         f.read(1)
        # except OSError:
        #     print(f"{file} ERROR")
        #     continue

        file_obj = nib.load(file)
        # get numpy data
        file_data = file_obj.get_fdata()

        z, y, x = file_data.shape

        path_z = get_file_path(file, input_path, output_path, "z")
        file_data_z = file_data[int(z / 2), :, :]
        jpg_save(file_data_z, path_z)

        path_y = get_file_path(file, input_path, output_path, "y")
        file_data_y = file_data[:, int(y / 2), :]
        jpg_save(file_data_y, path_y)

        path_x = get_file_path(file, input_path, output_path, "x")
        file_data_x = file_data[:, :, int(x / 2)]
        jpg_save(file_data_x, path_x)


def create_video(image_folder, output_video_path, fps):
    # Get the list of image files and ensure sorting by name
    images = [img for img in os.listdir(image_folder) if img.endswith((".jpg"))]

    # Ensure sorting of images by name
    # note: this line of code disturbs fps; no need in this case
    # images.sort()

    # Read the first image to get frame width, height, and layers
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Using mp4v codec
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    cv2.destroyAllWindows()


def count_data(input_path):
    count_flag = 1
    file_path = input_path + "*.nii" + "*"

    for file in glob.glob(file_path):
        if count_flag:
            count_result = np.zeros_like(file_data)
            count_num = 0
            count_flag = 0
            print("mask shape is: ", file_data.shape)

        print("count_data processing: file")

        file_obj = nib.load(file)
        # get numpy data
        file_data = file_obj.get_fdata()

        count_result += np.where(file_data > 0, 1, 0)
        count_num += 1

    print("file number is: ", count_num)
    print("count result shape is: ", count_result.shape)
    # mlab.contour3d(count_result)
    # mlab.show()

    count_max = np.max(count_result)
    print("max value of mask is: ", count_max)

    max_position = np.unravel_index(np.argmax(count_result), count_result.shape)
    print("max value location is: ", max_position)

    # return max_location


if __name__ == "__main__":
    # ---------- Make Statistics ----------
    mask_path = "data_preprocessed/mask_second/"
    mask_path = "dataset/AffinedManualSegImageNIfTI/"
    count_data(mask_path)

    # # ---------- Get NII Slice Image ----------
    # # raw image
    # # file_path = "dataset/RawImageNIfTI/"
    # # jpg_path = "dataset/RawImageJPG/"
    #
    # # processed 2
    # file_path = "data_preprocessed/preprocess_second/"
    # jpg_path = "data_preprocessed/preprocess_second_jpg/"
    #
    # if not os.path.exists(jpg_path):
    #     os.makedirs(jpg_path)  # create output folder
    #
    # # nii2jpg(file_path, jpg_path)

    # ---------- Get Slice Video ----------
    # video_char = "z"
    #
    # # image_folder = f"dataset/RawImageJPG/{video_char}/"
    # # output_video_path = f"dataset/RawImage_{video_char}.mp4"
    # image_folder = jpg_path + video_char + "/"
    #
    # if os.path.exists(image_folder):
    #     output_video_path = jpg_path + f"preprocess_second_{video_char}.mp4"
    #     fps = 2
    #     print("generating video...")
    #     create_video(image_folder, output_video_path, fps)
    #     print("video generated")
    # else:
    #     print("image folder empty, video generation failed")
