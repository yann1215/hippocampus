import numpy as np
import nibabel as nib
import glob
import os
from scipy.ndimage import affine_transform
from scipy.optimize import minimize


def mutual_information(hgram):
    """ Mutual information for joint histogram """
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def mutual_information_cost(transform_params, fixed_image, moving_image):
    """ Cost function for optimization """
    transform_matrix = np.array([[np.cos(transform_params[0]), -np.sin(transform_params[0]), 0, transform_params[1]],
                                 [np.sin(transform_params[0]),  np.cos(transform_params[0]), 0, transform_params[2]],
                                 [0, 0, 1, transform_params[3]],
                                 [0, 0, 0, 1]])
    transformed_image = affine_transform(moving_image, transform_matrix)
    hist_2d, _, _ = np.histogram2d(fixed_image.ravel(),
                                   transformed_image.ravel(),
                                   bins=20)
    return -mutual_information(hist_2d)


if __name__ == "__main__":
    # template
    template_image = "dataset/template.nii.gz"
    # template_mask = "dataset/template_mask.nii.gz"

    # data
    mask_input_path = "dataset/AffinedManualSegImageNIfTI"
    image_input_path = "dataset/RawImageNIfTI"

    # file name keyword
    mask_key = "_mask.nii.gz"
    image_key = ".nii.gz"

    # file output path
    mask_output_path = "data_registration/mask"
    image_output_path = "data_registration/image"
    matrix_output_path = "data_registration/matrix"

    if not os.path.exists(mask_output_path):
        os.makedirs(mask_output_path)  # create output folder
    if not os.path.exists(image_output_path):
        os.makedirs(image_output_path)  # create output folder
    if not os.path.exists(matrix_output_path):
        os.makedirs(matrix_output_path)  # create output folder

    # file output keyword
    mask_output_key = "_mask_2.nii.gz"
    image_output_key = "_2.nii.gz"
    matrix_key = "_matrix.npy"

    # load template
    template_obj = nib.load(template_image)
    fixed_image = template_obj.get_fdata()

    # do registration
    file_path = image_input_path + "/" + "*.nii" + "*"

    # file path name normalize
    file_path = os.path.normpath(file_path)
    image_input_path = os.path.normpath(image_input_path)
    mask_input_path = os.path.normpath(mask_input_path)
    image_output_path = os.path.normpath(mask_output_path)
    image_output_path = os.path.normpath(image_output_path)
    matrix_output_path = os.path.normpath(matrix_output_path)

    for file in glob.glob(file_path):
        print(f"registration processing: {file}")

        # get numpy data
        file_obj = nib.load(file)
        changing_image = file_obj.get_fdata()

        mask_name = file.replace(image_input_path, mask_input_path)
        mask_name = mask_name.replace(image_key, mask_key)
        print(mask_name)
        mask_obj = nib.load(mask_name)
        changing_mask = mask_obj.get_fdata()

        # apply transform
        print("computing affine transform matrix...")
        initial_guess = [0, 0, 0, 0]
        result = minimize(mutual_information_cost, initial_guess,
                          args=(fixed_image, changing_image),
                          method='Powell')
        optimal_transform_params = result.x

        rotation, translation_x, translation_y, translation_z = optimal_transform_params
        optimal_transform_matrix = np.array([[np.cos(rotation), -np.sin(rotation), 0, translation_x],
                                             [np.sin(rotation), np.cos(rotation), 0, translation_y],
                                             [0, 0, 1, translation_z],
                                             [0, 0, 0, 1]])

        # save matrix
        matrix_save_name = file.replace(image_input_path, matrix_output_path)
        matrix_save_name = matrix_save_name.replace(image_key, matrix_key)
        np.save(matrix_save_name, optimal_transform_matrix)

        # apply matrix to both image and mask
        print("applying transform...")
        registered_image = affine_transform(changing_image, optimal_transform_matrix)
        registered_mask = affine_transform(changing_mask, optimal_transform_matrix)

        new_image = nib.Nifti1Image(registered_image, file_obj.affine, file_obj.header)
        new_mask = nib.Nifti1Image(registered_mask, mask_obj.affine, mask_obj.header)

        # save nii data
        image_save_name = file.replace(image_input_path, image_output_path)
        image_save_name = image_save_name.replace(image_key, image_output_key)
        nib.save(new_image, image_save_name)
        print(f"saved: {image_save_name}")

        mask_save_name = mask_name.replace(mask_input_path, mask_output_path)
        mask_save_name = mask_save_name.replace(mask_key, mask_output_key)
        nib.save(new_mask, mask_save_name)
        print(f"saved: {mask_save_name}")
