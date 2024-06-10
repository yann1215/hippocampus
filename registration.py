import numpy as np
import nibabel as nib
import glob
import os
from scipy.ndimage import affine_transform
from scipy.optimize import minimize
import SimpleITK as sitk


def mutual_information(hgram):
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def mutual_information_cost(transform_params, fixed_image, moving_image):
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
    matrix_key = "_matrix.tfm"

    # load template
    fixed_image_sitk = sitk.ReadImage(template_image, sitk.sitkFloat32)

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

        moving_image_sitk = sitk.ReadImage(file, sitk.sitkFloat32)

        mask_name = file.replace(image_input_path, mask_input_path)
        mask_name = mask_name.replace(image_key, mask_key)
        mask_sitk = sitk.ReadImage(mask_name, sitk.sitkFloat32)

        # apply transform
        print("computing affine transform matrix...")

        # 创建配准方法对象
        registration_method = sitk.ImageRegistrationMethod()

        # 设置配准的初始变换
        initial_transform = sitk.CenteredTransformInitializer(fixed_image_sitk,
                                                              moving_image_sitk,
                                                              sitk.Euler3DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)

        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        # 选择优化方法和度量标准
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0,
                                                                     minStep=1e-4,
                                                                     numberOfIterations=200,
                                                                     gradientMagnitudeTolerance=1e-8)

        # 设置多分辨率策略
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # 执行配准
        final_transform = registration_method.Execute(sitk.Cast(fixed_image_sitk, sitk.sitkFloat32),
                                                      sitk.Cast(moving_image_sitk, sitk.sitkFloat32))

        # print("Optimizer's stopping condition: ", registration_method.GetOptimizerStopConditionDescription())
        # print("Final metric value: ", registration_method.GetMetricValue())

        # save matrix
        matrix_save_name = file.replace(image_input_path, matrix_output_path)
        matrix_save_name = matrix_save_name.replace(image_key, matrix_key)
        sitk.WriteTransform(final_transform, matrix_save_name)

        # 应用最终的变换到待配准图像
        print("applying affine transformation...")
        moving_resampled = sitk.Resample(moving_image_sitk, fixed_image_sitk, final_transform, sitk.sitkLinear, 0.0,
                                         moving_image_sitk.GetPixelID())
        mask_resampled = sitk.Resample(mask_sitk, fixed_image_sitk, final_transform, sitk.sitkLinear, 0.0,
                                         mask_sitk.GetPixelID())

        # 将SimpleITK图像转换为NumPy数组
        fixed_image_affine = np.eye(4)
        fixed_image_affine[:3, :3] = np.array(fixed_image_sitk.GetDirection()).reshape(3, 3)
        fixed_image_affine[:3, 3] = np.array(fixed_image_sitk.GetOrigin())

        registered_image_np = sitk.GetArrayFromImage(moving_resampled)
        registered_image_nii = nib.Nifti1Image(registered_image_np, fixed_image_affine)
        registered_mask_np = sitk.GetArrayFromImage(mask_resampled)
        registered_mask_nii = nib.Nifti1Image(registered_mask_np, fixed_image_affine)

        # 保存到磁盘
        image_save_name = file.replace(image_input_path, image_output_path)
        image_save_name = image_save_name.replace(image_key, image_output_key)
        nib.save(registered_image_nii, image_save_name)

        mask_save_name = mask_name.replace(mask_input_path, mask_output_path)
        mask_save_name = mask_save_name.replace(mask_key, mask_output_key)
        nib.save(registered_mask_nii, mask_save_name)
