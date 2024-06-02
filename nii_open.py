import numpy as np
import nibabel as nib
from ipywidgets import interact, interactive, IntSlider, ToggleButtons
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("darkgrid")

image_path = "data_preprocessed/preprocess_second/s001_second_preprocessed.nii"

image_obj = nib.load(image_path)
# print(f"Type of the image {type(image_obj)}")

image_data = image_obj.get_fdata()
# type(image_data)

height, width, depth = image_data.shape
print(f"The image object height: {height}, width:{width}, depth:{depth}")

# print("max value of an image", np.max(image_data))
print(f"image value range: [{image_data.min()}, {image_data.max()}]")
# image_data2 = image_data * 255 / (image_data.max() - image_data.min())

# print(image_obj.header.keys())
# pixdim = image_obj.header["pixdim"]
# print(f"z-axis resolution ratio： {pixdim[3]}")
# print(f"in plane resolution ratio： {pixdim[1]} * {pixdim[2]}")

z_range = pixdim[3] * depth
x_range = pixdim[1] * height
y_range = pixdim[2] * width
print(x_range, y_range, z_range)

i = 50
print(f"Plotting z Layer {i} of Image")
plt.imshow(image_data[:, :, i], cmap="gray")
plt.axis("off")

plt.show()
