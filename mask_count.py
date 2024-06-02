import numpy as np
import nibabel as nib
import pandas as pd
import glob
import os


def count_data(input_path):
    count_flag = 1

    for file in glob.glob(input_path):
        mask_obj = nib.load(file)
        mask_data = mask_obj.get_fdata()

        if count_flag:
            count_result = np.zeros_like(mask_data)
            count_flag = 0

        np.where()

    return 0


image_path = "data_preprocessed/mask_second/i"

