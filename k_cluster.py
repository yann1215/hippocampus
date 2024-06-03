import numpy as np
import nibabel as nib
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans


def k_cluster(file_name, k):
    file_obj = nib.load(file_name)
    # get numpy data
    file_data = file_obj.get_fdata()

    # test
    # np.random.seed(42)
    # file_data = np.random.rand(10, 10, 10)
    print(file_data.shape)

    # 展平三维数组为二维数组
    x, y, z = file_data.shape
    flat_data = file_data.reshape((x * y * z, 1))

    # 选择簇数 k
    k = 4

    # 进行 k-means 聚类
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(flat_data)
    labels = kmeans.labels_

    # 重构结果为三维数组
    clustered_data = labels.reshape((x, y, z))

    # 可视化结果
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 显示不同切片
    axes[0].imshow(clustered_data[:, :, 0], cmap='viridis')
    axes[0].set_title('Slice 1')

    axes[1].imshow(clustered_data[:, :, 5], cmap='viridis')
    axes[1].set_title('Slice 2')

    axes[2].imshow(clustered_data[:, :, 9], cmap='viridis')
    axes[2].set_title('Slice 3')

    plt.show()


if __name__ == "__main__":
    # input_path = "data_preprocessed/preprocess_second/"
    # file_path = input_path + "/*.nii" + "*"

    # test
    file_path = "data_preprocessed/preprocess_second/s001_second_preprocessed.nii"

    file_path = os.path.normpath(file_path)

    for file in glob.glob(file_path):
        print("k-cluster processing: ", file)
        k_cluster(file, 4)
        print(f"k-cluster for file {file} done")
