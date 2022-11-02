import glob
import random
from abc import abstractmethod

import torch

random.seed(1234)

import nrrd
import pydicom
from scipy import ndimage as nd
from torch.utils.data.dataset import Dataset
import SimpleITK as sitk
import nibabel as nib

import numpy as np
import matplotlib.pylab as plt
import os

from torchvision.transforms import transforms


def reshape_to_3d(arr):
    if len(arr.shape) == 3:
        return np.reshape(arr, (1, arr.shape[0], arr.shape[1], arr.shape[2]))
    else:
        return arr


class BaseDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None, **kwargs):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.processed_files = self.get_processed_files()

    def __len__(self):
        return len(self.processed_files)

    def get_processed_files(self):
        return {
            # key: path to processed file
            int(file_path.split("_")[-2]): file_path.replace('_image.npy', '') for file_path in
            glob.glob(f'{self.root_dir}/processed/*_image.npy') if
            (self.train == ('train' in file_path) and self.__class__.__name__ in file_path)
        }

    def __getitem__(self, idx):
        # get item
        image = reshape_to_3d(np.load(f'{self.processed_files[idx]}_image.npy'))
        label = reshape_to_3d(np.load(f'{self.processed_files[idx]}_label.npy'))

        # MinMaxScaler
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        label = (label - np.min(label)) / (np.max(label) - np.min(label))

        # transform
        # if self.transform:
        #     image, label = self.transform(image, label)
        return image, label


class Promise12(BaseDataset):
    def __init__(self, root_dir, train=True, transform=None, **kwargs):
        super().__init__(root_dir, train, transform, **kwargs)


class MedicalSegmentationDecathlon(BaseDataset):
    def __init__(self, root_dir, train=True, transform=None, **kwargs):
        super().__init__(root_dir, train, transform, **kwargs)


class NciIsbi2013(BaseDataset):
    def __init__(self, root_dir, train=True, transform=None, **kwargs):
        super().__init__(root_dir, train, transform, **kwargs)


class PROSTATEx(BaseDataset):
    def __init__(self, root_dir, train=True, transform=None, **kwargs):
        super().__init__(root_dir, train, transform, **kwargs)


def test_plot_dataset(data_obj):
    dataroot = 'data'

    trans = [transforms.ToTensor()]

    dataset = data_obj(
        dataroot,
        train=True,
        download=True,
        transform=trans
    )

    test_set = data_obj(
        dataroot,
        train=False,
        download=True,
        transform=trans
    )

    train_scans, train_seg = dataset[0]
    test_scans, test_seg = test_set[0]

    print(data_obj)
    print(train_scans.shape)
    print(train_seg.shape)
    print(test_scans.shape)
    print(test_seg.shape)

    plt.figure(figsize=(20, 16))
    plt.gray()
    plt.subplots_adjust(0, 0, 1, 1, 0.01, 0.01)
    for i in range(train_scans.shape[0]):
        plt.subplot(5, 6, i + 1), plt.imshow(train_scans[i]), plt.axis('off')
        # use plt.savefig(...) here if you want to save the images as .jpg, e.g.,
    plt.show()

    plt.figure(figsize=(20, 16))
    plt.gray()
    plt.subplots_adjust(0, 0, 1, 1, 0.01, 0.01)
    for i in range(train_seg.shape[0]):
        plt.subplot(5, 6, i + 1), plt.imshow(train_seg[i]), plt.axis('off')
        # use plt.savefig(...) here if you want to save the images as .jpg, e.g.,
    plt.show()

    plt.figure(figsize=(20, 16))
    plt.gray()
    plt.subplots_adjust(0, 0, 1, 1, 0.01, 0.01)
    for i in range(test_scans.shape[0]):
        plt.subplot(5, 6, i + 1), plt.imshow(test_scans[i]), plt.axis('off')
        # use plt.savefig(...) here if you want to save the images as .jpg, e.g.,
    plt.show()

    plt.figure(figsize=(20, 16))
    plt.gray()
    plt.subplots_adjust(0, 0, 1, 1, 0.01, 0.01)
    for i in range(test_seg.shape[0]):
        plt.subplot(5, 6, i + 1), plt.imshow(test_seg[i]), plt.axis('off')
        # use plt.savefig(...) here if you want to save the images as .jpg, e.g.,
    plt.show()


if __name__ == '__main__':
    test_plot_dataset(Promise12)
    test_plot_dataset(MedicalSegmentationDecathlon)
    test_plot_dataset(NciIsbi2013)
    test_plot_dataset(PROSTATEx)
