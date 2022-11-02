import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
import pydicom
import glob
import json
import nrrd
import nibabel as nib
from torchvision import transforms

random.seed(1234)


# utils
def split_dict(d, pct):
    """Split a dictionary into two dictionaries, one with pct of the keys and one with the rest."""
    keys = list(d.keys())
    random.shuffle(keys)
    split = int(len(keys) * pct)
    return {k: d[k] for k in keys[:split]}, {k: d[k] for k in keys[split:]}


def random_crop(image, size):
    """Randomly crop a 3D image to a given size."""
    _, x, y, z = image.shape
    cx, cy, cz = size
    x1 = random.randint(0, x - cx - 1) if x > cx + 1 else 0
    y1 = random.randint(0, y - cy - 1) if y > cy + 1 else 0
    z1 = random.randint(0, z - cz - 1) if z > cz + 1 else 0
    return x1, y1, z1


class BaseDataset(Dataset):
    def is_valid(self, image_path, label_path):
        im = self.load_image(image_path)
        label = self.load_label(label_path)
        if im.shape != label.shape:
            if len(im.shape) == len(label.shape):
                print(f"image: {im.shape} | label: {label.shape}")
                return False

        return True

    def load_image(self, path):
        pass

    def load_label(self, path):
        return self.load_image(path)

    def get_id_to_path_dict(self, root_dir):
        pass

    def __init__(self, root_dir, train=True, transform=None, **kwargs):
        self.root_dir = root_dir

        self.scan_size = (16, 160, 160)
        self.full_id_to_path_dict = self.get_id_to_path_dict(root_dir)
        self.train = train

        # remove invalid samples
        self.full_id_to_path_dict = {k: v for k, v in self.full_id_to_path_dict.items() if self.is_valid(v[0], v[1])}
        print(f"{self.__class__.__name__} \t {len(self.full_id_to_path_dict)}")

        train_dict, test_dict = split_dict(self.full_id_to_path_dict, 0.8)

        if self.train:
            self.id_to_path_dict = {idx: v for idx, v in enumerate(train_dict.values())}
        else:
            self.id_to_path_dict = {idx: v for idx, v in enumerate(test_dict.values())}

        self.transform = transform

    def __len__(self):
        return len(self.id_to_path_dict)

    def is_cached(self, idx):
        assert os.path.exists(self.root_dir + "/cache")
        file_name = f"{self.__class__.__name__}__{'train' if self.train else 'test'}__{idx}"
        return os.path.exists(self.root_dir + "/cache/" + file_name + "__image.npy")

    def __getitem__(self, idx):
        # print(f"{self.__class__} | idx = {idx}")
        if self.is_cached(idx):
            # load image, label from cache
            file_name = f"{self.__class__.__name__}__{'train' if self.train else 'test'}__{idx}"
            image = np.load(self.root_dir + "/cache/" + file_name + "__image.npy")
            label = np.load(self.root_dir + "/cache/" + file_name + "__label.npy")

        else:
            image_path, label_path = self.id_to_path_dict[idx]
            image = self.load_image(image_path)
            label = self.load_label(label_path)

            # print(f"after loading | image shape: {image.shape}, label shape: {label.shape}")

            # float type
            image = image.astype(float)
            label = label.astype(float)

            # all images are 3D, but some are 1xHxWxD, so we reshape them to 1xHxWxD
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            if len(label.shape) == 3:
                label = np.expand_dims(label, axis=0)

            # print(f"after expand_dims | image shape: {image.shape}, label shape: {label.shape}")

            # TODO:change num of padding slices
            # pad to minimum 16 slices
            if image.shape[1] < 16:
                image = np.pad(image, ((0, 0), (0, 16 - image.shape[1]), (0, 0), (0, 0)), 'constant',
                               constant_values=(0))
                label = np.pad(label, ((0, 0), (0, 16 - label.shape[1]), (0, 0), (0, 0)), 'constant',
                               constant_values=(0))
                # print(f"image shape: {image.shape}, label shape: {label.shape}")

            # print(f"after padding | image shape: {image.shape}, label shape: {label.shape}")

            # normalize non-zero elements by their mean and std
            non_zero_index = np.nonzero(image)
            non_zero_values = image[non_zero_index]
            mean = np.mean(non_zero_values)
            std = np.std(non_zero_values)
            image[non_zero_index] = (image[non_zero_index] - mean) / std

            # print(f"after normalization | image shape: {image.shape}, label shape: {label.shape}")

            # save to cache
            file_name = f"{self.__class__.__name__}__{'train' if self.train else 'test'}__{idx}"
            np.save(self.root_dir + "/cache/" + file_name + "__image.npy", image)
            np.save(self.root_dir + "/cache/" + file_name + "__label.npy", label)

            ## End of "Not Cached" ##

        # TODO: crop 3 pathces from each image
        # random crop image to pathces 160X160
        x1, y1, z1 = random_crop(image, self.scan_size)
        cx, cy, cz = self.scan_size
        # print(f"image shape: {image.shape}, selected point: {(x1, y1, z1)}")
        image = image[:, x1:(x1 + cx), y1:(y1 + cy), z1:(z1 + cz)]
        label = label[:, x1:(x1 + cx), y1:(y1 + cy), z1:(z1 + cz)]

        # print(f"after random_crop | image shape: {image.shape}, label shape: {label.shape}")

        # transform
        # if self.transform:
        #     image = self.transform(image)
        #     label = self.transform(label)

        return image, label


class Promise12(BaseDataset):

    def load_image(self, path):
        return sitk.GetArrayFromImage(sitk.ReadImage(path))

    def get_id_to_path_dict(self, root_dir):
        root_dir += '/Promise12'
        id_to_partition = {}
        for i in [1, 2, 3]:
            files_dict = {f[4:6]: i for f in os.listdir(f'{root_dir}/TrainingData_Part{i}/') if
                          f.endswith('.mhd') and 'segmentation' not in f}
            id_to_partition.update(files_dict)

        id_to_case_dict = {k: f'{root_dir}/TrainingData_Part{id_to_partition[k]}/Case{k}' for k in
                           id_to_partition.keys()}
        return {idx: (f'{id_to_case_dict[k]}.mhd', f'{id_to_case_dict[k]}_segmentation.mhd') for idx, k in
                enumerate(id_to_case_dict.keys())}


class PROSTATEx(BaseDataset):

    def load_image(self, path):
        slices = [pydicom.dcmread(s) for s in sorted(glob.glob(path + '/*.dcm'))]
        return np.stack([s.pixel_array for s in slices])

    def load_label(self, path):
        label = self.load_image(path)
        slices_num = label.shape[1] // 4
        new_label = np.zeros((label.shape[0], slices_num, label.shape[2], label.shape[3]))
        for i in range(4):
            new_label += label[0, (slices_num * i):(slices_num * (i + 1)), :, :]
        return new_label

    def get_id_to_path_dict(self, root_dir):
        root_dir += '/PROSTATEx'
        samples_metadata = pd.read_csv(f'{root_dir}/Samples/metadata.csv').set_index('Subject ID')
        labels_metadata = pd.read_csv(f'{root_dir}/Labels/metadata.csv').set_index('Subject ID')
        id_to_path_dict = {
            idx: (f"{root_dir}/Samples/" + samples_metadata.loc[s]['File Location'][2:],
                  f"{root_dir}/Labels/" + labels_metadata.loc[s]['File Location'][2:]) for idx, s in
            enumerate(samples_metadata.index)
        }
        return id_to_path_dict


class NciIsbi2013(BaseDataset):
    def load_image(self, path):
        slices = [pydicom.dcmread(s) for s in sorted(glob.glob(path + '/*.dcm'))]
        return np.stack([s.pixel_array for s in slices])

    def load_label(self, path):
        # nrrd returns a tuple of (data, header)
        label = nrrd.read(path)[0]
        label = np.where(label == 2, 1, label)
        return np.swapaxes(label, 0, 2)

    def get_id_to_path_dict(self, root_dir):
        root_dir += '/NCI-ISBI-2013'
        image_metadata = pd.read_csv(
            f'{root_dir}/ISBI-Prostate-Challenge-Training/metadata.csv').set_index('Subject ID')
        idx_to_subject_id = {idx: s for idx, s in enumerate(
            sorted(image_metadata.index.values))}
        id_to_path_dict = {
            idx: (
                f'{root_dir}/ISBI-Prostate-Challenge-Training/' +
                image_metadata.loc[idx_to_subject_id[idx],
                                   'File Location'][2:],
                f'{root_dir}/Labels/Training/{idx_to_subject_id[idx]}.nrrd'
            ) for idx in idx_to_subject_id.keys()
        }
        return id_to_path_dict


class MedicalSegmentationDecathlon(BaseDataset):
    def load_image(self, path):
        nii_img = nib.load(path)
        nii_data = nii_img.get_fdata()[:, :, :, 0]
        return np.swapaxes(nii_data, 0, 2)

    def load_label(self, path):
        nii_img = nib.load(path)
        nii_data = nii_img.get_fdata()
        nii_data = np.where(nii_data == 2, 1, nii_data)
        return np.swapaxes(nii_data, 0, 2)

    def get_id_to_path_dict(self, root_dir):
        root_dir += '/MedicalSegmentationDecathlon/Task05_Prostate/'
        with open(f'{root_dir}dataset.json') as f:
            data = json.load(f)
        id_to_path_dict = {id: (root_dir + v['image'][2:-3], root_dir + v['label'][2:-3])
                           for id, v in enumerate(data['training'])}

        return id_to_path_dict


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
