import glob

import nrrd
import pydicom
from torch.utils.data.dataset import Dataset
import SimpleITK as sitk
import nibabel as nib

import numpy as np
import matplotlib.pylab as plt
import os

from torchvision.transforms import transforms

MRI_SCAN_SHAPE = (15, 320, 320)


def random_crop(arr, shape):
    assert len(arr.shape) == len(shape)
    for i in range(len(shape)):
        assert shape[i] <= arr.shape[i]
    idx = [np.random.randint(0, arr.shape[i] - shape[i]) if arr.shape[i] > shape[i] else 0 for i in range(len(shape))]
    return arr[[slice(idx[i], idx[i] + shape[i]) for i in range(len(shape))]]


class BaseDataset(Dataset):
    pass


class PROSTATEx(Dataset):
    """PROSTATEx Dataset"""

    def __init__(self, root_dir, train=True, transform=None, **kwargs):
        """
        Args:
            root_dir (string): Directory with an inner dir named "PROSTATEx".
                |- root_dir
                |  |- PROSTATEx
                |  |  |- Labels
                |  |  |  |- PROSTATEx
                |  |  |- Samples
                |  |  |  |- PROSTATEx

            train (bool): Whether to take the training dataset or the test dataset
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # TODO: Save the data in this format also in cortex
        self.root_dir = root_dir + "/PROSTATEx"

        self.imgs_dir = self.root_dir + "/Samples" + "/PROSTATEx"
        self.labels_dir = self.root_dir + "/Labels" + "/PROSTATEx"

        self.train = train
        self.transform = transform

        file_names = [f for f in os.listdir(f"{self.imgs_dir}") if f.startswith("Prostate")]
        self.file_names = file_names
        self.idx_to_case_map = dict(enumerate(file_names))

    def __len__(self):
        return len(self.idx_to_case_map)

    def __getitem__(self, idx):
        patient_dir = self.idx_to_case_map[idx]

        dcm_files_list = glob.glob(f'{self.imgs_dir}/{patient_dir}/**/*.dcm', recursive=True)

        unstacked_list = []
        for dicom_filepath in dcm_files_list:
            # convert dicom file into jpg file
            np_pixel_array = pydicom.read_file(dicom_filepath).pixel_array
            unstacked_list.append(np_pixel_array)
        np_scans = np.array(unstacked_list).astype(float)

        # swap axes to match the shape of other Datasets
        # np_scans = np.swapaxes(np_scans, 0, 2)

        # read labels
        dicom_filepath = glob.glob(f'{self.labels_dir}/{patient_dir}/**/*.dcm', recursive=True)[0]

        # convert dicom file into jpg file
        # TODO: there are 4 segmentations, I'm not sure which one to use.
        # Looking at the images, it seems that seg_id=1 is the closest to other datasets.
        seg_id = 1
        np_labels = pydicom.read_file(dicom_filepath).pixel_array[19 * seg_id:19 * (seg_id + 1), :, :].astype(float)
        np_labels = random_crop(np_labels, MRI_SCAN_SHAPE)

        return random_crop(np_scans, MRI_SCAN_SHAPE), np_labels


class NciIsbi2013(Dataset):
    """NCI-ISBI-2013 Dataset"""

    def __init__(self, root_dir, train=True, transform=None, **kwargs):
        """
        Args:
            root_dir (string): Directory with an inner dir named "NCI-ISBI-2013".
                |- root_dir
                |  |- NCI-ISBI-2013
                |  |  |- ISBI-Prostate-Challenge-Training # Train images
                |  |  |- ISBI-Prostate-Challenge-Testing # Test images
                |  |  |- Labels
                |  |  |- |- Test # Test labels
                |  |  |- |- Train # Train labels

            train (bool): Whether to take the training dataset or the test dataset
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # TODO: Save the data in this format also in cortex
        self.root_dir = root_dir + "/NCI-ISBI-2013"

        self.train_imgs_dir = self.root_dir + "/ISBI-Prostate-Challenge-Training"
        self.train_labels_dir = self.root_dir + "/Labels/Training"
        self.test_imgs_dir = self.root_dir + "/ISBI-Prostate-Challenge-Testing"
        self.test_labels_dir = self.root_dir + "/Labels/Test"

        self.train = train
        self.transform = transform

        self.curr_imgs_dir = self.train_imgs_dir if self.train else self.test_imgs_dir

        subdir_to_prefix = {
            'Prostate-3T': 'Prostate3T',
            'PROSTATE-DIAGNOSIS': 'ProstateDx'
        }

        file_names = sum([[f"{subdir}/{f}" for f in
                           os.listdir(f"{self.curr_imgs_dir}/{subdir}") if f.startswith(subdir_to_prefix[subdir])] for
                          subdir in subdir_to_prefix], [])
        self.file_names = file_names
        self.idx_to_case_map = dict(enumerate(file_names))

    def __len__(self):
        return len(self.idx_to_case_map)

    def __getitem__(self, idx):
        patient_dir = self.idx_to_case_map[idx]

        dcm_files_list = glob.glob(f'{self.curr_imgs_dir}/{patient_dir}/**/*.dcm', recursive=True)
        unstacked_list = []
        for dicom_filepath in dcm_files_list:
            # convert dicom file into jpg file
            np_pixel_array = pydicom.read_file(dicom_filepath).pixel_array
            unstacked_list.append(np_pixel_array)
        np_scans = np.array(unstacked_list).astype(float)

        # swap axes to match the shape of other Datasets
        # np_scans = np.swapaxes(np_scans, 0, 2)

        # read labels
        labels_path = [f for f in os.listdir(self.train_labels_dir) if f.startswith(patient_dir.split("/")[-1])][0]
        np_labels, header = nrrd.read(f"{self.train_labels_dir}/{labels_path}")
        np_labels = np.swapaxes(np_labels, 0, 2).astype(float)
        np_labels = random_crop(np_labels, MRI_SCAN_SHAPE)

        return random_crop(np_scans, MRI_SCAN_SHAPE), np_labels


class MedicalSegmentationDecathlon(Dataset):
    """MedicalSegmentationDecathlon Dataset"""

    def __init__(self, root_dir, train=True, transform=None, **kwargs):
        """
        Args:
            root_dir (string): Directory with an inner dir named "MedicalSegmentationDecathlon".
                |- root_dir
                |  |- MedicalSegmentationDecathlon
                |  |  |- imagesTr # Train images
                |  |  |- imagesTs # Test images
                |  |  |- labelsTr # Train labels
            train (bool): Whether to take the training dataset or the test dataset
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # TODO: Save the data in this format also in cortex
        self.root_dir = root_dir + "/MedicalSegmentationDecathlon"

        self.train_imgs_dir = self.root_dir + "/imagesTr"
        self.train_labels_dir = self.root_dir + "/labelsTr"
        self.test_imgs_dir = self.root_dir + "/imagesTs"

        self.train = train
        self.transform = transform

        self.curr_imgs_dir = self.train_imgs_dir if self.train else self.test_imgs_dir

        file_names = [f for f in os.listdir(self.curr_imgs_dir) if f.endswith(".nii")]
        self.idx_to_case_map = dict(enumerate(file_names))

    def __len__(self):
        return len(self.idx_to_case_map)

    def __getitem__(self, idx):
        nii_file_name = self.idx_to_case_map[idx]

        # read scan
        orig_scans = nib.load(f"{self.curr_imgs_dir}/{nii_file_name}")

        # TODO: What is the difference between 0 and 1 at the last index? For now we use 0
        np_scans = orig_scans.get_fdata()[:, :, :, 0]

        # swap axes to match the shape of other Datasets
        np_scans = np.swapaxes(np_scans, 0, 2)

        # read labels
        orig_labels = nib.load(f"{self.curr_imgs_dir}/{nii_file_name}".replace("imagesTr", "labelsTr"))

        np_labels = orig_labels.get_fdata()
        np_labels = np.swapaxes(np_labels, 0, 2)
        np_labels = random_crop(np_labels, MRI_SCAN_SHAPE)



        return random_crop(np_scans, MRI_SCAN_SHAPE), np_labels


class Promise12(Dataset):
    """Promise12 Medical Dataset"""

    def __init__(self, root_dir, train=True, transform=None, **kwargs):
        """
        Args:
            root_dir (string): Directory with an inner dir named "Promise12".
                |- root_dir
                |  |- Promise12
                |  |  |- Test
                |  |  |- Train
            train (bool): Whether to take the training dataset or the test dataset
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # TODO: Save the data in this format also in cortex
        self.root_dir = root_dir + "/Promise12"
        self.train_dir = self.root_dir + "/Train"
        self.test_dir = self.root_dir + "/Test"

        self.train = train
        self.transform = transform

        self.curr_dir = self.train_dir if self.train else self.test_dir

        file_names = os.listdir(self.curr_dir)
        nums_in_filenames = sorted(list(set([''.join(filter(lambda i: i.isdigit(), s)) for s in file_names])))
        self.idx_to_case_map = dict(enumerate(nums_in_filenames))

    def __len__(self):
        return len(self.idx_to_case_map)

    def __getitem__(self, idx):
        case_idx = self.idx_to_case_map[idx]

        if not os.path.exists(f"{self.curr_dir}/Case{case_idx}.mhd"):
            raise Exception(f"Case {case_idx} not in {'Train' if self.train else 'Test'} directory")

        scans = sitk.GetArrayFromImage(sitk.ReadImage(f"{self.curr_dir}/Case{case_idx}.mhd", sitk.sitkFloat32))

        label_segmentations = sitk.GetArrayFromImage(
            sitk.ReadImage(f"{self.curr_dir}/Case{case_idx}_segmentation.mhd",
                           sitk.sitkFloat32))
        label_segmentations = random_crop(label_segmentations, MRI_SCAN_SHAPE)

        return random_crop(scans, MRI_SCAN_SHAPE), label_segmentations


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

    print(train_scans.shape)
    print(train_seg.shape)
    print(test_scans.shape)
    print(test_seg)

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
    # test_plot_dataset(Promise12)
    test_plot_dataset(MedicalSegmentationDecathlon)
    test_plot_dataset(NciIsbi2013)
    test_plot_dataset(PROSTATEx)
