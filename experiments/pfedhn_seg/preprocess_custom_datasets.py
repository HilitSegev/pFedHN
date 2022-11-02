import argparse
import glob
import random

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

MRI_SCAN_SHAPE = (16, 160, 160)


def resize_scan(arr, shape, idx=None):
        # resize image to "shape"
        dsfactor = [w / float(f) for w, f in zip(shape, arr.shape)]
        downed = np.round(nd.zoom(arr, zoom=dsfactor))
        return downed, idx


def reshape_to_3d(arr):
    if len(arr.shape) == 3:
        arr = resize_scan(arr, MRI_SCAN_SHAPE)[0]
        return np.reshape(arr, (1, arr.shape[0], arr.shape[1], arr.shape[2]))
    else:
        return arr


class BaseDataset(Dataset):
    def __len__(self):
        return len(self.processed_files)


class PROSTATEx(BaseDataset):
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
        # check if processed files exist
        self.train = train
        self.transform = transform
        if os.path.exists(
                f"{root_dir}/processed/{self.__class__.__name__}_{'train' if self.train else 'test'}_0_image.npy"):
            processed_files_list = glob.glob(
                f"{root_dir}/processed/{self.__class__.__name__}_{'train' if self.train else 'test'}_*_image.npy")
            self.processed_files = {key: f[:-10] for key, f in enumerate(processed_files_list)}
        else:
            print("PROSTATEx dataset not found. Preprocessing...")
            self.root_dir = root_dir + "/PROSTATEx"

            self.imgs_dir = self.root_dir + "/Samples" + "/PROSTATEx"
            self.labels_dir = self.root_dir + "/Labels" + "/PROSTATEx"

            file_names = [f for f in os.listdir(f"{self.imgs_dir}") if f.startswith("Prostate")]

            test_files = random.sample(file_names, int(len(file_names) * 0.2))
            train_files = [f for f in file_names if f not in test_files]
            self.file_names = train_files if self.train else test_files

            self.idx_to_case_map = dict(enumerate(self.file_names))

            self.processed_files = {
                key: f"{root_dir}/processed/{self.__class__.__name__}_{'train' if self.train else 'test'}_{key}" for key
                in
                self.idx_to_case_map
            }

    def __getitem__(self, idx):
        if os.path.exists(self.processed_files[idx] + '_image.npy'):
            image, label = np.load(self.processed_files[idx] + '_image.npy'), \
                           np.load(self.processed_files[idx] + '_label.npy')
            image = (image - image.min()) / (image.max() - image.min())
            return reshape_to_3d(image), reshape_to_3d(label)
        print(f"Processing {self.idx_to_case_map[idx]}")
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

        # normalize np_scans to be 0-255
        np_scans = 255 * (np_scans - np_scans.min()) / (np_scans.max() - np_scans.min())

        # read labels
        dicom_filepath = glob.glob(f'{self.labels_dir}/{patient_dir}/**/*.dcm', recursive=True)[0]

        # convert dicom file into jpg file
        # TODO: there are 4 segmentations, I'm not sure which one to use.
        # Looking at the images, it seems that seg_id=1 is the closest to other datasets.
        seg_id = 1
        np_labels = pydicom.read_file(dicom_filepath).pixel_array[19 * seg_id:19 * (seg_id + 1), :, :].astype(float)
        np_labels, idx = resize_scan(np_labels, MRI_SCAN_SHAPE)

        return resize_scan(np_scans, MRI_SCAN_SHAPE, idx)[0], np_labels


class NciIsbi2013(BaseDataset):
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
        # check if processed files exist
        self.train = train
        self.transform = transform
        if os.path.exists(
                f"{root_dir}/processed/{self.__class__.__name__}_{'train' if self.train else 'test'}_0_image.npy"):
            processed_files_list = glob.glob(
                f"{root_dir}/processed/{self.__class__.__name__}_{'train' if self.train else 'test'}_*_image.npy")
            self.processed_files = {key: f[:-10] for key, f in enumerate(processed_files_list)}
        else:
            print("NCI-ISBI-2013 dataset not found. Preprocessing...")
            self.root_dir = root_dir + "/NCI-ISBI-2013"

            self.train_imgs_dir = self.root_dir + "/ISBI-Prostate-Challenge-Training"
            self.train_labels_dir = self.root_dir + "/Labels/Training"
            self.test_imgs_dir = self.root_dir + "/ISBI-Prostate-Challenge-Testing"
            self.test_labels_dir = self.root_dir + "/Labels/Test"

            self.curr_imgs_dir = self.train_imgs_dir

            subdir_to_prefix = {
                'Prostate-3T': 'Prostate3T',
                'PROSTATE-DIAGNOSIS': 'ProstateDx'
            }

            file_names = sum([[f"{subdir}/{f}" for f in
                               os.listdir(f"{self.curr_imgs_dir}/{subdir}") if f.startswith(subdir_to_prefix[subdir])]
                              for
                              subdir in subdir_to_prefix], [])

            test_files = random.sample(file_names, int(len(file_names) * 0.2))
            train_files = [f for f in file_names if f not in test_files]
            self.file_names = train_files if self.train else test_files

            self.idx_to_case_map = dict(enumerate(self.file_names))

            self.processed_files = {
                key: f"{root_dir}/processed/{self.__class__.__name__}_{'train' if self.train else 'test'}_{key}" for key
                in
                self.idx_to_case_map
            }

    def __getitem__(self, idx):
        if os.path.exists(self.processed_files[idx] + '_image.npy'):
            image, label = np.load(self.processed_files[idx] + '_image.npy'), \
                           np.load(self.processed_files[idx] + '_label.npy')
            image = (image - image.min()) / (image.max() - image.min())
            return reshape_to_3d(image), reshape_to_3d(label)
        print(f"Processing {self.idx_to_case_map[idx]}")
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

        # normalize np_scans to be 0-255
        np_scans = 255 * (np_scans - np_scans.min()) / (np_scans.max() - np_scans.min())

        # read labels
        labels_path = [f for f in os.listdir(self.train_labels_dir) if f.startswith(patient_dir.split("/")[-1])][0]
        np_labels, header = nrrd.read(f"{self.train_labels_dir}/{labels_path}")
        np_labels = np.swapaxes(np_labels, 0, 2).astype(float)

        # remove labels of 2
        np_labels = np.where(np_labels == 2, 1, np_labels)

        np_labels, idx = resize_scan(np_labels, MRI_SCAN_SHAPE)

        return resize_scan(np_scans, MRI_SCAN_SHAPE, idx)[0], np_labels


class MedicalSegmentationDecathlon(BaseDataset):
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
        # check if processed files exist
        self.train = train
        self.transform = transform
        if os.path.exists(
                f"{root_dir}/processed/{self.__class__.__name__}_{'train' if self.train else 'test'}_0_image.npy"):
            processed_files_list = glob.glob(
                f"{root_dir}/processed/{self.__class__.__name__}_{'train' if self.train else 'test'}_*_image.npy")
            self.processed_files = {key: f[:-10] for key, f in enumerate(processed_files_list)}

        else:
            print("MedicalSegmentationDecathlon dataset not found. Preprocessing...")
            self.root_dir = root_dir + "/MedicalSegmentationDecathlon/Task05_Prostate"

            self.train_imgs_dir = self.root_dir + "/imagesTr"
            self.train_labels_dir = self.root_dir + "/labelsTr"
            self.test_imgs_dir = self.root_dir + "/imagesTs"

            self.curr_imgs_dir = self.train_imgs_dir

            file_names = [f for f in os.listdir(self.curr_imgs_dir) if f.endswith(".nii")]

            test_files = random.sample(file_names, int(len(file_names) * 0.2))
            train_files = [f for f in file_names if f not in test_files]
            self.file_names = train_files if self.train else test_files

            self.idx_to_case_map = dict(enumerate(self.file_names))

            self.processed_files = {
                key: f"{root_dir}/processed/{self.__class__.__name__}_{'train' if self.train else 'test'}_{key}" for key
                in
                self.idx_to_case_map
            }

    def __getitem__(self, idx):
        if os.path.exists(self.processed_files[idx] + '_image.npy'):
            image, label = np.load(self.processed_files[idx] + '_image.npy'), \
                           np.load(self.processed_files[idx] + '_label.npy')
            image = (image - image.min()) / (image.max() - image.min())
            return reshape_to_3d(image), reshape_to_3d(label)
        print(f"Processing {self.idx_to_case_map[idx]}")
        nii_file_name = self.idx_to_case_map[idx]

        # read scan
        orig_scans = nib.load(f"{self.curr_imgs_dir}/{nii_file_name}")

        # TODO: What is the difference between 0 and 1 at the last index? For now we use 0
        np_scans = orig_scans.get_fdata()[:, :, :, 0]

        # swap axes to match the shape of other Datasets
        np_scans = np.swapaxes(np_scans, 0, 2)

        # normalize np_scans to be 0-255
        np_scans = 255 * (np_scans - np_scans.min()) / (np_scans.max() - np_scans.min())

        # read labels
        orig_labels = nib.load(f"{self.curr_imgs_dir}/{nii_file_name}".replace("imagesTr", "labelsTr"))

        np_labels = orig_labels.get_fdata()
        np_labels = np.swapaxes(np_labels, 0, 2)

        # rmove labels of 2
        np_labels = np.where(np_labels == 2, 1, np_labels)
        np_labels, idx = resize_scan(np_labels, MRI_SCAN_SHAPE)

        return resize_scan(np_scans, MRI_SCAN_SHAPE, idx)[0], np_labels


class Promise12(BaseDataset):
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
        # check if processed files exist
        self.train = train
        self.transform = transform
        if os.path.exists(
                f"{root_dir}/processed/{self.__class__.__name__}_{'train' if self.train else 'test'}_0_image.npy"):
            processed_files_list = glob.glob(
                f"{root_dir}/processed/{self.__class__.__name__}_{'train' if self.train else 'test'}_*_image.npy")
            self.processed_files = {key: f[:-10] for key, f in enumerate(processed_files_list)}

        else:
            print("Promise12 dataset not found. Preprocessing...")
            self.root_dir = root_dir + "/Promise12"
            self.train_dir = self.root_dir + "/Train"
            self.test_dir = self.root_dir + "/Test"

            self.train = train
            self.transform = transform

            self.curr_dir = self.train_dir

            file_names = os.listdir(self.curr_dir)
            nums_in_filenames = sorted(list(set([''.join(filter(lambda i: i.isdigit(), s)) for s in file_names])))

            test_files = random.sample(nums_in_filenames, int(len(nums_in_filenames) * 0.2))
            train_files = [f for f in nums_in_filenames if f not in test_files]
            self.nums_in_filenames = train_files if self.train else test_files

            self.idx_to_case_map = dict(enumerate(self.nums_in_filenames))

            self.processed_files = {
                key: f"{root_dir}/processed/{self.__class__.__name__}_{'train' if self.train else 'test'}_{key}" for key
                in
                self.idx_to_case_map
            }

    def __getitem__(self, idx):
        if os.path.exists(self.processed_files[idx] + '_image.npy'):
            image, label = np.load(self.processed_files[idx] + '_image.npy'), \
                           np.load(self.processed_files[idx] + '_label.npy')
            image = (image - image.min()) / (image.max() - image.min())
            return reshape_to_3d(image), reshape_to_3d(label)
        print(f"Processing {self.idx_to_case_map[idx]}")
        case_idx = self.idx_to_case_map[idx]

        if not os.path.exists(f"{self.curr_dir}/Case{case_idx}.mhd"):
            raise Exception(f"Case {case_idx} not in {'Train' if self.train else 'Test'} directory")

        scans = sitk.GetArrayFromImage(sitk.ReadImage(f"{self.curr_dir}/Case{case_idx}.mhd", sitk.sitkFloat32))

        # normalize np_scans to be 0-255
        # scans = 255 * (scans - scans.min()) / (scans.max() - scans.min())

        label_segmentations = sitk.GetArrayFromImage(
            sitk.ReadImage(f"{self.curr_dir}/Case{case_idx}_segmentation.mhd",
                           sitk.sitkFloat32))
        label_segmentations, idx = resize_scan(label_segmentations, MRI_SCAN_SHAPE)

        return resize_scan(scans, MRI_SCAN_SHAPE, idx)[0], label_segmentations


def process_dataset(dataset, root_dir, processed_dir):
    for train in [True, False]:
        dataset_obj = dataset(
            root_dir=root_dir,
            train=train,
        )
        for idx in range(len(dataset_obj)):
            # get item
            image, label = dataset_obj[idx]
            image, label = reshape_to_3d(image), reshape_to_3d(label)
            # create url
            url = f'{dataset.__name__}_{"train" if train else "test"}_{idx}'
            np.save(f'{processed_dir}/{url}_image.npy', image)
            np.save(f'{processed_dir}/{url}_label.npy', label)


def process_data(root_dir, processed_dir=None):
    if processed_dir is None:
        processed_dir = root_dir + "/processed"
    for dataset in [Promise12, MedicalSegmentationDecathlon, NciIsbi2013, PROSTATEx]:
        process_dataset(dataset, root_dir, processed_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Preprocess datasets for pFedHN for Medical Segmentation - Prostate"
    )

    parser.add_argument("--root-dir", type=str, default="data", help="root directory for datasets")
    parser.add_argument("--processed-dir", type=str, default=None, help="processed data directory")

    args = parser.parse_args()

    process_data(root_dir=args.root_dir, processed_dir=args.processed_dir)
    # test_plot_dataset(Promise12)
    # test_plot_dataset(MedicalSegmentationDecathlon)
    # test_plot_dataset(NciIsbi2013)
    # test_plot_dataset(PROSTATEx)
