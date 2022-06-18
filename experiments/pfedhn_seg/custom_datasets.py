from torch.utils.data.dataset import Dataset
import SimpleITK as sitk
import matplotlib.pylab as plt
import os

from torchvision.transforms import transforms


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

        # TODO: maybe it's better to use subdirectories for each case
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

        if self.train:
            label_segmentations = sitk.GetArrayFromImage(
                sitk.ReadImage(f"{self.curr_dir}/Case{case_idx}_segmentation.mhd",
                               sitk.sitkFloat32))
        else:
            label_segmentations = None

        return scans, label_segmentations


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

    train_scans, train_seg = dataset[1]
    test_scans, test_seg = test_set[1]

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


if __name__ == '__main__':
    test_plot_dataset(Promise12)
