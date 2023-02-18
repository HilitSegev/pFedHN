# DEFAULTS
import json

from monai.data import CacheDataset, Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, DivisiblePadd, \
    RandCropByPosNegLabeld, RandFlipd, NormalizeIntensityd, RandScaleIntensityd, RandShiftIntensityd, EnsureTyped, \
    KSpaceSpikeNoised, GibbsNoised


def get_datasets(data_name, root_dir):
    """
    get_datasets returns train/val/test data splits of our relevant datasets
    :param data_name:
    :param root_dir:
    :return:
    """
    assert root_dir.endswith("/"), "root_dir must end with '/'"

    dataset_obj = Dataset

    # get data_lists for the client
    path_to_datalists = root_dir + 'datalist/'  # folder of json files
    if data_name != "NCI_ISBI":
        datalist_json_path = path_to_datalists + 'client_' + data_name + ".json"
        with open(datalist_json_path, "r") as f:
            data_list = json.load(f)

        train_list = data_list["training"]
        valid_list = data_list["validation"]
        test_list = data_list["testing"]
    else:
        # combine NCI_ISBI_Dx and NCI_ISBI_3T
        train_list = []
        valid_list = []
        test_list = []
        for dn in ["_Dx", "_3T"]:
            datalist_json_path = path_to_datalists + 'client_NCI_ISBI' + dn + ".json"
            with open(datalist_json_path, "r") as f:
                data_list = json.load(f)

            train_list += data_list["training"]
            valid_list += data_list["validation"]
            test_list += data_list["testing"]

    # update data_lists to include full path
    for data_list in [train_list, valid_list, test_list]:
        for data in data_list:
            data["image"] = root_dir + "dataset/" + data["image"]
            data["label"] = root_dir + "dataset/" + data["label"]
    print("read data lists from json files.")

    # # try overfitting on a single image
    # print("========================================")
    # train_list = [{'image': '/dsi/shared/hilita/ProstateSegmentation/dataset/Promise12/Image/Case28.nii.gz',
    #                'label': '/dsi/shared/hilita/ProstateSegmentation/dataset/Promise12/Mask/Case28.nii.gz'}]
    # valid_list = train_list
    # test_list = train_list
    # print("overfitting on a single image.")
    # print("========================================")

    # add noise params
    noise_params = {
        "Promise12": GibbsNoised(keys=["image"], alpha=0.2),
        "MSD": GibbsNoised(keys=["image"], alpha=0),
        "NCI_ISBI": KSpaceSpikeNoised(keys=["image"],
                                      loc=(10, 10, 10),
                                      k_intensity=13),
        "PROSTATEx": GibbsNoised(keys=["image"], alpha=0.5)
    }
    NoiseTransform = noise_params[data_name]

    # define transforms
    transform_train = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            NoiseTransform,
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(0.3, 0.3, 1.0),
                mode=("bilinear", "nearest"),
            ),
            DivisiblePadd(keys=["image", "label"], k=32),
            # RandSpatialCropSamplesd(
            #     keys=["image", "label"],
            #     roi_size=[160,160,32],
            #     random_size=False,
            #     num_samples=3,
            # ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(160, 160, 32),
                pos=1,
                neg=1,
                num_samples=4,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    transform_valid_test = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(
                keys=["image", "label"],
                pixdim=(0.3, 0.3, 1.0),
                mode=("bilinear", "nearest"),
            ),
            DivisiblePadd(keys=["image", "label"], k=32),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    print("transforms defined")
    # create datasets
    train_ds = dataset_obj(
        data=train_list,
        transform=transform_train,
    )
    valid_ds = dataset_obj(
        data=valid_list,
        transform=transform_valid_test,
    )

    test_ds = dataset_obj(
        data=test_list,
        transform=transform_valid_test,
    )
    print("datasets created")
    return train_ds, valid_ds, test_ds
