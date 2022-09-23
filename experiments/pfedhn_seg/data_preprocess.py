import numpy as np

from custom_datasets import Promise12, MedicalSegmentationDecathlon, NciIsbi2013, PROSTATEx


if __name__ == '__main__':
    root_dir = 'data'
    for dataset in [Promise12, MedicalSegmentationDecathlon, NciIsbi2013, PROSTATEx]:
        for train in [True, False]:
            dataset_obj = dataset(
                root_dir=root_dir,
                train=train,
            )
            for idx in range(len(dataset_obj)):
                # get item
                image, label = dataset_obj[idx]
                # create url
                url = f'{dataset.__name__}_{"train" if train else "test"}_{idx}'
                np.save(f'{root_dir}/processed/{url}_image.npy', image)
                np.save(f'{root_dir}/processed/{url}_label.npy', label)
