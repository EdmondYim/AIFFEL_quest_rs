import os
from glob import glob
import numpy as np
import torch
from imageio import imread
from torch.utils.data import Dataset


class KittiDataset(Dataset):
    """
    KITTI-style road dataset loader.
    If num_classes <= 2, masks are binarized (road class 7 -> 1, else 0).
    Otherwise, raw semantic labels are kept; unlabeled(255) can be mapped to ignore_index.
    """

    def __init__(
        self,
        dir_path,
        is_train=True,
        augmentation=None,
        num_classes=2,
        split_ratio=0.9,
        ignore_index=-100,
    ):
        self.dir_path = dir_path
        self.is_train = is_train
        self.augmentation = augmentation
        self.num_classes = num_classes
        self.split_ratio = split_ratio
        self.ignore_index = ignore_index

        self.data = self.load_dataset()

    def load_dataset(self):
        input_images = sorted(glob(os.path.join(self.dir_path, "image_2", "*.png")))
        label_images = sorted(glob(os.path.join(self.dir_path, "semantic", "*.png")))

        assert len(input_images) == len(label_images), (
            f"Image count({len(input_images)}) and label count({len(label_images)}) differ"
        )

        data = list(zip(input_images, label_images))
        total_len = len(data)
        train_len = int(total_len * self.split_ratio)

        if self.is_train:
            return data[:train_len]
        return data[train_len:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_img_path, output_path = self.data[index]

        _input = imread(input_img_path)
        _output = imread(output_path)

        if self.num_classes <= 2:
            # Binary road mask: road(class 7)=1, background=0
            _output = (_output == 7).astype(np.uint8)
        else:
            _output = _output.astype(np.int64)
            if self.ignore_index is not None:
                # Map unlabeled 255 to ignore_index for losses/metrics
                _output[_output == 255] = self.ignore_index

        data = {"image": _input, "mask": _output}

        if self.augmentation:
            augmented = self.augmentation(**data)
            _input = augmented["image"] / 255.0
            _output = augmented["mask"]
        else:
            _input = _input / 255.0

        return (
            torch.tensor(_input, dtype=torch.float32).permute(2, 0, 1),
            torch.tensor(_output, dtype=torch.long),
        )

    def shuffle_data(self):
        if self.is_train:
            np.random.shuffle(self.data)
