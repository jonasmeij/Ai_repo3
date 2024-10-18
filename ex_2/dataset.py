import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os


class PolImgDataset(Dataset):
    def __init__(self, dataset_path, prefix="", h5=False, augment=False):
        self.h5 = h5
        self.augment = augment
        if self.h5:
            self.file = h5py.File(
                os.path.join(dataset_path, prefix + "dataset.h5"), "r"
            )
            self.maps = self.file["maps"]
            self.vector = self.file["labels"]
            self.angles = torch.tensor(self.file["angles"][:])
        else:
            self.maps = torch.tensor(
                np.load(os.path.join(dataset_path, prefix + "maps.npy"))
            )
            self.vector = torch.tensor(
                np.load(os.path.join(dataset_path, prefix + "labels.npy"))
            )
            angles = torch.rad2deg(torch.atan2(self.vector[:, 1], self.vector[:, 0]))
            self.angles = torch.remainder(angles, 360)

    def __len__(self):
        return len(self.vector)

    def __getitem__(self, idx):
        if self.h5:
            maps, vector, angles = (
                torch.tensor(self.maps[idx]),
                torch.tensor(self.vector[idx]),
                torch.tensor(self.angles[idx]),
            )
        else:
            maps, vector, angles = self.maps[idx], self.vector[idx], self.angles[idx]

        if self.augment:
<<<<<<< HEAD
            swap_45_135 = False
            if self.augment and torch.rand(1) < 0.5:
                # Horizontal Flip
                maps = maps.flip(2)
                vector = vector * torch.tensor([-1, 1])
                angles = torch.remainder(180 - angles, 360)
                swap_45_135 = not swap_45_135
            if self.augment and torch.rand(1) < 0.5:
                # Vertical Flip
                maps = maps.flip(1)
                swap_45_135 = not swap_45_135
                vector = vector * torch.tensor([1, -1])
                angles = torch.remainder(360 - angles, 360)

            if swap_45_135:
                maps = maps.index_select(0, torch.tensor([0, 3, 2, 1]))
=======

            # Horizontal flip
            if random.random() < 0.5:
                maps = torch.flip(maps, dims=[2])
                maps = maps[[0, 3, 2, 1], :, :]  # Swap I45 and I135
                vector[0] = -vector[0]  # Flip x component of vector
                angles = torch.remainder(360 - angles, 360)  # Adjust angle

            # Vertical flip
            if random.random() < 0.5:
                maps = torch.flip(maps, dims=[1])
                maps = maps[[0, 3, 2, 1], :, :]  # Swap I45 and I135
                vector[1] = -vector[1]  # Flip y component of vector
                angles = torch.remainder(360 - angles, 360)  # Adjust angle
>>>>>>> 8dfd9e0 (t)

        return maps, vector, angles


if __name__ == "__main__":
    dataset = PolImgDataset(
        "/workspaces/msc-ai-course/data/polarization_dataset", h5=False, augment=True
    )
    sample = dataset[5]
    print(sample)
