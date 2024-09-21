import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import hdf5plugin
import os


class DroneDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_info = []
        h5_files = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".h5")
        ]

        for data_path in h5_files:
            with h5py.File(data_path, "r") as file:
                num_images = len(file["images"])
                for i in range(num_images):
                    self.data_info.append((data_path, i))

        # targets have 12 values with folowing data:
        # x1, y1, v1: Coordinates and visibility flag for the first point of Gate 1.
        # x2, y2, v2: Coordinates and visibility flag for the second point of Gate 1.
        # x3, y3, v3: Coordinates and visibility flag for the third point of Gate 1.
        # x4, y4, v4: Coordinates and visibility flag for the fourth point of Gate 1.

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        data_path, image_idx = self.data_info[idx]
        with h5py.File(data_path, "r") as file:
            image = file["images"][image_idx]
            target = file[f"targets/{image_idx:05d}"][()]

        return torch.tensor(image, dtype=torch.float32), torch.tensor(
            target, dtype=torch.float32
        )


if __name__ == "__main__":
    data_path = "/workspaces/AE4353-Y24/competition/data/Autonomous"
    dataset = DroneDataset(data_path)

    # Access the first frame and its target
    first_frame, first_target = dataset[0]
    print(first_frame.shape, first_target.shape)
    print(first_target)
