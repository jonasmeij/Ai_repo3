import torch
from torch.utils.data import Dataset
import h5py


class GateDetectionDataset(Dataset):
    def __init__(self, h5_file_path):
        """
        Initialize the dataset by loading the HDF5 file.

        Args:
            h5_file_path (str): Path to the HDF5 file containing images and targets.
        """
        # Open the HDF5 file
        self.h5f = h5py.File(h5_file_path, "r")

        # Load images and targets into the dataset
        self.images = self.h5f["images"]
        self.targets = [
            self.h5f[f"targets/{i:05d}"][()] for i in range(len(self.images))
        ]

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            image (torch.Tensor): The image as a torch tensor.
            target (torch.Tensor): The corresponding target for the image.
        """
        # Load image and target
        image = self.images[idx]  # Image as numpy array
        target = self.targets[idx]  # Target as numpy array

        # Convert to torch tensors
        image = torch.tensor(image, dtype=torch.float32)  # Convert to float tensor
        target = torch.tensor(target, dtype=torch.float32)  # Convert to float tensor

        return image, target

    def __del__(self):
        """Close the HDF5 file when the object is deleted."""
        self.h5f.close()
