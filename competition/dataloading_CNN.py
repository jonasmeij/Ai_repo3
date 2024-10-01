import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os
import torch.nn.functional as F
import hdf5plugin  # Required for reading HDF5 files with GZIP compression


class DroneDataset(Dataset):
    def __init__(self, data_dir, max_gates=5):
        self.data_dir = data_dir
        self.max_gates = max_gates
        self.images = []
        self.targets = []

        # Iterate over all HDF5 files in the directory
        for file_name in os.listdir(data_dir):
            if file_name.endswith(".h5"):
                file_path = os.path.join(data_dir, file_name)
                self._load_data_from_file(file_path)

        # Ensure data was loaded before concatenation
        if len(self.targets) > 0:
            # Convert list of dictionaries to separate lists of arrays
            coords_list = [t["coords"] for t in self.targets]
            conf_list = [t["conf"] for t in self.targets]

            # Ensure all confidence arrays have the same shape
            for i, conf in enumerate(conf_list):
                if conf.shape[0] < self.max_gates:
                    conf_list[i] = np.pad(
                        conf,
                        (0, self.max_gates - conf.shape[0]),
                        mode="constant",
                        constant_values=0,
                    )
                elif conf.shape[0] > self.max_gates:
                    conf_list[i] = conf[: self.max_gates]

            # Convert to numpy arrays for efficient indexing
            self.images = np.concatenate(self.images, axis=0)
            self.targets_coords = np.stack(coords_list, axis=0)
            self.targets_conf = np.stack(conf_list, axis=0)
        else:
            raise ValueError("No valid targets were loaded. Check your data files.")

    def _load_data_from_file(self, file_path):
        with h5py.File(file_path, "r") as h5f:
            if "images" in h5f:
                images = h5f["images"][:]
            else:
                return

            targets = []

            for i in range(len(images)):
                try:
                    if f"targets/{i:05d}" in h5f:
                        gate = h5f[f"targets/{i:05d}"][()]

                        # Create a confidence array for existing gates
                        conf = np.ones(gate.shape[0])

                        # Pad gt_gates and confidence to have the same number of gates
                        if gate.shape[0] < self.max_gates:
                            # Ensure both gate and conf are padded to match max_gates
                            padding = ((0, self.max_gates - gate.shape[0]), (0, 0))
                            gate = np.pad(
                                gate, padding, mode="constant", constant_values=0
                            )
                            conf = np.pad(
                                conf,
                                (0, self.max_gates - gate.shape[0]),
                                mode="constant",
                                constant_values=0,
                            )
                        # If the gate shape is greater than max_gates, truncate it
                        elif gate.shape[0] > self.max_gates:
                            gate = gate[: self.max_gates, :]
                            conf = conf[: self.max_gates]

                        # Combine gate coordinates and confidence into a single array
                        targets.append({"coords": gate, "conf": conf})
                except KeyError:
                    continue

            # Append data to class lists only if there are valid targets
            if len(targets) > 0:
                self.images.append(images)
                self.targets.extend(
                    targets
                )  # Use extend instead of append for dictionaries

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target_dict = self.targets[idx]
        coords = torch.tensor(target_dict["coords"], dtype=torch.float32)
        conf = torch.tensor(target_dict["conf"], dtype=torch.float32)

        # Ensure the confidence is padded correctly in case of missing gates
        if conf.shape[0] < self.max_gates:
            conf = F.pad(conf, (0, self.max_gates - conf.shape[0]), "constant", 0)

        targets_dict = {"coords": coords, "conf": conf}
        return torch.tensor(image, dtype=torch.float32), targets_dict


# Testing the modified class (you can remove this part in your main code)
if __name__ == "__main__":
    data_path = "/workspaces/AE4353-Y24/competition/data/Autonomous"
    batch_size = 1  # Example batch size

    # Initialize the dataset
    dataset = DroneDataset(data_path)

    # Check the dataset length and first item
    print(f"Dataset length: {len(dataset)}")

    # Create DataLoader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Fetch a batch of data
    for batch_idx, (images, targets) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print(f"Images shape: {images.shape}")
        print(f"Targets shape: {targets['coords'].shape}")
        print(f"Confidence shape: {targets['conf'].shape}")
        print("Coordinates:", targets["coords"])
        print("Confidence:", targets["conf"])

        break  # Fetch only the first batch for demonstration
