import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import load_volume


def pad_image(image, input_size, output_size):
    """zero-pad a volume for input context and the minimum output size"""
    padding = [((before - after) // 2, (before - after) // 2 + max(0, after - size))
               for size, before, after in zip(image.shape, input_size, output_size)]
    return np.pad(image, padding, mode="constant")


class PatchDataset(Dataset):
    def __init__(self, cases, settings):
        """store scan cases and input and output patch sizes"""
        self.cases = cases
        self.input_size = settings["patch_size"]
        self.output_size = settings["output_patch_size"]

    def __len__(self):
        """return the number of scans"""
        return len(self.cases)

    def __getitem__(self, index):
        """sample an image patch and labels with padded voxels marked to ignore"""
        item = load_volume(self.cases[index])
        shape = item["image"].shape
        starts = [np.random.randint(max(size - length, 0) + 1)
                  for size, length in zip(shape, self.output_size)]
        image = pad_image(item["image"], self.input_size, self.output_size)
        
        label = np.pad(item["label"].astype(np.int64),
                       [(0, max(0, length - size)) for size, length in zip(shape, self.output_size)],
                       constant_values=-100)
        input_region = tuple(slice(start, start + size) for start, size in zip(starts, self.input_size))
        output_region = tuple(slice(start, start + size) for start, size in zip(starts, self.output_size))
        return {"image": torch.from_numpy(np.ascontiguousarray(image[input_region][None], dtype=np.float32)),
                "label": torch.from_numpy(np.ascontiguousarray(label[output_region][None]))}
