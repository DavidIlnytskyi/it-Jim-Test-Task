"""Image dataset and sampling helpers."""


from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import WeightedRandomSampler
from torchvision.transforms.functional import to_tensor


class customImageDataset(nn.Module):
  def __init__(self, data_path, transform=None):
    super().__init__()
    self.data_path = Path(data_path)
    self.img_paths = list(self.data_path.iterdir())
    self.transform = transform

  def __len__(self):
    return len(self.img_paths)

  def __getitem__(self, idx):
    img_path = self.img_paths[idx]
    image = Image.open(img_path)
    cls = int(img_path.stem.split("_")[-1])
    if self.transform:
      image = self.transform(image)
    else:
      image = to_tensor(image)

    return image, cls

  def get_labels(self):
    return [int(img_path.stem.split("_")[-1]) for img_path in self.img_paths]


def create_balanced_sampler(dataset):
  class_weights = torch.tensor([9.0, 1.0])

  labels = torch.tensor(dataset.get_labels())

  sample_weights = class_weights[labels]

  sampler = WeightedRandomSampler(
      sample_weights,
      num_samples=len(labels),
      replacement=True,
  )
  return sampler

def create_oversampling_sampler(dataset):
    labels = torch.tensor(dataset.get_labels())

    class_counts = torch.bincount(labels)
    class_weights = 1.0 / class_counts.float()

    sample_weights = class_weights[labels]

    num_samples = class_counts.max().item() * len(class_counts)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,
    )

    return sampler
