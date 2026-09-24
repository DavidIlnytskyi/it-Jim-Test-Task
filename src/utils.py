"""Plotting and training-result persistence helpers."""


import json
import random
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from .constants import NEGATIVE_CLASS, POSITIVE_CLASS


def plot_losses(train_losses, valid_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, valid_losses, label="Valid Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()


def plot_metric(metric_values, metric_name="F1"):
    epochs = range(1, len(metric_values) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metric_values, label=metric_name)

    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"Validation {metric_name}")
    plt.legend()
    plt.grid()
    plt.show()


def show_random_images(dir_path, n=3):
    paths_with_art = list(Path(dir_path).glob(f"*{0}.png"))
    paths_without_art = list(Path(dir_path).glob(f"*{1}.png"))

    artifacted = random.sample(paths_with_art, min(n, len(paths_with_art)))
    artifactless = random.sample(paths_without_art, min(n, len(paths_without_art)))

    fig, axes = plt.subplots(n, 2, figsize=(10,10))
    axes[0][0].set_title('Artifactless (1) class')
    axes[0][1].set_title('Artifact (0) class')
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(n):
        ax1, ax2 = axes[i]
        ax1.imshow(Image.open(artifactless[i]))
        ax2.imshow(Image.open(artifacted[i]))
        ax1.axis('off')
        ax2.axis('off')


    plt.show()

def plot_class_distribution(data_path):
  positive_samples = list((data_path).glob(f"*{POSITIVE_CLASS}.png"))
  negative_samples = list((data_path).glob(f"*{NEGATIVE_CLASS}.png"))
  names = ['Positive', 'negative']
  values = [len(positive_samples), len(negative_samples)]

  plt.bar(names, values)
  plt.title(f'Class distribution, neg/pos ratio: {len(negative_samples)/len(positive_samples)}')
  plt.xlabel('Class')
  plt.ylabel('Number of samples')
  plt.show()

def plot_image(path):
  plt.axis('off')
  plt.imshow(Image.open(path))
  plt.show()


def save_model(cfg, save_name):
    save_path = Path(cfg.save_dir + "/" + cfg.name)
    save_path.mkdir(exist_ok=True, parents=True)
    torch.save(cfg.model.state_dict(), save_path / save_name)

def save_callback(*args, **kwargs):
  if kwargs["min_valid_loss"] > kwargs["valid_loss"]:
    save_model(kwargs["cfg"], "best.pt")


def save_training_data(data, save_path):
  with open(save_path + "/training_results.json", "w") as f:
      json.dump(asdict(data), f)
