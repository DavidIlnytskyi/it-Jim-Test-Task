"""Duplicate removal, validation splitting, and face preprocessing."""


import random
import shutil
from copy import deepcopy
from pathlib import Path

import face_recognition
from PIL import Image
from tqdm import tqdm


def remove_duplicates(duplicates, data_path):
    data_path = Path(data_path)
    duplicate_path = data_path.parent / "duplicates"
    duplicate_path.mkdir(parents=True, exist_ok=True)

    moved = set()

    for org_img, dup_imgs in duplicates.items():
        for dup_img in dup_imgs:
            if dup_img in moved:
                continue

            src = data_path / dup_img
            dst = duplicate_path / dup_img

            if not src.exists():
                continue

            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src, dst)
            moved.add(dup_img)


def count_duplicates(duplicates):
  duplications_count = 0
  duplications_copy = deepcopy(duplicates)

  for org_img, dupl_imgs in duplications_copy.items():
    if len(dupl_imgs) == 0:
      continue

    duplications_count += len(dupl_imgs) + 1

    for duplicated_img in dupl_imgs:
      duplications_copy[duplicated_img] = []

  return duplications_count, duplications_copy


# define new valid size of 10%
def create_valid_split(train_path, valid_fraction=0.1):
  """
  Create validation split without duplications

  """
  valid_dir_path = (train_path.parent/ "valid")
  valid_dir_path.mkdir(exist_ok=True)

  if len(list(valid_dir_path.glob("*.png"))) > 0:
    print("Validation split already exists.")
    return 1

  pos_samples = list(train_path.glob("*0.png"))
  neg_samples = list(train_path.glob("*1.png"))

  valid_pos_samples = random.sample(pos_samples, k = int(valid_fraction * len(pos_samples)))
  valid_neg_samples = random.sample(neg_samples, k = int(valid_fraction * len(neg_samples)))

  for file_path in valid_pos_samples:
    shutil.move(file_path, valid_dir_path)

  for file_path in valid_neg_samples:
    shutil.move(file_path, valid_dir_path)

  print(f"Validation split created: pos samples {len(valid_neg_samples)}; neg samples {len(valid_pos_samples)}")
  return 0


def create_crop_face_data(data_path):
  for split in ["test", "train", "valid"]:
    src_dir = data_path / split
    dst_dir = data_path / (split + "_crop_face")
    preprocess_faces(src_dir, dst_dir)
    print(f"{dst_dir} processed.")


def preprocess_faces(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    dst_dir.mkdir(parents=True, exist_ok=True)
    dir_images_paths = list(src_dir.iterdir())

    for image_path in tqdm(dir_images_paths, total=len(dir_images_paths)):
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)

        if not face_locations:
            continue

        top, right, bottom, left = face_locations[0]

        face = image[top:bottom, left:right]
        face = Image.fromarray(face)

        face.save(dst_dir / image_path.name)
