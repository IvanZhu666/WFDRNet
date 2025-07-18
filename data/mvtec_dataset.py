import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

from .data_utils import perlin_noise


class MVTecDataset(Dataset):
    def __init__(
        self,
        is_train: bool,
        mvtec_dir: str,
        resize_shape: list = [256, 256],
        normalize_mean: list = [0.485, 0.456, 0.406],
        normalize_std: list = [0.229, 0.224, 0.225],
        dtd_dir: str = None,
        rotate_90: bool = False,
        random_rotate: int = 0,
    ):
        super().__init__()
        self.is_train = is_train
        self.resize_shape = resize_shape
        self.rotate_90 = rotate_90
        self.random_rotate = random_rotate

        self.mvtec_paths = sorted(
            glob.glob(os.path.join(mvtec_dir, "*.png" if is_train else "*/*.png"))
        )

        if is_train:
            self.dtd_paths = sorted(glob.glob(os.path.join(dtd_dir, "*/*.jpg")))
        else:
            self.mask_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(
                    (resize_shape[1], resize_shape[0]),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True
                )
            ])

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ])

    def __len__(self):
        return len(self.mvtec_paths)

    def _load_image(self, path: str) -> Image.Image:
        image = Image.open(path).convert("RGB")
        return image.resize(self.resize_shape, Image.BILINEAR)

    def _apply_rotation(self, image: Image.Image) -> Image.Image:
        fill_color = (114, 114, 114)
        if self.rotate_90:
            degree = np.random.choice([0, 90, 180, 270])
            image = image.rotate(degree, resample=Image.BILINEAR, fillcolor=fill_color)
        if self.random_rotate > 0:
            degree = np.random.uniform(-self.random_rotate, self.random_rotate)
            image = image.rotate(degree, resample=Image.BILINEAR, fillcolor=fill_color)
        return image

    def _load_mask(self, image_path: str) -> torch.Tensor:
        dir_path, file_name = os.path.split(image_path)
        category = os.path.basename(dir_path)
        if category == "good":
            mask = torch.zeros((1, *self.resize_shape))
        else:
            mask_name = file_name.rsplit('.', 1)[0] + "_mask.png"
            mask_path = os.path.join(dir_path, "../../ground_truth", category, mask_name)
            mask = Image.open(mask_path)
            mask = self.mask_transform(mask)
            mask = torch.where(mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask))
        return mask

    def __getitem__(self, index):
        image = self._load_image(self.mvtec_paths[index])

        if self.is_train:
            dtd_image = self._load_image(
                self.dtd_paths[torch.randint(0, len(self.dtd_paths), (1,)).item()]
            )

            image = self._apply_rotation(image)
            aug_image, aug_mask = perlin_noise(image, dtd_image, aug_probability=1.0)

            return {
                "img_aug": self.image_transform(aug_image),
                "img_origin": self.image_transform(image),
                "mask": aug_mask,
            }
        else:
            image = self.image_transform(image)
            mask = self._load_mask(self.mvtec_paths[index])

            return {
                "img": image,
                "mask": mask,
            }
