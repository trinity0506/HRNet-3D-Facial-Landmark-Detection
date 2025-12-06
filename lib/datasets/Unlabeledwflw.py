import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import random
import torch

class UnlabeledWFLW(Dataset):
    def __init__(self, img_dir, transform=None):
        self.image_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = read_image(img_path).float() / 255.0  # 归一化到[0,1]
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.image_paths))

        # 数据增强
        if self.transform is not None:
            image = self.transform(image)
        
        # 额外增强（可选）
        if self.training:  # 注意需要判断是否处于训练模式
            # 随机缩放
            scale = 1.0 + random.uniform(-self.scale_factor, self.scale_factor)
            new_size = (int(image.shape[1]*scale), int(image.shape[2]*scale))
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=new_size, mode='bilinear', align_corners=False
            )[0]

            # 随机旋转（90度倍数）
            angle = random.uniform(-self.rot_factor, self.rot_factor)
            k = int(angle / 90) % 4  # 限制旋转次数不超过4次
            image = torch.rot90(image, k=k, dims=[1, 2])

            # 随机水平翻转
            if random.random() < self.flip_prob:
                image = torch.flip(image, [2])

        # 标准化
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)

        return image  

