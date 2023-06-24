# -*- coding: utf-8 -*-
import torch

from torch.utils import data
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from load_xml import load_coco_box


def imread_cn(path):
    try:
        im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    except Exception as e:
        print(e)
        return None
    return im


def calc_mean_std(root_path, image_list):
    images = []
    image_dir = os.path.join(root_path, 'Images')

    with open(image_list, 'r') as f:
            lines = f.readlines()

    for line in tqdm(lines):
        line = line.strip()
        image = imread_cn(os.path.join(image_dir, line+'.png'))
        images.append(image)

    data = np.stack(images).flatten()
    data = data / 255.0
    mean, std = np.mean(data), np.std(data)
    return mean, std


if __name__ == '__main__':
    root_dir = "/data/wanglin/Data/marine-debris-fls-datasets/md_fls_dataset/data/watertank-segmentation"
    image_list = './dataset/marine_debris/train.txt'
    mean, std = calc_mean_std(root_dir, image_list)
    print(mean, std)

    # 0.1701 0.1848