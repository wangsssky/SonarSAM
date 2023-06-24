# -*- coding: utf-8 -*-
import torch

from torch.utils import data
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from torchvision.transforms.v2 import functional as F
import torchvision.transforms.v2 as transforms
from torchvision import datapoints
from torchvision import io, utils
from torch.utils.data import DataLoader

import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import random
import copy

import xml.etree.ElementTree as ET


class_to_id = {
    "Background": 0,
    "Bottle": 1,
    "Can": 2,            
    "Chain": 3,          
    "Drink-carton": 4,   
    "Hook": 5,           
    "Propeller": 6,      
    "Shampoo-bottle": 7, 
    "Standing-bottle": 8,
    "Tire": 9,           
    "Valve": 10,         
    "Wall": 11 
}


def load_coco_box(xml_path):
    boxes = []
    tree = ET.ElementTree(file=xml_path)
    obj_list = tree.findall('object')
    for obj in obj_list:
        name = obj.find('name').text        
        box = obj.find('bndbox')
        x = int(box.find('x').text)
        y = int(box.find('y').text)
        w = int(box.find('w').text)
        h = int(box.find('h').text)
        boxes.append([x, y, w, h, name])
    return boxes


def imread_cn(path):
    try:
        im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    except Exception as e:
        print(e)
        return None
    return im


class DebrisDataset(Dataset):
    def __init__(self, root_path, image_list, input_size=1000, use_augment=False):
        self.images = []
        self.masks = []
        self.boxes = []
        self.use_augment = use_augment
        self.input_size = input_size
        self.normlize = T.Normalize(mean=[0.1701], std=[0.1848])
        self.transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),                
                transforms.RandomResizedCrop(scale=(0.6, 1), size=self.input_size, 
                                             interpolation=transforms.InterpolationMode.NEAREST),             
                transforms.ColorJitter(contrast=0.2, brightness=0.2),
            ])
        self.transform_test = transforms.Compose([])

        # src path for debris dataset
        image_dir = os.path.join(root_path, 'Images')
        mask_dir = os.path.join(root_path, 'Masks')
        box_dir = os.path.join(root_path, 'BoxAnnotations')

        # load image list
        with open(image_list, 'r') as f:
            lines = f.readlines()

        # load data
        for line in tqdm(lines):
            line = line.strip()
            image = imread_cn(os.path.join(image_dir, line+'.png'))
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            mask = imread_cn(os.path.join(mask_dir, line+'.png'))
            boxes_ori = load_coco_box(xml_path=os.path.join(box_dir, line+'.xml'))

            # padding
            h, w, _ = image.shape
            pad_l = 0
            pad_r = 0
            pad_u = 0
            pad_b = 0
            side_length = max(h, w)
            if h > w:
                pad = h - w
                pad_l = pad // 2
                pad_r = pad - pad_l
            else:
                pad = w - h
                pad_u = pad // 2
                pad_b = pad - pad_u
            
            # resize
            image = cv2.copyMakeBorder(image, pad_u, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0)) # filling mean value
            image = cv2.resize(image, (input_size, input_size), cv2.INTER_CUBIC)  
            mask_list = []          
            for i in range(12):
                m = (mask==i).astype('uint8')
                mask_list.append(m)
                # cv2.imwrite('{}.png'.format(i), m*255)
            for id, m in enumerate(mask_list):
                temp = cv2.copyMakeBorder(m, pad_u, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
                mask_list[id] = cv2.resize(temp, (input_size, input_size), cv2.INTER_NEAREST)
            # mask = cv2.copyMakeBorder(mask, pad_u, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
            # mask = cv2.resize(mask, (input_size, input_size), cv2.INTER_NEAREST)

            self.images.append(Image.fromarray(image))
            self.masks.append(mask_list)
            # for i, mask in enumerate(mask_list):
            #     m = (np.asarray(mask) * 255).astype('uint8')
            #     cv2.imwrite('{}.png'.format(i), m)
            # print(boxes_ori)

            scale_ratio = float(input_size) / side_length
            boxes = []
            for box in boxes_ori:
                x1, y1, w, h, cls = box
                x2 = x1 + w
                y2 = y1 + h
                x1 = int((x1 + pad_l) * scale_ratio)
                y1 = int((y1 + pad_u) * scale_ratio)
                x2 = int((x2 + pad_l) * scale_ratio)               
                y2 = int((y2 + pad_u) * scale_ratio)
                boxes.append([x1, y1, x2, y2, class_to_id[cls]])        
            self.boxes.append(boxes)

        print('loaded {} images'.format(len(self.images)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image = copy.deepcopy(self.images[index])
        mask = copy.deepcopy(self.masks[index])
        boxes = copy.deepcopy(self.boxes[index])

        
        boxes_ = copy.deepcopy(boxes)
        boxes = []
        labels = []
        
        for box in boxes_:
            x1, y1, x2, y2, cls_id = box
            boxes.append([x1, y1, x2, y2])
            labels.append(cls_id)
        image = datapoints.Image(image)
        
        temp = np.zeros_like(mask[0])
        for idx, m in enumerate(mask):
            temp += idx * m
        mask = datapoints.Mask(Image.fromarray(temp))
        bboxes = datapoints.BoundingBox(boxes,
                                        format=datapoints.BoundingBoxFormat.XYXY,
                                        spatial_size=F.get_spatial_size(image),
                                        )
        if self.use_augment:
            image, bboxes, mask, labels = self.transform_train(image, bboxes, mask, labels)
        else:
            image, bboxes, mask, labels = self.transform_test(image, bboxes, mask, labels)

        boxes = []
        for box, label in zip(bboxes, labels):
            x1, y1, x2, y2 = box
            boxes.append([x1, y1, x2, y2, label])

        image = T.ToTensor()(F.to_image_pil(image))
        mask = np.asarray(F.to_image_pil(mask))
        masks = []
        for id in range(12):
            m = (mask == id).astype('uint8')
            # masks.append(np.asarray(mask[id]))
            masks.append(m)
        masks = np.stack(masks, axis=0)
        masks = torch.tensor(masks, dtype=torch.float32)
             
        # boxes = torch.tensor(boxes, dtype=torch.long)
        box_warpper = {'boxes': boxes}

        # normalize
        image = self.normlize(image)

        return image, masks, box_warpper 

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)
    

class SAM_DebrisDataset(Dataset):
    def __init__(self, root_path, image_list, input_size=1000, use_augment=False):
        self.images = []
        self.masks = []
        self.boxes = []
        self.use_augment = use_augment
        self.input_size = input_size
        self.normlize = T.Normalize(mean=[0.1701], std=[0.1848])
        self.transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),                
                transforms.RandomResizedCrop(scale=(0.6, 1), size=self.input_size, 
                                             interpolation=transforms.InterpolationMode.NEAREST),             
                transforms.ColorJitter(contrast=0.2, brightness=0.2),
            ])
        self.transform_test = transforms.Compose([])

        # src path for debris dataset
        image_dir = os.path.join(root_path, 'Images')
        mask_dir = os.path.join(root_path, 'Masks')
        box_dir = os.path.join(root_path, 'BoxAnnotations')

        # load image list
        with open(image_list, 'r') as f:
            lines = f.readlines()

        # load data
        for line in tqdm(lines):
            line = line.strip()
            image = imread_cn(os.path.join(image_dir, line+'.png'))
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            mask = imread_cn(os.path.join(mask_dir, line+'.png'))
            boxes_ori = load_coco_box(xml_path=os.path.join(box_dir, line+'.xml'))            

            # padding
            h, w, _ = image.shape
            pad_l = 0
            pad_r = 0
            pad_u = 0
            pad_b = 0
            side_length = max(h, w)
            if h > w:
                pad = h - w
                pad_l = pad // 2
                pad_r = pad - pad_l
            else:
                pad = w - h
                pad_u = pad // 2
                pad_b = pad - pad_u
            
            # resize
            image = cv2.copyMakeBorder(image, pad_u, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0)) # filling mean value
            image = cv2.resize(image, (input_size, input_size), cv2.INTER_CUBIC)  
            mask_list = []          
            for i in range(12):
                m = (mask==i).astype('uint8')
                mask_list.append(m)
                # cv2.imwrite('{}.png'.format(i), m*255)
            for id, m in enumerate(mask_list):
                temp = cv2.copyMakeBorder(m, pad_u, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
                mask_list[id] = cv2.resize(temp, (input_size, input_size), cv2.INTER_NEAREST)
            # mask = cv2.copyMakeBorder(mask, pad_u, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
            # mask = cv2.resize(mask, (input_size, input_size), cv2.INTER_NEAREST)

            self.images.append(Image.fromarray(image))
            self.masks.append(mask_list)
            # for i, mask in enumerate(mask_list):
            #     m = (np.asarray(mask) * 255).astype('uint8')
            #     cv2.imwrite('{}.png'.format(i), m)
            # print(boxes_ori)

            scale_ratio = float(input_size) / side_length
            boxes = []
            for box in boxes_ori:
                x1, y1, w, h, cls = box
                x2 = x1 + w
                y2 = y1 + h
                x1 = int((x1 + pad_l) * scale_ratio)
                y1 = int((y1 + pad_u) * scale_ratio)
                x2 = int((x2 + pad_l) * scale_ratio)               
                y2 = int((y2 + pad_u) * scale_ratio)
                boxes.append([x1, y1, x2, y2, class_to_id[cls]])        
            self.boxes.append(boxes)

        print('loaded {} images'.format(len(self.images)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image = copy.deepcopy(self.images[index])
        mask = copy.deepcopy(self.masks[index])
        boxes = copy.deepcopy(self.boxes[index])

        
        boxes_ = copy.deepcopy(boxes)
        boxes = []
        labels = []
        
        for box in boxes_:
            x1, y1, x2, y2, cls_id = box
            boxes.append([x1, y1, x2, y2])
            labels.append(cls_id)
        image = datapoints.Image(image)
        
        temp = np.zeros_like(mask[0])
        for idx, m in enumerate(mask):
            temp += idx * m
        mask = datapoints.Mask(Image.fromarray(temp))
        bboxes = datapoints.BoundingBox(boxes,
                                        format=datapoints.BoundingBoxFormat.XYXY,
                                        spatial_size=F.get_spatial_size(image),
                                        )
        if self.use_augment:
            image, bboxes, mask, labels = self.transform_train(image, bboxes, mask, labels)
        else:
            image, bboxes, mask, labels = self.transform_test(image, bboxes, mask, labels)

        image = T.ToTensor()(F.to_image_pil(image))
        mask = np.asarray(F.to_image_pil(mask))
        
        boxes, masks = [], []
        for box, label in zip(bboxes, labels):
            x1, y1, x2, y2 = box            
            boxes.append([x1, y1, x2, y2, label])
            
            m = np.zeros_like(mask)
            m[y1:y2, x1:x2] = (mask==label)[y1:y2, x1:x2]
            masks.append(torch.tensor(m, dtype=torch.float32))
          
        warpper = {'boxes': boxes, 'masks': masks}

        # normalize
        image = self.normlize(image)

        return image, warpper 

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)
   
   
def collate_fn_seq(batch):
    images = [ item[0] for item in batch ]
    masks = [ item[1] for item in batch ]
    targets = [ item[2] for item in batch ]

    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)

    tars = []

    for i in range(len(batch)):
        tars.append(targets)

    return images, masks, tars


def collate_fn_seq_box_seg_pair(batch):
    images = [ item[0] for item in batch ]
    targets = [ item[1] for item in batch ]

    images = torch.stack(images, dim=0)

    tars = []

    for i in range(len(batch)):
        tars.append(targets)

    return images, tars


if __name__ == '__main__':
    root_dir = "/data/wanglin/Data/marine-debris-fls-datasets/md_fls_dataset/data/watertank-segmentation"
    image_list = '/data/wanglin/Code/SAM_Sonar/dataset/marine_debris/test.txt'
    dataset = SAM_DebrisDataset(root_path=root_dir, image_list=image_list, input_size=1024, use_augment=True)
    
    # for (image, mask, boxes) in dataset:    
    #     print('image shape:', image.shape)
    #     cv_image = image.permute(1, 2, 0).contiguous().numpy()
    #     mean = 0.1701
    #     std = 0.1848
    #     cv_image = ((cv_image * std + mean) * 255).astype('uint8')

    #     boxes = boxes['boxes']
    #     for box in boxes:
    #         box = box.numpy()
    #         x1, y1, x2, y2, _ = box
    #         cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    #     cv2.imwrite('tmp.png', cv_image)

    #     for i in range(mask.shape[0]):
    #         m = (mask[i, ...].numpy() * 255).astype('uint8')
    #         cv2.imwrite('{}.png'.format(i), m)
        
    #     print('mask shape:', mask.shape)
    #     print('boxes:\n', boxes)
        
    test_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn_seq_box_seg_pair)
    for (image, box_seg_pairs) in test_loader:
        print(box_seg_pairs)