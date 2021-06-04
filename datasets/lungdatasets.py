import torch
from torch.utils.data import Dataset
import albumentations as A
import cv2
import numpy as np
import json
from utils.utils import annotation2binarymask

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

class SchenzenMontgomeryLungSegmentationDataset(Dataset):
    def __init__(self, df, data_dir, aug_transform, test=False, mean=MEAN,std=STD):
        self.df = df
        self.data_dir = data_dir
        self.test = test
        self.aug_transform = aug_transform
        self.norm_transform = A.Normalize(mean=mean, std=std)

    def __getitem__(self, idx):
        if self.test:
            img_path = self.data_dir + 'data/test/' + self.df.iloc[idx, 0] + ".png"
        else:
            img_path = self.data_dir + 'data/CXR_png/' + self.df.iloc[idx, 0] + ".png"

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.test:
            augmented = self.aug_transform(image=img)
            img = augmented['image']
            img = self.norm_transform(image=img)["image"]
            return torch.FloatTensor(img)
        else:
            ending = '_mask.png' if self.df.iloc[idx, 0].startswith('CHN') else '.png'
            mask_path = self.data_dir + 'data/masks/' + self.df.iloc[idx, 0] + ending
            mask = cv2.imread(mask_path)[:, :, 0]
            mask = np.clip(mask, 0, 1).astype("float32")

            augmented = self.aug_transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            img = self.norm_transform(image=img)["image"]
            return torch.FloatTensor(img), torch.FloatTensor(mask)

    def __len__(self):
        return len(self.df)

class CheXpertLungSegmentationDataset(Dataset):
    def __init__(self, data_dir, image_dir, aug_transform, mean=MEAN,std=STD, test = False):
        with open(data_dir, "r") as read_file:
            self.data = json.load(read_file)
        print(data_dir)
        self.image_dir = image_dir
        self.data_dir = data_dir
        self.aug_transform = aug_transform
        self.test = test
        self.norm_transform = A.Normalize(mean=mean, std=std)

    def __getitem__(self, idx):
        whole_data = self.data['images']
        d = whole_data[idx]
        w,h = d['width'], d['height']
        annotations = [x for x in self.data['annotations'] if x['image_id'] == d['id']]
        mask = annotation2binarymask(annotations,h,w)
        #print(mask.shape)
        #print(d['file_name'])
        img_path = self.image_dir + d['file_name'].replace('_','/', 2)
        #print(img_path)
        img = cv2.imread(img_path)
        #print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        augmented = self.aug_transform(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        img = self.norm_transform(image=img)["image"]
        if not self.test:
            return torch.FloatTensor(img), torch.FloatTensor(mask)
        else:
            return torch.FloatTensor(img), torch.FloatTensor(mask), d['file_name']        

    def __len__(self):
        return len(self.data['images'])