import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd


class VinBigDataHeartDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(f'{self.image_dir}{image_id}.png', cv2.IMREAD_COLOR)

        boxes = torch.as_tensor(records[['x_min', 'y_min', 'x_max', 'y_max']].values, dtype=torch.float32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        labels = torch.zeros((len(records.class_id.values),), dtype=torch.int64) + 1

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return self.image_ids.shape[0]


class CheXpertHeartDataset(Dataset):
    def __init__(self, data_dir, image_dir, transforms=None, test=False):
        super().__init__()

        self.labels_paths = os.listdir(data_dir)
        self.paths = [s.replace('_', '/', 2) for s in os.listdir(data_dir)]
        self.paths = [s.replace('.txt', '.jpg') for s in self.paths]
        self.data_dir = data_dir
        self.image_dir = image_dir
        self.transforms = transforms
        self.test = test

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        label_path = self.labels_paths[idx]
        #print(f'{self.image_dir}/{image_path}')
        image = cv2.imread(f'{self.image_dir}/{image_path}', 0)
        image = cv2.merge([image, image, image])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        labels = pd.read_csv(self.data_dir + '/' + label_path, header=None, delimiter=" ",
                             names=['clazz', 'x_min', 'y_min', 'x_max', 'y_max'])

        labels = labels[labels.clazz.eq(0)]

        height, width, _ = image.shape

        target = {}

        if len(labels[['x_min', 'y_min', 'x_max', 'y_max']].values) > 0:
            x, y, w, h = labels[['x_min', 'y_min', 'x_max', 'y_max']].values[0]

            boxes = torch.as_tensor(
                np.array([[(x - (w / 2)) * width, (y - (h / 2)) * height, (x + (w / 2)) * width, (y + (h / 2)) * height]]),
                dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            area = torch.as_tensor(area, dtype=torch.float32)
            labels = torch.zeros((len(labels),), dtype=torch.int64) + 1
        else:
            print(image_path)
            return self.__getitem__(idx+1)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((1,), dtype=torch.int64)
        
        target['boxes'] = boxes
        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = torch.tensor([idx])
        if self.test:
            target['extra'] = image_path
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.paths)