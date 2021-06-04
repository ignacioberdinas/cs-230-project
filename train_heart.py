import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from datasets.heartdatasets import VinBigDataHeartDataset
from utils.evaluate_utils import evaluate
import torchvision.transforms as T
from models.fastrcnn import get_model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def get_vinbigdata_dataframe(path):
    IMG_SIZE = 512
    df = pd.read_csv(path)
    df['x_min'] = IMG_SIZE * df['x_min'] / df['width']
    df['x_max'] = IMG_SIZE * df['x_max'] / df['width']
    df['y_min'] = IMG_SIZE * df['y_min'] / df['height']
    df['y_max'] = IMG_SIZE * df['y_max'] / df['height']
    df = df[df.class_name.eq('Cardiomegaly')]
    return df

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


from tqdm import tqdm


def train_one_epoch(model, optimizer, data_loader, device):
    model.train()

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    epoch_loss = 0
    for images, targets in tqdm(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        epoch_loss += losses.item()
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()
    return epoch_loss

from sklearn.model_selection import train_test_split

df = get_vinbigdata_dataframe('./data/vinbigdata/train.csv')
df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)

ds_train = VinBigDataHeartDataset(df_train, './data/vinbigdata/train/', get_transform(train=False))
ds_test = VinBigDataHeartDataset(df_test, './data/vinbigdata/train/', get_transform(train=False))

torch.manual_seed(1)
indices = torch.randperm(len(ds_train)).tolist()

# define training and validation data loaders
data_loader = DataLoader(ds_train, batch_size=2, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))
data_loader_test = DataLoader(ds_test, batch_size=1, shuffle=False, collate_fn=lambda batch: tuple(zip(*batch)))


from datasets.heartdatasets import CheXpertHeartDataset
import torch

ds_train_no_finding = CheXpertHeartDataset('./data/hand-label/nofinding/','../CheXpert-v1.0-small/train' ,get_transform(train=False))
ds_train_cardiomegaly = CheXpertHeartDataset('./data/hand-label/cardiomegaly-certain/','../CheXpert-v1.0-small/train' ,get_transform(train=False))
ds_train_cardiomegalyunc = CheXpertHeartDataset('./data/hand-label/cardiomegaly-uncertain/','../CheXpert-v1.0-small/train' ,get_transform(train=False))

full_ds_chexpert = torch.utils.data.ConcatDataset([ds_train_no_finding, ds_train_cardiomegaly,ds_train_cardiomegalyunc])
train_ds_chexpert,val_ds_chexpert = torch.utils.data.random_split(full_ds_chexpert, [len(full_ds_chexpert) - 100, 100], generator=torch.Generator().manual_seed(42))

data_loader_chex = DataLoader(train_ds_chexpert, batch_size=1, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))
data_loader_chex_test = DataLoader(val_ds_chexpert, batch_size=1, shuffle=False, collate_fn=lambda batch: tuple(zip(*batch)))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#Two classes, background/heart
num_classes = 2

model = get_model(num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.Adam(params, lr=0.005)
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 10

for epoch in range(num_epochs):
    print('Epoch[{}]'.format(epoch))
    epoch_loss = train_one_epoch(model, optimizer, data_loader, device)
    print('Epoch[{}]: {}'.format(epoch,epoch_loss))

    a,b = evaluate(model, data_loader_chex_test, device)
    # update the learning rate
    #lr_scheduler.step()
    torch.save(model.state_dict(), f'./intermediate/heart_weights/model3PRETRAINING{epoch}.pth')


optimizer2 = torch.optim.Adam(params, lr=0.001)

for epoch in range(10):
    print('Epoch[{}]'.format(epoch))
    epoch_loss = train_one_epoch(model, optimizer2, data_loader_chex, device)
    print('Epoch[{}]: {}'.format(epoch,epoch_loss))

    a,b = evaluate(model, data_loader_chex_test, device)

    torch.save(model.state_dict(), f'./intermediate/heart_weights/model3NOPRETRAINING{epoch}.pth')