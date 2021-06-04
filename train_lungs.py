import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
import albumentations as A

from datasets.lungdatasets import SchenzenMontgomeryLungSegmentationDataset
from datasets.lungdatasets import CheXpertLungSegmentationDataset
from models.unet import ResNetUNet
from utils.utils import bce_dice_loss, dice_metric
import numpy as np
from tqdm import tqdm
import glob

BASE_DATA = './data/lung-segmentation/'
BASE_WEIGHTS = './intermediate/lung_mask_weights/'
IMAGE_SIZE = 512
BATCH_SIZE = 2
DEVICE = "cuda:0"

def get_transforms(size, test = False):
    #Do test-time augmentation?
    return A.Compose([
        A.Resize(height=size, width=size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.Transpose(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=45, p=0.3),
    ])

def load_dataset(base_path,train_valid_split_pct = 0.05):
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(base_path + 'raw_patients_info.csv')

    test_ids = [x.split('.')[0] for x in os.listdir(base_path + 'data/test/')]
    test_dataset = df[df['Id'].isin(test_ids)]
    not_test = df[~df['Id'].isin(test_ids)]

    train_dataset, validation_dataset = train_test_split(not_test, test_size=train_valid_split_pct)

    return train_dataset, validation_dataset, test_dataset


def do_epoch(model, optimizer, dataloader, loss_fn=bce_dice_loss, train = True):
    if train:
        model.train()
    else:
        model.eval()

    metrics = {
        'losses': [],
        'acc': []
    }

    for data, target in tqdm(dataloader):
        data = data.permute(0, 3, 1, 2).to(DEVICE)
        targets = target.unsqueeze(1).to(DEVICE)

        outputs = model(data)

        # Calculate accuracy
        out_cut = np.copy(outputs.data.cpu().numpy())
        out_cut[np.nonzero(out_cut < 0.5)] = 0.0
        out_cut[np.nonzero(out_cut >= 0.5)] = 1.0

        train_dice = dice_metric(out_cut, targets.data.cpu().numpy())
        metrics['acc'].append(train_dice)

        if train:
            #Calculate and save loss for tracking
            loss = loss_fn(outputs, targets)
            metrics['losses'].append(loss.item())

            #Do the optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return metrics


train, val, test = load_dataset(BASE_DATA, train_valid_split_pct=0.05)

train_ds = SchenzenMontgomeryLungSegmentationDataset(df=train, data_dir=BASE_DATA, aug_transform=get_transforms(IMAGE_SIZE))
val_ds = SchenzenMontgomeryLungSegmentationDataset(df=val, data_dir=BASE_DATA, aug_transform=get_transforms(IMAGE_SIZE))
test_ds = SchenzenMontgomeryLungSegmentationDataset(df=test, data_dir=BASE_DATA, aug_transform=get_transforms(IMAGE_SIZE), test=True)

ds_train_no_finding = CheXpertLungSegmentationDataset("./data/hand-label/cardiomegaly-certain.json", '../CheXpert-v1.0-small/train/', aug_transform=get_transforms(320))
ds_train_cardiomegaly = CheXpertLungSegmentationDataset("./data/hand-label/nofinding.json", '../CheXpert-v1.0-small/train/', aug_transform=get_transforms(320))

full_ds_chexpert = torch.utils.data.ConcatDataset([ds_train_no_finding, ds_train_cardiomegaly])


train_ds_chexpert,val_ds_chexpert = torch.utils.data.random_split(full_ds_chexpert, [len(full_ds_chexpert) - 100, 100], generator=torch.Generator().manual_seed(42))

print(len(train_ds_chexpert))
print(len(val_ds_chexpert))
train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=2, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=2, shuffle=False)

train_dl_chex = DataLoader(train_ds_chexpert, batch_size=2, shuffle=True)
val_dl_chex = DataLoader(val_ds_chexpert, batch_size=2, shuffle=True)

model = ResNetUNet().cuda()

for param in model.parameters():
    param.requires_grad = True

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.Adam(params, lr=0.00005)

loss_history = []
train_dice_history = []
val_dice_history = []

for epoch in range(5):

    print('Start epoch ', str(epoch))

    metrics_train = do_epoch(model, optimizer, train_dl, train=True)

    print('Validation epoch ', str(epoch))

    metrics_val = do_epoch(model, optimizer, val_dl_chex, train=False)

    print('Valdation dice loss', str(np.array(metrics_val['acc']).mean()))

    # train history
    loss_history.append(np.array(metrics_train['losses']).mean())
    train_dice_history.append(np.array(metrics_train['acc']).mean())
    val_dice_history.append(np.array(metrics_val['acc']).mean())

    # Save best model
    best_dice = max(val_dice_history)
    if val_dice_history[-1] >= best_dice:
        torch.save({'state_dict': model.state_dict()}, os.path.join(BASE_WEIGHTS, f"pretraining{val_dice_history[-1]:0.6f}_.pth"))

print('FINISH PRETRAINING')

optimizer2 = torch.optim.Adam(params, lr=0.000005)

for epoch in range(10):

    print('Start epoch ', str(epoch))

    metrics_train = do_epoch(model, optimizer2, train_dl_chex, train=True)

    print('Validation epoch ', str(epoch))

    metrics_val = do_epoch(model, optimizer2, val_dl_chex, train=False)

    print('Valdation dice loss', str(np.array(metrics_val['acc']).mean()))

    # train history
    loss_history.append(np.array(metrics_train['losses']).mean())
    train_dice_history.append(np.array(metrics_train['acc']).mean())
    val_dice_history.append(np.array(metrics_val['acc']).mean())

    # Save best model
    best_dice = max(val_dice_history)
    if val_dice_history[-1] >= best_dice:
        torch.save({'state_dict': model.state_dict()}, os.path.join(BASE_WEIGHTS, f"nopretraining{val_dice_history[-1]:0.6f}_.pth"))

best_weights = sorted(glob.glob(BASE_WEIGHTS + "/*"), key=lambda x: x[8:-5])[-1]
print(best_weights)