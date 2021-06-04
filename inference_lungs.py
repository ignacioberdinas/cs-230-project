import os
import glob
import torch
from tqdm import tqdm
import cv2
import albumentations as A
import numpy as np
from datasets.lungdatasets import MEAN,STD
from models.unet import ResNetUNet

IMAGE_SIZE = 512

LUNG_MODEL_WEIGHTS = './intermediate/lung_mask_weights'
PATH = "./intermediate/out_lung_mask/"

base_path = 'C:/Users/ignacio/workspace/stanford/cs230/CheXpert-v1.0-small/train/'
CHEXPERT_VALIDATION_BASE = './data/chexpert-cardio-nofinding'

paths = os.listdir(CHEXPERT_VALIDATION_BASE)
inference_transforms = A.Compose([A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1.0)])

def load_image(base_path, path):
    path = path.replace('_','/',2)
    img_path = base_path + path
    image = cv2.imread(img_path,0)
    image = cv2.merge([image,image,image])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    augmented = inference_transforms(image=image)
    image = augmented['image']
    image = A.Normalize(mean=MEAN, std=STD)(image=image)["image"]
    return torch.FloatTensor(image).unsqueeze(0)

model = ResNetUNet().cuda()

best_weights = sorted(glob.glob(LUNG_MODEL_WEIGHTS + "/*"), key=lambda x: x[8:-5])[-1]

checkpoint = torch.load(best_weights)
model.load_state_dict(checkpoint['state_dict'])

print('Loaded model: ', best_weights.split("/")[1])
model.eval()
predictions = []

for p in tqdm(paths):
    img = load_image(base_path, p)
    data_batch = img.permute(0, 3, 1, 2).cuda()
    outputs = model(data_batch)

    out_cut = np.copy(outputs.data.cpu().numpy())
    out_cut[np.nonzero(out_cut < 0.5)] = 0.0
    out_cut[np.nonzero(out_cut >= 0.5)] = 1.0

    print(img.shape)
    print(img.squeeze().shape)
    print(out_cut[0].shape)
    cv2.imwrite(PATH + p, (out_cut[0].transpose(1, 2, 0) * 255).astype(np.uint8))