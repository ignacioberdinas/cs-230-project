import torch.nn as nn
import torch

def draw_bounding_box(tensor,bbox,nrow=8,padding=2,normalize=False,range=None,scale_each=False,pad_value=0,fill=None,):
    from torchvision.utils import make_grid
    from PIL import Image, ImageDraw
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(
        1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)

    ImageDraw.Draw(im).rectangle(bbox, fill=fill, outline=(255, 0, 0), width=3)
    from numpy import array as to_numpy_array
    return torch.from_numpy(to_numpy_array(im))

import numpy as np
import cv2


def decodeSeg(mask, segmentations):
    """
    Draw segmentation
    """
    pts = [
        np
            .array(anno)
            .reshape(-1, 2)
            .round()
            .astype(int)
        for anno in segmentations
    ]
    mask = cv2.fillPoly(mask, pts, 1)

    return mask

def annotation2binarymask(annotations,h,w):
    mask = np.zeros((h, w), np.uint8)
    for annotation in annotations:
        segmentations = annotation['segmentation']
        if isinstance(segmentations, list):
            mask = decodeSeg(mask, segmentations)
    return mask

def jaccard_metric(inputs, target, eps=1e-7):
    intersection = (target * inputs).sum()
    union = (target.sum() + inputs.sum()) - intersection + eps

    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return (intersection + eps) / union

def dice_metric(inputs, target):
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return intersection / union

def dice_loss(inputs, target):
    smooth = 1.0
    intersection = 2.0 * ((target * inputs).sum()) + smooth
    union = target.sum() + inputs.sum() + smooth

    return 1 - (intersection / union)

def bce_dice_loss(inputs, target, bce_weight=0.5):
    bceloss = nn.BCELoss()
    return bceloss(inputs, target) * bce_weight + dice_loss(inputs, target) * (1-bce_weight)