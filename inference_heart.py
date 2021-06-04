import cv2
import torch
import os
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from models.fastrcnn import get_model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def evaluate_image(model, path,device):
    img_loaded = cv2.imread(path, cv2.IMREAD_COLOR)
    img_loaded = get_transform(train=False)(img_loaded)
    model.eval()
    with torch.no_grad():
        prediction = model([img_loaded.to(device)])
        return img_loaded,prediction[0]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2

# get the model using our helper function
model = get_model(num_classes)
# move model to the right device
model.to(device)

model_names = [name.split('.')[0] for name in os.listdir('./intermediate/heart_weights/')]

print(model_names)
for name in ['modelPRETRAINING9']:
    model_params = torch.load(f'./intermediate/heart_weights/{name}.pth')
    model.load_state_dict(model_params)

    import os
    from tqdm import tqdm
    import pickle

    CHEXPERT_VALIDATION_BASE = './data/chexpert-cardio-nofinding'

    paths = os.listdir(CHEXPERT_VALIDATION_BASE)
    predictions = []
    for p in tqdm(paths):
        prediction = evaluate_image(model, CHEXPERT_VALIDATION_BASE+'/'+p,device)
        predictions.append(prediction)

    with open(f'./intermediate/out_paths{name}.pickle', 'wb') as handle:
        pickle.dump(paths, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'./intermediate/out_heart{name}.pickle', 'wb') as handle:
        pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)