
# %%
import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import glob
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.image as mpimg
import albumentations as A
from torch import nn
from model import *
from config import *
from dataload import *
from util import *
from sklearn.model_selection import train_test_split
import torch.optim.lr_scheduler as lr_scheduler
print(torch.cuda.is_available())
import shutil



outptfile1=outptfile
# print(outptfile1)

indir=r'E:\task\crack'
X_train=sorted(glob.glob(indir+"/JPEGImages/*.jpg"))
y_train=sorted(glob.glob(r"C:\Users\Admin\Desktop\lables/*.json"))


X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=0.3, random_state=42)

train_dataset = LoadData(X_train, y_train)
valid_dataset = LoadData(X_val, y_val,'val')


train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,)
valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,)

model=SegmentationModel()

if loadstate:
    model.load_state_dict(torch.load(rf'{loadstateptfile}'))
model.to(DEVICE)
optimizer=torch.optim.SGD(model.parameters(), lr=LR)

outloss={}
trainloss=outloss.setdefault('trainloss',[])
valloss=outloss.setdefault('valloss',[])

best_val_dice_loss=np.inf
best_val_loss=np.inf
for i in range(EPOCHS):
    outfile=basedir+rf"jpgoutnew/{str(i)}.jpg"
    # train_loss=1
    # valid_loss=1
    os.makedirs(os.path.split(outfile)[0],exist_ok=True)
    train_loss = train_fn(train_loader,model,optimizer)
    valid_loss = eval_fn(valid_loader,model,outfile)
    # train_dice,train_bce=train_loss
    # valid_dice,valid_bce=valid_loss
    trainloss.append(train_loss)
    valloss.append(valid_loss)
    print(f'Epochs:{i+1}\nTrain_loss --> {train_loss} \nValid_loss --> { valid_loss:} ')
    if valid_loss< best_val_loss:
        torch.save(model.state_dict(),outptfile1)
        print('Model Saved')
        # best_val_dice_loss=valid_dice 
        best_val_loss= valid_loss

import pandas as pd
outcsv=pd.DataFrame(outloss)
outcsv.to_csv(outptfile.replace('.pt','.csv'))




