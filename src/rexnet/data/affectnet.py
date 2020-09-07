from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
import cv2

import os

class AffectNetDataset(Dataset):
    def __init__(self,
                 path: str,
                 transform: transforms,
                 phase: str='Train'):

        self.phase = phase

        self.root_path = path
        self.image_path = os.path.join(self.root_path, 'Manually_Annotated_Images')

        self.calm = [0]
        self.bright = [1]
        self.surprise = [3]
        self.dark = [2, 4, 5, 6, 7]
        self.pass_list = [8, 9, 10]

        self.transform = transform

        self.img_list, self.anno_list = self.load()

    def load(self):
        img_list = []
        anno_list = []

        if self.phase == 'train':
            csv_file = os.path.join(self.root_path, 'training.csv')
            data = pd.read_csv(csv_file,
                               names=['dir', 'face_x', 'face_y', 'face_w', 'face_h', 'face_land', 'exp', 'valence', 'arousal'],
                              skiprows=[0])

        elif self.phase == 'validation':
            csv_file = os.path.join(self.root_path, 'validation.csv')
            data = pd.read_csv(csv_file,
                               names=['dir', 'face_x', 'face_y', 'face_w', 'face_h', 'face_land', 'exp', 'valence', 'arousal'])

        for idx, d in data.iterrows():
            if int(d['exp']) in self.calm:
                anno_list.append(0)
                img_list.append(d['dir'])
            elif int(d['exp']) in self.bright:
                anno_list.append(1)
                img_list.append(d['dir'])
            elif int(d['exp']) in self.surprise:
                anno_list.append(2)
                img_list.append(d['dir'])
            elif int(d['exp']) in self.dark:
                anno_list.append(3)
                img_list.append(d['dir'])
            else:
                continue
            
        return img_list, anno_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.image_path, self.img_list[idx]))
        anno = self.anno_list[idx]

        if self.transform:
            img = self.transform(img)

        return img, anno