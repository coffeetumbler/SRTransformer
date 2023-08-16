import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize

import config

class batch_making:
    def __init__(self, items, batch=16):
        self.current = 0
        self.items = items
        self.stop = len(items)
        self.batch = batch

        concatenated = []
        concatenated_keys = []

        for key, value in self.items.items():
            concatenated.append(value)
            concatenated_keys.append(key)

        self.items = torch.stack(concatenated)
        self.keys = concatenated_keys

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.stop:
            _next = min(self.current + self.batch, self.stop)
            batched_degraded_img = self.items[self.current:_next]
            keys = self.keys[self.current:_next]
            self.current = _next
            return keys, batched_degraded_img
        else:
            self.current = 0
            raise StopIteration


            
class sr_dataset(Dataset):
    def __init__(self,
                 setting='train',
                 augmentation=True,
                 channel_wise_noise=0.5,
                 upscale=2,
                 dataset='DIV2K',
                 data_merge=False,
                 lr_img_size=64,
                 normalize_origin=True,
                 sliding_window=False):
        
        super(sr_dataset, self).__init__()
        self.upscale = upscale
        self.lr_img_size = lr_img_size
        self.img_size = upscale * lr_img_size
        
        self.dataset = dataset
        self.setting = setting
        self.is_train = setting == 'train'
        self.sliding_window = sliding_window
        
        self.augmentation = augmentation
        self.channel_wise_noise = channel_wise_noise
        self.rotation = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
        
        self.normalize_img = Normalize(config.IMG_NORM_MEAN, config.IMG_NORM_STD)
        self.normalize_origin = normalize_origin
        self.data_merge = data_merge
        
        self.intersection = config.PIXEL_INTERSECTION
        self.dataset_path = config.DATASET_PATH[self.setting]

        self.color_shifting_range = config.COLOR_SHIFTING_RANGE
    
        if self.is_train:
            if self.data_merge:
                data_HR = []
                data_degraded = []
                for i in config.TRAINING_DATA_LIST:
                    df = pd.read_csv(config.DATA_LIST_DIR + i + '_train_HR.csv').sort_values(by='name')
                    data_HR.append(df)
                    df_degraded = pd.read_csv(config.DATA_LIST_DIR + i + '_train_LR_bicubic_X{}.csv'.format(upscale)).sort_values(by='name')
                    data_degraded.append(df_degraded)
                self.data_name_list_HR = pd.concat(data_HR, axis=0, ignore_index=True)
                self.data_name_list_degraded = pd.concat(data_degraded, axis=0, ignore_index=True)
            elif dataset == 'DF2K':
                data_name = os.listdir(self.dataset_path + 'DF2K_train_HR')
                self.data_name_list_HR = pd.DataFrame({'name':data_name})
                self.data_name_list_HR['data'] = 'DF2K'
                self.data_name_list_degraded = self.data_name_list_HR
            else:
                self.data_name_list_HR = pd.read_csv(config.DATA_LIST_DIR + self.dataset + '_train_HR.csv').sort_values(by='name')
                self.data_name_list_degraded = pd.read_csv(config.DATA_LIST_DIR + self.dataset + '_train_LR_bicubic_X{}.csv'.format(upscale)).sort_values(by='name')

        else:
            self.data_name_list_HR = pd.read_csv(config.DATA_LIST_DIR + self.dataset + '_valid_ipt_HR.csv').sort_values(by='name')
            self.data_name_list_degraded = pd.read_csv(config.DATA_LIST_DIR + self.dataset + '_valid_ipt_LR_bicubic_X{}.csv'.format(upscale)).sort_values(by='name')


    def __len__(self):
        return len(self.data_name_list_HR)
    

    def __getitem__(self, idx):
        if self.is_train:
            view_path_HR = self.dataset_path + self.data_name_list_HR.iloc[idx]['data'] + '_train_HR/' + self.data_name_list_HR.iloc[idx]['name']
            view_path_degraded = self.dataset_path + self.data_name_list_degraded.iloc[idx]['data'] + '_train_LR_bicubic/X{}/'.format(self.upscale)\
                                 + self.data_name_list_degraded.iloc[idx]['name']

            HR_image = cv2.imread(view_path_HR, cv2.IMREAD_COLOR).copy()
            degraded_image = cv2.imread(view_path_degraded, cv2.IMREAD_COLOR).copy()

            # multiple image augmentation
            p, q = degraded_image.shape[:2]
            p -= self.lr_img_size
            q -= self.lr_img_size

            p = np.random.randint(0, p)
            q = np.random.randint(0, q)

            indices = np.array([p, p+self.lr_img_size, q, q+self.lr_img_size], dtype=int)
            hr_indices = indices * self.upscale
            HR_image = HR_image[hr_indices[0]:hr_indices[1], hr_indices[2]:hr_indices[3]]
            degraded_image = degraded_image[indices[0]:indices[1], indices[2]:indices[3]]
            
            # Augmentation probabilities
            probs = np.random.rand(3)

            # flip in 1/2 probability
            if probs[0] < 0.5:
                HR_image = cv2.flip(HR_image, 1)
                degraded_image = cv2.flip(degraded_image, 1)

            # rotation
            if probs[1] < 0.75:
                choice = np.random.choice(self.rotation)
                HR_image = cv2.rotate(HR_image, choice)
                degraded_image = cv2.rotate(degraded_image, choice)

            # color shifting
            HR_image = HR_image.astype(int)
            degraded_image = degraded_image.astype(int)
            if probs[2] < self.channel_wise_noise:
                pn = np.random.randint(-self.color_shifting_range, self.color_shifting_range, size=(3,))
                HR_image = np.minimum(255, np.maximum(0, HR_image + pn[np.newaxis, np.newaxis, :]))
                degraded_image = np.minimum(255, np.maximum(0, degraded_image + pn[np.newaxis, np.newaxis, :]))

            items = {}
            if self.normalize_origin:
                items['origin'] = self.normalize_img(torch.from_numpy(HR_image.transpose(2,0,1)).float() / 255)
            else:
                items['origin'] = torch.from_numpy(HR_image.transpose(2,0,1)).float() / 255
            items['degraded'] = self.normalize_img(torch.from_numpy(degraded_image.transpose(2,0,1)).float() / 255)
            
            return items
        
        else:
            view_path_HR = self.dataset_path + self.data_name_list_HR.iloc[idx]['data'] + '/HR/' + self.data_name_list_HR.iloc[idx]['name']
            view_path_degraded = self.dataset_path + self.data_name_list_degraded.iloc[idx]['data'] + '/LR_bicubic/X{}/'.format(self.upscale)\
                                 + self.data_name_list_degraded.iloc[idx]['name']
            
            if self.sliding_window:
                origin = cv2.imread(view_path_HR, cv2.IMREAD_COLOR).copy()
                degraded = cv2.imread(view_path_degraded, cv2.IMREAD_COLOR).copy()
                origin = origin[0:degraded.shape[0]*self.upscale, 0:degraded.shape[1]*self.upscale]

                if origin.shape[0] >= self.img_size:
                    idx_x = np.arange(0, degraded.shape[0]-self.lr_img_size+1, self.lr_img_size-self.intersection)
                    if idx_x[-1] != degraded.shape[0] - self.lr_img_size:
                        idx_x = np.append(idx_x, degraded.shape[0]-self.lr_img_size)
                else:
                    idx_x = np.zeros(1)

                if origin.shape[1] >= self.img_size:
                    idx_y = np.arange(0, degraded.shape[1]-self.lr_img_size+1, self.lr_img_size-self.intersection)
                    if idx_y[-1] != degraded.shape[1] - self.lr_img_size:
                        idx_y = np.append(idx_y, degraded.shape[1]-self.lr_img_size)
                else:
                    idx_y = np.zeros(1)

                item_degraded = {}
                mask = torch.zeros(origin.shape[0], origin.shape[1])

                for i, _i in zip(idx_x, idx_x*self.upscale):
                    for j, _j in zip(idx_y, idx_y*self.upscale):
                        item_degraded[(_i,_j)] = self.normalize_img(torch.from_numpy(degraded[i:i+self.lr_img_size,
                                                                                              j:j+self.lr_img_size].transpose(2,0,1)).float() / 255)
                        mask[_i:_i+self.img_size, _j:_j+self.img_size] += 1

                origin = torch.from_numpy(origin.transpose(2,0,1)).float() / 255
                items = {"origin" : origin, "degraded" : item_degraded, "mask" : mask}
                return items
            
            else:
                HR_image = cv2.imread(view_path_HR, cv2.IMREAD_COLOR).copy()
                degraded_image = cv2.imread(view_path_degraded, cv2.IMREAD_COLOR).copy()
                HR_image = HR_image[0:degraded_image.shape[0]*self.upscale, 0:degraded_image.shape[1]*self.upscale]

                HR_image = torch.from_numpy(HR_image.transpose(2,0,1)).float() / 255
                degraded_image = self.normalize_img(torch.from_numpy(degraded_image.transpose(2,0,1)).float() / 255)
                items = {"origin" : HR_image, "degraded" : degraded_image}
                return items
        
        
        
def get_dataloader(batch_size=16, setting='train', pin_memory=False, num_workers=0, drop_last=False, **kwargs):
    if setting == 'train':    
        augmentation = True
    else:
        augmentation = False
        batch_size = 1
    dataloader = sr_dataset(setting=setting, augmentation=augmentation, **kwargs)
    return DataLoader(dataloader, batch_size=batch_size, shuffle=augmentation, pin_memory=pin_memory,
                      drop_last=drop_last, num_workers=num_workers)
