"""
Lightning Data Module class
"""
from typing import Optional, Tuple

import logging

import pytorch_lightning as pl
import torch

import glob 
import json

from collections import Counter
from pathlib import Path

from baseline.utils import load_obj
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms

import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data.distributed import Sampler, Dataset
import torch.distributed as dist

import torchvision.transforms.functional as TF
import random

T_co = TypeVar('T_co', covariant=True)

import os
import json
import pandas as pd
import numpy as np
from PIL import Image
import math

from omegaconf import DictConfig, OmegaConf

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MEANS = {
    "ImageFolder": (0.485, 0.456, 0.406), # 
}
STDS = {
    "ImageFolder": (0.229, 0.224, 0.225), # 
}

class dataset_OVR(Dataset):
    def __init__(self, folder, split, transform):
        '''
        dataset_type = either AB or SD task
        key: base case or trivial case
        split: train, val and test
        '''
        super().__init__()
        self.root = '/cifs/data/tserre_lrs/projects/prj_visreason/dcsr/for_GAMR/EXP_OVR_CONFIG'

        self.folder = folder
        self.split = split
        self.preprocess = transform
        
        # obtain folder_split as train test or validation
        for directory in os.listdir(self.folder):
            # test-SetB  train-SetA  validation-SetC
            if split.casefold() in directory.casefold():
                self.folder_split = os.path.join(self.folder, directory)

        # list all the files in that split
        self.data = []
        self.target = []
        excluded_list = ['TAR_IMG', 'OrgCanvas']
        file_list = [a for a in os.listdir(self.folder_split) if a not in excluded_list]

        # reference files:
        for class_path in file_list:
            if class_path == 'ABOVE' or class_path == 'SAME':
                class_label = 0
            else:
                class_label = 1
            class_name = os.path.join(self.folder_split, class_path, 'FULL_IMG')
            for img_path in glob.glob(class_name + "/*.png"):
                self.data.append(img_path)
                self.target.append(class_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # read image
        data_orig = self.data[idx]
        if 'Base' in data_orig:
            data_ref = self.data[idx].replace('FULL_IMG', 'REF_IMG').replace('_FullStim', '_objSA-Ref')
            data_tar = self.data[idx].replace('FULL_IMG', 'TAR_IMG').replace('_FullStim', '_objSA-Tar')
        elif 'ObjLOC' in data_orig:
            data_ref = self.data[idx].replace('FULL_IMG', 'REF_IMG').replace('_FullStim', '_objLOC-Ref')
            data_tar = self.data[idx].replace('FULL_IMG', 'TAR_IMG').replace('_FullStim', '_objLOC-Tar')
        else:
            data_ref = self.data[idx].replace('FULL_IMG', 'REF_IMG').replace('_FullStim', '_objID-Ref')
            data_tar = self.data[idx].replace('FULL_IMG', 'TAR_IMG').replace('_FullStim', '_objID-Tar')
        
        # import pdb; pdb.set_trace()
        data_orig = self.preprocess(Image.open(data_orig)) #.convert("RGB"))
        data_ref = self.preprocess(Image.open(data_ref)) #.convert("RGB"))
        data_tar = self.preprocess(Image.open(data_tar)) #.convert("RGB"))

        # data = torch.stack([data_orig, data_ref, data_tar], 0)
        data = torch.cat([data_orig, data_ref, data_tar], 0)
        
        target = torch.tensor(self.target[idx], dtype=torch.long)
        
        return data, target

class DataModule_OVR(pl.LightningDataModule):
    def __init__(self,
                batch_size: int = 400,
                num_workers: int = 2,
                data_type: str = 'AB',
                key: str = 'Base',
                pin_memory: bool = True,):
        '''
        data_type = either AB or SD task
        key: base case or trivial case
        '''
        super().__init__()

        means, stds = MEANS['ImageFolder'], STDS['ImageFolder']
        logger.debug(f"hard coded means: {means}, stds: {stds}")

        self.key = key
        self.root = '/cifs/data/tserre_lrs/projects/prj_visreason/dcsr/for_GAMR/EXP_OVR_CONFIG'
        self.dataset_type = [a for a in os.listdir(self.root) if data_type.casefold() in a.casefold()]
        self.dataset_path = os.path.join(self.root, self.dataset_type[0])
        
        # obtain folder as basecase or trivial case
        for directory in os.listdir(self.dataset_path):
            if self.key.casefold() in directory.casefold():
                # BaseCase_AB-ObjSA-ND1-OrgCanvas, NonTrivialCase_AB-ObjID-ND1-OrgCanvas
                folder = os.path.join(self.dataset_path, directory)

        self.batch_size = batch_size
        self.num_workers =  num_workers  
        self.pin_memory = pin_memory

        self.transforms = transforms.Compose(
            [
                # transforms.Resize((224,224)),
                transforms.ToTensor(),
                # transforms.Normalize(means, stds)
            ]
        )

        self.train_data = dataset_OVR(folder = folder, \
            split = 'train', transform=self.transforms)

        self.val_data = dataset_OVR(folder = folder, \
            split = 'validation', transform=self.transforms)

        self.test_data = dataset_OVR(folder = folder, \
            split = 'test', transform=self.transforms)


    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size,
                            num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,
                            num_workers=self.num_workers, pin_memory=self.pin_memory)
    
    def test_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,
                            num_workers=self.num_workers, pin_memory=self.pin_memory)