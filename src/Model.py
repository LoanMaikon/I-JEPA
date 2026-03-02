from torchvision.transforms import v2
import torch
import yaml
import torch.nn as nn
import torch.optim as optim
import shutil
import random
import matplotlib.pyplot as plt
import numpy as np
import os
from time import strftime, localtime
import json

from .imagenet_dataset import ImageNetDataset
from .custom_vit import CustomViT

class Model():
    def __init__(self,
                 operation, # train or test
                 config_path,
                 device,
                 output_path,
                 ):
        
        self.operation = operation
        self.config_path = config_path
        self.device = device
        self.output_path = output_path

        self._create_output_folder()
        self._load_config()

    def _create_output_folder(self):
        os.makedirs(self.output_path)
        shutil.copy(self.config_path, os.path.join(self.output_path, "config.yaml"))

    def _load_config(self):
        self.config = yaml.safe_load(open(self.config_path, 'r'))

        self.data_batch_size = int(self.config['data']['batch_size'])
        self.data_crop_scale = tuple(self.config['data']['crop_scale'])
        self.data_crop_size = int(self.config['data']['crop_size'])
        self.data_dataset_folder_path = str(self.config['data']['dataset_folder_path'])
        self.data_num_workers = int(self.config['data']['num_workers'])
        self.data_pin_mem = bool(self.config['data']['pin_mem'])

        self.mask_target_aspect_ratio = tuple(self.config['mask']['target_aspect_ratio'])
        self.mask_context_mask_scale = tuple(self.config['mask']['context_mask_scale'])
        self.mask_min_context_patches = int(self.config['mask']['min_context_patches'])
        self.mask_num_target_masks = int(self.config['mask']['num_target_masks'])
        self.mask_patch_size = int(self.config['mask']['patch_size'])
        self.mask_target_mask_scale = tuple(self.config['mask']['target_mask_scale'])

        self.meta_model_name = str(self.config['meta']['model_name'])
        self.meta_predictor_depth = int(self.config['meta']['predictor_depth'])
        self.meta_predictor_emb_dim = int(self.config['meta']['predictor_emb_dim'])

        self.optimization_ema = tuple(self.config['optimization']['ema'])
        self.optimization_lr = tuple(self.config['optimization']['lr'])
        self.optimization_start_wd = tuple(self.config['optimization']['start_wd'])
        self.optimization_epochs = int(self.config['optimization']['epochs'])
        self.optimization_warmup_epochs = int(self.config['optimization']['warmup_epochs'])

        self.data_dataset_folder_path += "/" if not self.data_dataset_folder_path.endswith("/") else ""

    def write_on_log(self, text):
        time = strftime("%Y-%m-%d %H:%M:%S - ", localtime())
        mode = "w" if not os.path.exists(os.path.join(self.output_path, "log.txt")) else "a"
        with open(os.path.join(self.output_path, "log.txt"), mode) as file:
            file.write(time + text + "\n")