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
import copy
import torch.nn.functional as F

from .imagenet_dataset import ImageNetDataset
from .models import vit_predictor, vit_tiny, vit_small, vit_base, vit_large, vit_huge, vit_giant
from .mask_collator import MaskCollator
from .schedulers import WarmupCosineSchedule, CosineWDSchedule

class Model():
    def __init__(self,
                 operation,
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
        self._load_device()

        self._load_transform()
        self._load_dataloader()
        self._load_model()
        self._load_optimizer()
        self._load_schedulers()
        self._load_criterion()
        self._load_momentum_schedule()

    def get_optimizer(self):
        return self.optimizer

    def get_model(self):
        return self.model
    
    def get_predictor(self):
        return self.predictor

    def get_target_model(self):
        return self.target_model

    def get_dataloader(self):
        return self.dataloader

    def get_num_epochs(self):
        return self.optimization_epochs

    def _load_optimizer(self):
        # Biases and LayerNorm weights should not be decayed
        param_groups = [
            {
                'params': (p for n, p in self.model.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1)),
                'weight_decay': self.optimization_wd[0],
            }, 
            {
                'params': (p for n, p in self.predictor.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1)),
                'weight_decay': self.optimization_wd[0],
            },
            {
                'params': (p for n, p in self.model.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0,
            },
            {
                'params': (p for n, p in self.predictor.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0,
            }
        ]
        
        self.optimizer = optim.AdamW(param_groups, lr=self.optimization_lr[0])
    
    def _load_schedulers(self):
        self.lr_scheduler = WarmupCosineSchedule(
            optimizer=self.optimizer,
            warmup_steps=self.optimization_warmup_epochs * len(self.dataloader),
            start_lr=self.optimization_lr[0],
            middle_lr=self.optimization_lr[1],
            final_lr=self.optimization_lr[2],
            T_max=int(self.optimization_ipe_scale * self.optimization_epochs * len(self.dataloader))
        )

        self.wd_scheduler = CosineWDSchedule(
            optimizer=self.optimizer,
            start_wd=self.optimization_wd[0],
            final_wd=self.optimization_wd[1],
            T_max=int(self.optimization_ipe_scale * self.optimization_epochs * len(self.dataloader))
        )
    
    def step_schedulers(self):
        self.lr_scheduler.step()
        self.wd_scheduler.step()
    
    def print_schedulers(self):
        lr = self.optimizer.param_groups[0].get('lr', 0.0)

        wd = 0.0
        for group in self.optimizer.param_groups:
            if not group.get('WD_exclude', False):
                wd = group.get('weight_decay', 0.0)
                break

        self.write_on_log(f"LR: {lr:.12f}, WD: {wd:.12f}")

    def _load_criterion(self):
        self.criterion = nn.MSELoss()
    
    def apply_criterion(self, pred, target):
        return self.criterion(pred, target)

    def _load_transform(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transform = v2.Compose([
            v2.Resize((self.data_crop_size, self.data_crop_size)),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=mean, std=std)
        ])

    def _load_dataloader(self):
        mask_collator = MaskCollator(
            crop_size=self.data_crop_size,
            patch_size=self.mask_patch_size,
            n_targets=self.mask_num_target_masks,
            min_keep=self.mask_min_context_patches,
            context_mask_scale=self.mask_context_mask_scale,
            pred_aspect_ratio=self.mask_target_aspect_ratio,
            pred_mask_scale=self.mask_target_mask_scale
        )

        dataset = ImageNetDataset(operation=self.operation, dataset_folder_path=self.data_dataset_folder_path, transform=self.transform)
        self.dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.data_batch_size,
            shuffle=True,
            num_workers=self.data_num_workers,
            pin_memory=self.data_pin_mem,
            collate_fn=mask_collator,
            drop_last=self.data_drop_last,
        )
    
    def _load_model(self):
        match self.meta_model_name:
            case "vit_tiny":
                self.model = vit_tiny(patch_size=self.mask_patch_size)
            case "vit_small":
                self.model = vit_small(patch_size=self.mask_patch_size)
            case "vit_base":
                self.model = vit_base(patch_size=self.mask_patch_size)
            case "vit_large":
                self.model = vit_large(patch_size=self.mask_patch_size)
            case "vit_huge":
                self.model = vit_huge(patch_size=self.mask_patch_size)
            case "vit_giant":
                self.model = vit_giant(patch_size=self.mask_patch_size)


        self.predictor = vit_predictor(num_patches=self.model.get_num_patches(),
                                       embed_dim=self.model.get_embed_dim(),
                                       depth=self.meta_predictor_depth,
                                       predictor_embed_dim=self.meta_predictor_emb_dim,
                                       num_heads=self.meta_predictor_num_heads,                     
        )

        self.target_model = copy.deepcopy(self.model)

        self._unfreeze_model(self.model)
        self._unfreeze_model(self.predictor)
        self._freeze_model(self.target_model)

        self.model.to(self.device)
        self.predictor.to(self.device)
        self.target_model.to(self.device)

        self.model.train()
        self.predictor.train()
        self.target_model.train()

    def save_models(self):
        os.makedirs(os.path.join(self.output_path, "models"), exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.output_path, "models", "model.pth"))
        torch.save(self.predictor.state_dict(), os.path.join(self.output_path, "models", "predictor.pth"))
        torch.save(self.target_model.state_dict(), os.path.join(self.output_path, "models", "target_model.pth"))
    
    def _load_momentum_schedule(self):
        self.momentum_scheduler = (self.optimization_ema[0] + i * (self.optimization_ema[1] - self.optimization_ema[0]) / (self.optimization_epochs * len(self.dataloader) * self.optimization_ipe_scale)
                          for i in range(int(len(self.dataloader) * self.optimization_epochs * self.optimization_ipe_scale)))
    
    def step_momentum_schedule(self):
        return next(self.momentum_scheduler)

    def update_target_model(self, print=False):
        momentum = self.step_momentum_schedule()

        for param_q, param_k in zip(self.model.parameters(), self.target_model.parameters()):
            param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data

        self.write_on_log(f"Updated target model with momentum: {momentum:.12f}") if print else None
    
    def _unfreeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = True

    def _freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
        
    def _create_output_folder(self):
        os.makedirs(self.output_path)
        shutil.copy(self.config_path, os.path.join(self.output_path, "config.yaml"))
    
    def _load_device(self):
        self.device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")

    def _load_config(self):
        self.config = yaml.safe_load(open(self.config_path, 'r'))

        self.data_batch_size = int(self.config['data']['batch_size'])
        self.data_crop_scale = tuple(self.config['data']['crop_scale'])
        self.data_crop_size = int(self.config['data']['crop_size'])
        self.data_dataset_folder_path = str(self.config['data']['dataset_folder_path'])
        self.data_num_workers = int(self.config['data']['num_workers'])
        self.data_pin_mem = bool(self.config['data']['pin_mem'])
        self.data_drop_last = bool(self.config['data']['drop_last'])

        self.mask_target_aspect_ratio = tuple(self.config['mask']['target_aspect_ratio'])
        self.mask_context_mask_scale = tuple(self.config['mask']['context_mask_scale'])
        self.mask_min_context_patches = int(self.config['mask']['min_context_patches'])
        self.mask_num_target_masks = int(self.config['mask']['num_target_masks'])
        self.mask_patch_size = int(self.config['mask']['patch_size'])
        self.mask_target_mask_scale = tuple(self.config['mask']['target_mask_scale'])

        self.meta_model_name = str(self.config['meta']['model_name'])
        self.meta_predictor_depth = int(self.config['meta']['predictor_depth'])
        self.meta_predictor_emb_dim = int(self.config['meta']['predictor_emb_dim'])
        self.meta_predictor_num_heads = int(self.config['meta']['predictor_num_heads'])

        self.optimization_ipe_scale = float(self.config['optimization']['ipe_scale'])
        self.optimization_ema = tuple(self.config['optimization']['ema'])
        self.optimization_lr = tuple(self.config['optimization']['lr'])
        self.optimization_wd = tuple(self.config['optimization']['wd'])
        self.optimization_epochs = int(self.config['optimization']['epochs'])
        self.optimization_warmup_epochs = int(self.config['optimization']['warmup_epochs'])

        self.data_dataset_folder_path += "/" if not self.data_dataset_folder_path.endswith("/") else ""

    def write_on_log(self, text):
        time = strftime("%Y-%m-%d %H:%M:%S - ", localtime())
        mode = "w" if not os.path.exists(os.path.join(self.output_path, "log.txt")) else "a"
        with open(os.path.join(self.output_path, "log.txt"), mode) as file:
            file.write(time + text + "\n")
