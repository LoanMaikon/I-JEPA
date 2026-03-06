<div align="center">

# I-JEPA

I-JEPA implementation. Some code is adapted from [I-JEPA official code](https://github.com/facebookresearch/ijepa).

</div>

## 1. Overview
This repository contains:
- `train.py`: Pretraining of the encoder.
- `src/`: Source code of the project.
- `configs/`: Configurations for the pretraining.


## 2. Train encoder

In `configs/config.yaml` there is an example of pretraining of the encoder:

<pre>
data:
  batch_size: 32 # Batch size per GPU
  crop_scale: # Scale of the random resized crop
  - 0.3
  - 1.0
  crop_size: 224 # Size of the random resized crop
  dataset_folder_path: ../imagenet/ # Path to the folder where the imagenet folder is
  num_workers: 4 # Number of workers for the dataloader
  pin_mem: true # Whether to pin memory for the dataloader
  drop_last: true # Whether to drop the last batch if it's smaller than the batch size

mask:
  target_aspect_ratio: # Aspect ratio of the target mask
  - 0.75
  - 1.5
  context_mask_scale: # Scale of the context mask
  - 0.85
  - 1.0
  min_context_patches: 10 # Minimum number of patches in the context mask
  num_target_masks: 4 # Number of target masks per image
  patch_size: 14 # Size of each patch
  target_mask_scale: # Scale of the target mask
  - 0.15
  - 0.2

meta:
  model_name: vit_tiny # Name of the model. It can be vit_tiny, vit_small, vit_base, vit_large, vit_huge or vit_giant
  predictor_depth: 6 # Number of layers in the predictor
  predictor_emb_dim: 384 # Embedding dimension of the predictor
  predictor_num_heads: 3 # Number of heads in the predictor
  checkpoint: false # Whether to use checkpointing for the model. It trades memory efficiency for time efficiency

optimization:
  ipe_scale: 1.0 # Scale of IPE for schedulers
  ema: # Exponential Moving Average
  - 0.996
  - 1.0
  lr: # Learning rate
  - 0.0001
  - 0.001
  - 0.000001
  wd: # Weight decay
  - 0.04
  - 0.4
  epochs: 300 # Number of epochs for training
  warmup_epochs: 15 # Number of warmup epochs
