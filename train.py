import argparse
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.Model import Model
from src.models import apply_masks, repeat_interleave_batch

def main():
    args = get_args()
    local_rank, world_size = setup_distributed()

    device_index = local_rank if is_distributed() else args.devices[0]

    model = Model(
        operation="train",
        config_path=args.config,
        device_index=device_index,
        output_path=args.output_path,
        distributed=is_distributed(),
        world_size=world_size,
        rank=int(os.environ.get("RANK", "0")),
    )

    try:
        train(model, distributed=is_distributed())
    finally:
        cleanup_distributed()

def train(model, distributed=False):
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(1, model.get_num_epochs() + 1):
        if distributed and hasattr(model.get_dataloader().sampler, "set_epoch"):
            model.get_dataloader().sampler.set_epoch(epoch)

        if model.is_main_process():
            model.write_on_log(f"Epoch {epoch}/{model.get_num_epochs()}")

        epoch_loss = 0.0
        epoch_samples = 0

        for (images, labels), masks_context, masks_pred in model.get_dataloader():
            model.get_optimizer().zero_grad(set_to_none=True)

            imgs = images.to(model.device, non_blocking=True)
            masks_context = [m.to(model.device, non_blocking=True) for m in masks_context]
            masks_pred = [m.to(model.device, non_blocking=True) for m in masks_pred]

            with torch.amp.autocast("cuda", dtype=torch.float16):
                z = model.get_model()(imgs, masks_context)
                z_pred = model.get_predictor()(z, masks_context, masks_pred)

                with torch.no_grad():
                    z_target = model.get_target_model()(imgs)
                    z_target = F.layer_norm(z_target, (z_target.size(-1),))
                    B = len(z_target)
                    z_target = apply_masks(z_target, masks_pred)
                    z_target = repeat_interleave_batch(z_target, B, repeat=len(masks_context))

                loss = model.apply_criterion(z_pred, z_target)

            scaler.scale(loss).backward()
            scaler.step(model.get_optimizer())
            scaler.update()

            model.step_schedulers()
            # model.print_schedulers()
            model.update_target_model(print=False)

            epoch_loss += loss.item() * images.size(0)
            epoch_samples += images.size(0)

        if distributed:
            loss_tensor = torch.tensor([epoch_loss, epoch_samples], device=model.device, dtype=torch.float64)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = (loss_tensor[0] / loss_tensor[1]).item()
        else:
            epoch_loss /= max(epoch_samples, 1)

        if model.is_main_process():
            model.write_on_log(f"Epoch {epoch} Loss: {epoch_loss:.4f}")
            model.save_models()

def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0

def setup_distributed():
    if not is_distributed():
        return 0, 1

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    return local_rank, world_size

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--devices", type=int, nargs="+", required=True, help="GPU indices, e.g. --devices 0 1")
    parser.add_argument("--output_path", type=str, required=True, help="Output folder")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise ValueError(f"Config file '{args.config}' does not exist.")
    if len(args.devices) < 1:
        raise ValueError("Provide at least one GPU index in --devices")

    if not is_distributed():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices[0])

    return args

if __name__ == "__main__":
    main()

'''
nohup torchrun --nproc_per_node=2 train.py --config configs/config.yaml --devices 0 1 --output_path ../test_output &
'''