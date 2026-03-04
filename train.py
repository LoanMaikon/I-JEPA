import torch.nn.functional as F
import argparse
import torch
import os

from src.Model import Model
from src.models import apply_masks, repeat_interleave_batch

def main():
    args = get_args()

    model = Model(
        operation="train",
        config_path=args.config,
        device=args.device,
        output_path=args.output_path,
    )

    train(model)
    
def train(model):
    scaler = torch.amp.GradScaler()

    for epoch in range(1, model.get_num_epochs() + 1):
        model.write_on_log(f"Epoch {epoch}/{model.get_num_epochs()}")

        for (images, labels), masks_context, masks_pred in model.get_dataloader():
            model.get_optimizer().zero_grad(set_to_none=True)

            imgs = images.to(model.device)
            masks_context = masks_context.to(model.device)
            masks_pred = masks_pred.to(model.device)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                z = model.get_model()(imgs, masks_context)
                z_pred = model.get_predictor()(z, masks_context, masks_pred)

                with torch.no_grad():
                    z_target = model.get_target_model()(imgs, masks_pred)
                    z_target = F.layer_norm(z_target, (z_target.size(-1),)) # Normalize target features
                    z_target = apply_masks(z_target, masks_pred) # Apply masks to target features
                    z_target = repeat_interleave_batch(z_target, imgs.size(0), repeat=len(masks_context))
                
                loss = model.apply_criterion(z_pred, z_target)
                scaler.scale(loss).backward()
                scaler.step(model.get_optimizer())
                scaler.update()

            model.write_on_log(f"Loss: {loss.item():.4f}")
            model.step_schedulers()
            model.update_target_model()

        model.save_models()

def get_args():
    def _handle_args(args):
        if os.path.exists(args.output_path):
            raise ValueError(f"Output path '{args.output_path}' already exists.")
        if not os.path.exists(args.config):
            raise ValueError(f"Config file '{args.config}' does not exist.")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--device', type=int, required=True, help='GPU index to use.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output folder.')
    args = parser.parse_args()
    _handle_args(args)
    return args

if __name__ == "__main__":
    main()
