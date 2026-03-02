import argparse
import os

from src.Model import Model

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
    pass

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
