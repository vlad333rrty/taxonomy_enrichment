import argparse

from src.taxo_expantion_methods.app.temp_model_training import run_temp_model_training

args = argparse.ArgumentParser(description='Training taxonomy expansion model')
args.add_argument('-d', '--device', default='cpu', type=str, help='indices of GPUs to enable (default: all)')
args.add_argument('-e', '--epochs', default=1, type=int, help='number of epochs')

args = args.parse_args()

run_temp_model_training(args.device, args.epochs)
