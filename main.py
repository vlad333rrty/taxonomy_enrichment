import argparse

import torch

from src.taxo_expantion_methods.TEMP.temp_model import TEMP
from src.taxo_expantion_methods.app.temp_model_training import run_temp_model_training

args = argparse.ArgumentParser(description='Training taxonomy expansion model')
args.add_argument('-d', '--device', default='cpu', type=str, help='used device')
args.add_argument('-e', '--epochs', default=None, type=int, help='number of epochs')
args.add_argument('-l', '--load-path', default=None, type=str, help='path to saved model')
args = args.parse_args()

model = TEMP(768 * 2, 256)
if args.load_path is not None:
    model = model.load_state_dict(torch.load(args.load_state, map_location=torch.device(args.device)))

run_temp_model_training(args.device, args.epochs, model)
