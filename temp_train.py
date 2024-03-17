import argparse

import torch

from src.taxo_expantion_methods.TEMP.temp_model import TEMP
from src.taxo_expantion_methods.app.temp_model_training import run_temp_model_training, run_temp_model_training_ru

args = argparse.ArgumentParser(description='Training taxonomy expansion model')
args.add_argument('-d', '--device', default='cpu', type=str, help='used device')
args.add_argument('-e', '--epochs', default=None, type=int, help='number of epochs')
args.add_argument('-l', '--load-path', default=None, type=str, help='path to saved model')
args.add_argument('-lng', '--language', default='en', type=str, help='en or ru')
args = args.parse_args()

model = TEMP()
if args.load_path is not None:
    model.load_state_dict(torch.load(args.load_path, map_location=torch.device(args.device)))

if args.language == 'en':
    run_temp_model_training(args.device, args.epochs, model)
else:
    run_temp_model_training_ru(args.device, args.epochs, model)