import argparse

import torch

from src.taxo_expantion_methods.TEMP.temp_model import TEMP
from src.taxo_expantion_methods.app.temp_model_training import run_temp_model_training
from src.taxo_expantion_methods.common.wn_dao import WordNetDao

args = argparse.ArgumentParser(description='Training taxonomy expansion model')
args.add_argument('-d', '--device', default='cpu', type=str, help='used device')
args.add_argument('-e', '--epochs', default=None, type=int, help='number of epochs')
args.add_argument('-l', '--load-path', default=None, type=str, help='path to saved model')
args.add_argument('-res', '--result-path', default='data/models/TEMP/checkpoints', type=str)
args.add_argument('-bs', '--batch-size', default=32, type=int)
args = args.parse_args()

model = TEMP()
if args.load_path is not None:
    model.load_state_dict(torch.load(args.load_path, map_location=torch.device(args.device)))

wn_reader = WordNetDao.get_wn_30()
run_temp_model_training(args.device, args.epochs, args.result_path, model, wn_reader, args.batch_size)