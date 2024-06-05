from src.taxo_expantion_methods.TaxoPrompt.client.taxo_prompt_client import run

import argparse


args = argparse.ArgumentParser(description='Training taxonomy expansion model')
args.add_argument('-d', '--device', default='cpu', type=str, help='used device')
args.add_argument('-e', '--epochs', default=1, type=int, help='number of epochs')
args.add_argument('-l', '--load-path', default=None, type=str, help='path to saved model')
args.add_argument('-res', '--result-path', default='data/models/TEMP/checkpoints', type=str)
args = args.parse_args()

run(args.device, args.epochs, 0.8, 'data/datasets/taxo-prompt/nouns/ds_food', args.load_path, args.result_path)