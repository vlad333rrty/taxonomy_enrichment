from src.taxo_expantion_methods.TaxoPrompt.client.taxo_prompt_client import run

device = 'cuda:0'
epochs=16
run(device, epochs, 0.7, 'data/datasets/taxo-prompt/nouns/ds', 'data/models/TaxoPrompt/pre-trained/taxo_prompt_model_epoch_2')