from src.taxo_expantion_methods.TaxoPrompt.client.taxo_prompt_client import run

device = 'cpu'
epochs=4
run(device, epochs, 0.7, 'data/datasets/taxo-prompt/nouns/ds', 'data/models/TaxoPrompt/checkpoints/taxo_prompt_model_epoch_0')