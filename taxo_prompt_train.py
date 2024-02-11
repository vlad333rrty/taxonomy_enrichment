from src.taxo_expantion_methods.TaxoPrompt.client.taxo_prompt_client import run

device = 'cpu'
epochs=16
run(device, epochs, 0.01, 'data/datasets/taxo-prompt/nouns/ds')