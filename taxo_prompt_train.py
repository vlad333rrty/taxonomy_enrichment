from src.taxo_expantion_methods.TaxoPrompt.client.taxo_prompt_client import run

device = 'cuda:0'
epochs=10
run(device, epochs, 6)