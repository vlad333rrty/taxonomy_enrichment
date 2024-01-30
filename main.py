import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
encoding = tokenizer.batch_encode_plus(
    ['I Love You', ' Do', 'You', 'Love'],  # List of input texts
    padding=True,  # Pad to the maximum sequence length
    truncation=True,  # Truncate to the maximum sequence length if necessary
    return_tensors='pt',  # Return PyTorch tensors
    add_special_tokens=True  # Add special tokens CLS and SEP
)
input_ids = encoding['input_ids']  # Token IDs
attention_mask = encoding['attention_mask']  # Attention mask
with torch.no_grad():
    outputs = bert_model(input_ids, attention_mask=attention_mask)
    word_embeddings = outputs.last_hidden_state  # This contains the embeddings

# Output the shape of word embeddings
print(f"Shape of Word Embeddings: {word_embeddings.shape}")
