from torch.utils.data import DataLoader
from dataloader import custom_dataset
import tiktoken


epochs = 1  # Just for now
split_ratio = 0.85
context_length = 8
offset = 1  # I am wondering what would be the results for 2, for example
batch_size = 4
file_path = "data/poe_data.txt"
tokenizer = tiktoken.get_encoding("cl100k_base")
vocab_size = tokenizer.n_vocab

train_set = custom_dataset.PoeDataset(file_path, 'train', split_ratio, context_length, tokenizer, offset=offset)
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
