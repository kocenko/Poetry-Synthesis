from torch.utils.data import DataLoader
from custom_dataset import PoeDataset
from only_decoder import OnlyDecoder
from tokenizer import Tokenizer


epochs = 1  # Just for now
split_ratio = 0.85
context_length = 8
offset = 1  # I am wondering what would be the results for 2, for example
batch_size = 4
file_path = "data/poe_data.txt"

# Reading file, preparing tokenizer
with open(file_path, 'r', encoding="utf-8") as f:
            text = f.read()

tokenizer = Tokenizer(text)

# Dataset and dataloader
train_set = PoeDataset(text, 'train', split_ratio, context_length, tokenizer, offset=offset)
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
x, y = next(iter(train_dataloader))

# # Model
# model = OnlyDecoder(net_config)
# out = model(x, y)
# print(out.shape)