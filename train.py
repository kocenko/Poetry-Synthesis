from torch.utils.data import DataLoader
from dataloader import custom_dataset


max_iterations = 100  # 6000 would be better
split_ratio = 0.85
context_length = 8
offset = 1
batch_size = 4

file_path = "data/poe_data.txt"
train_set = custom_dataset.PoeDataset(file_path, 'train', split_ratio, context_length, offset=offset)
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
train_iterator = iter(train_dataloader)

for _ in range(10):
    print(next(train_iterator))