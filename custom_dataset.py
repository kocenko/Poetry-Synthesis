import torch
from torch.utils.data import Dataset


class PoeDataset(Dataset):
    valid_split_params = ["train", "valid"]

    def __init__(self, text: str, split: str, split_ratio: float, context_length: int, tokenizer, device, offset: int = 1):
        ''' Poe Dataset constructor

        Args:
            str:
                file_path: Path to the file containing dataset
                splt: String indicating what type of data this dataset contains
            float:
                split_ratio: Value between (0, 1] of what should be the ratio
                             between training and validation set
            int:
                context_length: Length of the context
                offset: An offset between the end of the context and the target
        '''

        assert split in PoeDataset.valid_split_params, f"{split} is the wrong split type"
        assert split_ratio <= 1 and split_ratio > 0, f"Split ratio value should be from range (0, 1]"
        assert len(text) > 0, f"Dataset file should not be empty"
        assert context_length < len(text), f"Context length should not be more than {len(text) - 1}"

        self.text = text
        self.offset = offset
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.data = torch.tensor(self.tokenizer.encode(self.text), dtype=torch.int32, device=device)

        split_idx = int(len(self.data) * split_ratio)
        if split == "train":
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]
        

    def __len__(self):
        ''' Returns the size of the dataset
        
        Returns:
            Number of possible shifts in the dataset for choosing the context chunk
        '''
        return len(self.data) - self.context_length - self.offset + 1
    
    def __getitem__(self, index):
        ''' Returns an item of given index

        Params:
            index: Which item should be returned
        
        Returns:
            Sample of given index
        '''

        x = self.data[index: index + self.context_length]
        y = self.data[index + self.offset: index + self.context_length + self.offset]

        return x, y

