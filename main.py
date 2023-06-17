import torch
from custom_dataset import PoeDataset
from only_decoder import OnlyDecoder
from tokenizer import Tokenizer
from train import train_model
import tiktoken



def main():
    # Setting up GPU if available
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    file_path = "data/poe_data.txt"

    # Reading file, preparing tokenizer
    with open(file_path, 'r', encoding="utf-8") as f:
                text = f.read()

    # Setting up dataset
    split_ratio = 0.9
    context_length = 256
    offset = 1  # I am wondering what would be the results for 2, for example
    custom_tokenizer = False

    if custom_tokenizer:
        tokenizer = Tokenizer(text)
        vocab_size = tokenizer.vocab_size
    else:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        vocab_size = tokenizer.n_vocab

    net_config = { "vocab_size": vocab_size,
                   "n_embed": 32,
                   "context_length": context_length,
                   "att_head_num": 4,
                   "blocks_num": 3,
                   "dropout": .2}
    
    model = OnlyDecoder(net_config)
    model.to(device)

    # Training parameters
    hypers = {
        "lr": .3e-4,
        "epochs": 2,
        "batch_size": 8,
        "eval_per_epoch": 10,
        "eval_iterations": 200,
        "break_iter": None
    }

    # Training
    train_set = PoeDataset(text, 'train', split_ratio, context_length, tokenizer, offset=offset)
    val_set = PoeDataset(text, 'valid', split_ratio, context_length, tokenizer, offset=offset)

    # TODO: add saving to file along the training, so to load when interrupted
    train_model(model, train_set, val_set, hypers, device=device)
    save_path = 'state.pt'
    torch.save(model.state_dict(), save_path)
    test_text_length = 200

    # Test it
    starter = torch.zeros((1,1), dtype=torch.long, device=device)
    print(tokenizer.decode([t.item() for t in model.generate_new_text(starter, test_text_length)]))

if __name__ == '__main__':
     main()