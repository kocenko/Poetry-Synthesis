import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def calc_loss(model, iterations, batch_size, train_set, val_set):
    ''' Used to evalute model by averaging on many iterations
    
    Args:
        model: Evaluated model
        iterations: Number of iterations to average through
        batch_size: Batch size
        train_set: Training dataset
        val_set: Validation dataset

    Returns:
        Dictionary with averaged losses for 'train' nad 'valid'
    '''

    split_type = ["train", "valid"]
    outcome_losses = {}
    model.eval()
    for t, split in enumerate([train_set, val_set]):
        loader = DataLoader(split, batch_size = batch_size, shuffle=True, drop_last=True)
        loader = iter(loader)
        losses = torch.zeros(iterations)
        for i in range(iterations):
            x, y = loader.__next__()
            _, loss = model(x, y)
            losses[i] = loss.item()
        outcome_losses[split_type[t]] = losses.mean()
    model.train()
    return outcome_losses


def train_model(model, train_set, valid_set, hyper_params: dict, device):
    ''' Trains the model

    Args:
        model: Model to train
        train_set: Training dataset
        valid_set: Validation dataset
        hyper_params: dict of hyperparameters
    '''

    lr = hyper_params["lr"]
    epochs = hyper_params["epochs"]
    batch_size = hyper_params["batch_size"]
    eval_per_epoch = hyper_params["eval_per_epoch"]
    eval_iterations = hyper_params["eval_iterations"]
    break_iter = hyper_params["break_iter"]
    if break_iter == None:
        break_iter = len(train_set) // batch_size

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    eval_each = break_iter // eval_per_epoch
        

    for e in range(epochs):
        for i, (x, y) in enumerate(train_dataloader):
            if i == break_iter:
                break

            if i % eval_each == 0:
                losses = calc_loss(model, eval_iterations, batch_size, train_set, valid_set)
                print(f"Epoch: [{e+1}/{epochs}] Step: [{i}/{break_iter}], train loss: {losses['train']:.4f}, val loss: {losses['valid']:.4f}")
            
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
