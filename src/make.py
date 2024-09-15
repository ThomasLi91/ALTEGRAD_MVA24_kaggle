from torch.utils.data import ConcatDataset
from transformers import AutoTokenizer
from torch_geometric.data import DataLoader
from src.losses import criterion_dico
import numpy as np
from original_src.dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch import optim



def make(config, model):
    # Make the data (train_loader, val_loader)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = GraphTextDataset(
        root="./data/", gt=gt, split="val", tokenizer=tokenizer
    )
    train_dataset = GraphTextDataset(
        root="./data/", gt=gt, split="train", tokenizer=tokenizer
    )
    if config.train_val:
        train_dataset = ConcatDataset([train_dataset, val_dataset])

    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=16
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=16
    )

    # Make the loss and optimizer (criterion, optimizer)
    criterion = criterion_dico[config.criterion]
    model_params_dico = [
        {"params": model.graph_encoder.parameters(), "lr": config.graph_learning_rate},
        {"params": model.text_encoder.parameters(), "lr": config.text_learning_rate},
    ]
    optimizer = optim.AdamW(
        model_params_dico,
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    return train_loader, val_loader, criterion, optimizer