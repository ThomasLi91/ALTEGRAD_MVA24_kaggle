from src.losses import criterion_dico
from src.batch_sampler import get_batch_sampler
from torch.utils.data import DataLoader as TorchDataLoader
from transformers import AutoTokenizer
from original_src.dataloader import GraphTextDataset
from torch_geometric.data import DataLoader
from torch import optim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import LRAP_accuracy
from src.finetune_evaluate import evaluate



def make(config, model, shuffle=True):
    # Make the data (train_loader, val_loader)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = GraphTextDataset(
        root="./data/", gt=gt, split="val", tokenizer=tokenizer
    )
    train_dataset = GraphTextDataset(
        root="./data/", gt=gt, split="train", tokenizer=tokenizer
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=16
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=16
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




def finetune_make(model, train_loader, criterion, example_ct_before, config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
    train_dataset = GraphTextDataset(
        root="./data/", gt=gt, split="train", tokenizer=tokenizer
    )
    train_similarity_matrix = evaluate(
        model,
        train_loader,
        criterion,
        cosine_similarity,
        LRAP_accuracy,
        "train",
        example_ct_before,
        wandb_log=True,
    )
    batch_sampler, stats_dico = get_batch_sampler(
        train_similarity_matrix, config.batch_size
    )
    finetune_train_loader = DataLoader(
        train_dataset, batch_sampler=batch_sampler, num_workers=16
    )
    finetune_train_data_size = config.batch_size * len(finetune_train_loader)

    return finetune_train_loader, finetune_train_data_size