from sklearn.metrics.pairwise import cosine_similarity
import wandb
from tqdm import tqdm
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

similarity_dico = {
    "cosine_similarity": cosine_similarity,
}


def get_empty_edge_index(batch):
    n_nodes = batch.x.shape[0]
    self_edge_index = torch.arange(n_nodes).unsqueeze(0)
    self_edge_index = torch.cat((self_edge_index, self_edge_index), dim=0)
    return self_edge_index



def finetune_train_one_epoch(
    model, finetune_train_loader, criterion, optimizer, example_ct_before
):
    log_every_n_batches = 50  # TODO
    # Tell wandb to watch the model (gradients, weights) over time
    # wandb.watch(model, criterion, log="all", log_freq=log_every_n_batches)  # TODO

    # Run training
    model.train()
    example_ct = example_ct_before  # number of examples seen (example count)
    batch_ct = 0  # number of batches seen (batch count)
    moving_average_loss = 0
    log_example_ct = 0
    for batch in tqdm(finetune_train_loader):
        current_loss, batch_example_ct = train_batch(batch, model, optimizer, criterion)
        example_ct += batch_example_ct
        batch_ct += 1
        moving_average_loss += current_loss
        log_example_ct += batch_example_ct

        # log metrics every 25th batch with wandb
        if ((batch_ct + 1) % log_every_n_batches) == 0:
            moving_average_loss /= log_example_ct
            wandb.log(
                {
                    "epoch": example_ct / 26000,
                    "train_current_loss": moving_average_loss,
                },
                step=example_ct,
            )
            print(
                f"train_loss after {str(example_ct).zfill(5)} examples: {moving_average_loss:.5f}"
            )
            moving_average_loss = 0
            log_example_ct = 0


def train_batch(batch, model, optimizer, criterion):
    # extract batch
    input_ids = batch.input_ids
    batch_example_ct = batch.input_ids.shape[0]
    if batch.edge_index.shape[0] == 0:
        edge_index = get_empty_edge_index(batch)
        batch.edge_index = edge_index
    batch.pop("input_ids")
    attention_mask = batch.attention_mask
    batch.pop("attention_mask")
    graph_batch = batch

    # compute loss
    x_graph, x_text = model(
        graph_batch.to(device), input_ids.to(device), attention_mask.to(device)
    )
    current_loss = criterion(x_graph, x_text)
    optimizer.zero_grad()
    current_loss.backward()
    optimizer.step()

    return current_loss, batch_example_ct