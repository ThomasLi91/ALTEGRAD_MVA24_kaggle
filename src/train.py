from sklearn.metrics.pairwise import cosine_similarity
import yaml
from tqdm import tqdm
import torch
import os
import wandb
from src.utils import get_date_time_string, LRAP_accuracy, load_yaml_config_as_dict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


similarity_dico = {
    "cosine_similarity": cosine_similarity,
}


def train(
    model,
    train_loader,
    criterion,
    optimizer,
    config,
    run_name,
    val_loader=None,
    example_ct_before=0,
):
    log_every_n_batches = 50  # TODO
    # Tell wandb to watch the model (gradients, weights) over time
    # wandb.watch(model, criterion, log="all", log_freq=log_every_n_batches)  # TODO

    # Setup saving parameters
    folder_path = "models/" + run_name
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    yaml_file_path = folder_path + "/" + "config.yaml"
    with open(yaml_file_path, "w") as file:
        yaml.dump(wandb.config._as_dict(), file, default_flow_style=False)

    # Run training
    model.train()
    example_ct = example_ct_before  # number of examples seen (example count)
    batch_ct = 0  # number of batches seen (batch count)
    for epoch in tqdm(range(config.epochs)):
        # Train on train dataloader
        moving_average_loss = 0
        log_example_ct = 0
        for batch in tqdm(train_loader, leave=False):
            current_loss, batch_example_ct = train_batch(
                batch, model, optimizer, criterion
            )
            example_ct += batch_example_ct
            batch_ct += 1
            moving_average_loss += current_loss
            log_example_ct += batch_example_ct

            # log metrics every 25th batch with wandb
            if ((batch_ct + 1) % log_every_n_batches) == 0:
                moving_average_loss /= log_example_ct
                wandb.log(
                    {"epoch": epoch, "train_current_loss": moving_average_loss},
                    step=example_ct,
                )
                print(
                    f"train_loss after {str(example_ct).zfill(5)} examples: {moving_average_loss:.5f}"
                )
                moving_average_loss = 0
                log_example_ct = 0

        # Evaluation
        similarity_fn = similarity_dico[config.similarity_fn]

        # Evaluate on train dataloader
        if config.turbo == False:
            train_similarity_matrix = evaluate(
                model=model,
                loader=train_loader,
                criterion=criterion,
                similarity_fn=similarity_fn,
                metric_fn=LRAP_accuracy,
                log_prefix="train",
                example_ct=example_ct,
            )

        # Evaluate on val dataloader
        if val_loader != None:
            val_similarity_matrix = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                similarity_fn=similarity_fn,
                metric_fn=LRAP_accuracy,
                log_prefix="val",
                example_ct=example_ct,
            )

        # Save the model at every epoch
        save_path = folder_path + "/" + run_name + ".pth"
        torch.save(model.state_dict(), save_path)


def train_batch(batch, model, optimizer, criterion):
    # extract batch
    input_ids = batch.input_ids
    batch_example_ct = batch.input_ids.shape[0]
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



def evaluate(
    model, loader, criterion, similarity_fn, metric_fn, log_prefix, example_ct
):
    model.eval()
    with torch.no_grad():
        graph_embeddings = []
        text_embeddings = []
        for batch in tqdm(loader, leave=False):
            input_ids = batch.input_ids
            batch.pop("input_ids")
            attention_mask = batch.attention_mask
            batch.pop("attention_mask")
            graph_batch = batch

            # compute loss
            x_graph, x_text = model(
                graph_batch.to(device), input_ids.to(device), attention_mask.to(device)
            )

            for output in x_graph:
                graph_embeddings.append(output.tolist())
            for output in x_text:
                text_embeddings.append(output.tolist())

        # Compute metrics on the whole dataset
        similarity_matrix = similarity_fn(text_embeddings, graph_embeddings)
        LRAP = metric_fn(similarity_matrix)
        epoch_loss = criterion(
            torch.FloatTensor(text_embeddings), torch.FloatTensor(graph_embeddings)
        )

        # log the metrics
        wandb.log(
            {log_prefix + "_LRAP": LRAP, log_prefix + "_loss": epoch_loss},
            step=example_ct,
        )

    return similarity_matrix