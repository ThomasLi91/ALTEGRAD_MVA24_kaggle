import torch
import wandb
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(
    model, loader, criterion, similarity_fn, metric_fn, log_prefix, example_ct, wandb_log = True
):
    model.eval()
    with torch.no_grad():
        graph_embeddings = []
        text_embeddings = []
        for batch in loader:
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
        if wandb_log:
            wandb.log(
                {log_prefix + "_LRAP": LRAP, log_prefix + "_loss": epoch_loss},
                step=example_ct,
            )

    return similarity_matrix