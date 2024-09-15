import torch

CE = torch.nn.CrossEntropyLoss()

def contrastive_loss(v1, v2):
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)


def contrastive_loss_graph_classif(x_graph, x_text):
    logits = torch.matmul(x_graph, torch.transpose(x_text, 0, 1))  # (graph, text)
    labels = torch.arange(logits.shape[0], device=x_graph.device)
    return CE(torch.transpose(logits, 0, 1), labels)


def n_uplet_loss(x_graph, x_text):
    # x_graph and x_text must be normalized
    logits = torch.matmul(x_graph, torch.transpose(x_text, 0, 1))  # (graph, text)
    batch_size = logits.shape[0]
    device = x_graph.device
    labels = 2 * torch.eye(batch_size, device=device) - torch.ones(
        (batch_size, batch_size), device=device
    )
    loss = torch.mean((logits - labels) ** 2)

    return loss


BCEL = torch.nn.BCEWithLogitsLoss()


def negative_sampling_contrastive_loss(v1, v2):
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
    labels = torch.diag_embed(torch.ones(logits.shape[0])).to(v1.device)
    return BCEL(logits, labels)



criterion_dico = {
    "contrastive_loss": contrastive_loss,
    "contrastive_loss_graph_classif": contrastive_loss_graph_classif,
    "n_uplet_loss": n_uplet_loss,
    "negative_sampling_contrastive_loss": negative_sampling_contrastive_loss,
}