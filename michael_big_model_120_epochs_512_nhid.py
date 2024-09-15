# Imports
# import os
# os.environ["WANDB_MODE"] = "offline"
import wandb
wandb.login()
from src.pipeline import model_pipeline
from src.Model4 import Model # TODO
import torch

print("Model4")


# Config
config = dict(
    model_name='distilbert-base-uncased',
    batch_size=32,
    graph_learning_rate=3e-4,
    text_learning_rate=3e-5,
    learning_rate=1e-4,
    epochs=120,
    criterion="negative_sampling_contrastive_loss",
    similarity_fn="cosine_similarity",
    layer_norm=True,
    train_val=False,
    turbo=True,
)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device :", device)

model = Model(
    model_name=config["model_name"],
    num_node_features=300,
    nout=768,
    nhid=512,
    graph_hidden_channels=512,
)
model.to(device)

# Run Name
run_name = "michael_big_model_120_epochs_512_nhid_online"


# This block ensures that the code inside main() is only executed if the script is run directly
if __name__ == "__main__":
    model_pipeline(config, model, run_name=run_name)