from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import os
from src.utils import load_yaml_config_as_dict
from transformers import AutoTokenizer
from original_src.dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm
import pandas as pd


def get_scalar_product_similarity(text_embeddings, graph_embeddings):
    text_embeddings = np.array(text_embeddings)
    graph_embeddings = np.array(graph_embeddings)
    similarity = text_embeddings @ (graph_embeddings.T)
    return similarity


similarity_dico = {
    "cosine_similarity": cosine_similarity,
    "scalar_product_similarity": get_scalar_product_similarity,
}


# Default config as class instance
class Config_Class:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)


def get_submission(model_path, model, similarity_name="cosine_similarity"):
    # Load model
    # yaml_path = os.path.dirname(model_path) + "/config.yaml"
    # config_dico = load_yaml_config_as_dict(yaml_path)
    # config = Config_Class(config_dico)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Model(
    #     model_name=config.model_name,
    #     num_node_features=300,
    #     nout=768,
    #     nhid=300,
    #     graph_hidden_channels=300,
    # )
    model.load_state_dict(torch.load(model_path))  # a .pth file
    print("Loaded the model")
    model.to(device)
    model.eval()

    batch_size = 64
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    similarity_fn = similarity_dico[similarity_name]

    # Compute the predictions
    graph_model = model.get_graph_encoder()
    text_model = model.get_text_encoder()
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
    test_cids_dataset = GraphDataset(root="./data/", gt=gt, split="test_cids")
    test_text_dataset = TextDataset(
        file_path="./data/test_text.txt", tokenizer=tokenizer
    )
    idx_to_cid = test_cids_dataset.get_idx_to_cid()
    test_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)

    graph_embeddings = []
    for batch in tqdm(test_loader):
        for output in graph_model(batch.to(device)):
            graph_embeddings.append(output.tolist())

    test_text_loader = TorchDataLoader(
        test_text_dataset, batch_size=batch_size, shuffle=False
    )
    text_embeddings = []
    for batch in tqdm(test_text_loader):
        for output in text_model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ):
            text_embeddings.append(output.tolist())

    similarity = similarity_fn(text_embeddings, graph_embeddings)  # TODO
    solution = pd.DataFrame(similarity)
    solution["ID"] = solution.index
    solution = solution[["ID"] + [col for col in solution.columns if col != "ID"]]

    # Saving the predictions in a csv file
    folder_name = "submissions/"
    file_name = os.path.basename(model_path)[:-4]  # get file name and remove ".pth"
    file_name += "_submission.csv"
    submission_path = folder_name + file_name
    solution.to_csv(submission_path, index=False)
    print("Successfully saved submission in : ", submission_path)

    return solution