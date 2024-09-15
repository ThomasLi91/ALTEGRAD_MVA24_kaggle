from datetime import datetime

def get_date_time_string():
    # Get current date and time
    now = datetime.now()

    # Extract year, month, day, hour, and minute
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute

    # Format as a string of integers (yyyymmddHHMM)
    date_time_string = f"_{month:02d}{day:02d}{hour:02d}{minute:02d}"

    return date_time_string


import numpy as np
from sklearn.metrics import label_ranking_average_precision_score

def LRAP_accuracy(similarity_matrix):
    N, M = similarity_matrix.shape
    assert N == M
    y_true = np.eye(N, dtype=int)
    LRAP = label_ranking_average_precision_score(y_true, similarity_matrix)

    return LRAP


import yaml

def load_yaml_config_as_dict(yaml_path):
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
        config.pop("_wandb")
        return config
    

thisdict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}