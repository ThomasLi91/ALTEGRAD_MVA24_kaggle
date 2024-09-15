import wandb
from src.utils import get_date_time_string
import torch
from src.make import make
from src.train import train

PROJECT_NAME = "ALTEGRAD_kaggle"  # TODO



def model_pipeline(
    hyperparameters, model, run_name="run", model_path=None, n_epochs_before=0
):
    print(hyperparameters)
    # tell wandb to get started
    run_name = run_name + get_date_time_string()
    print("run_name : ", run_name)
    with wandb.init(project=PROJECT_NAME, config=hyperparameters, name=run_name):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        print("Making dataloaders and loading model")
        if model_path != None:
            model.load_state_dict(torch.load(model_path))  # a .pth file
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        train_loader, val_loader, criterion, optimizer = make(config, model)

        # and use them to train the model
        print("Start training")
        if model_path != None:
            n_train_samples = 26408
            example_ct_before = n_epochs_before * n_train_samples
        else:
            example_ct_before = 0
        train(
            model,
            train_loader,
            criterion,
            optimizer,
            config,
            run_name,
            val_loader=val_loader,
            example_ct_before=example_ct_before,
        )
        print("Successful training")

        return model