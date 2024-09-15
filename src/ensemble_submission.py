from src.utils import get_date_time_string
import numpy as np
import pandas as pd


def softmax(x):
    # Ensure numerical stability by subtracting the maximum value
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)




def get_ensemble_submission(liste_submission_paths, liste_poids, rank = True, softmax_bool = False):
    assert len(liste_submission_paths) == len(liste_poids)
    ensemble_submission = 0
    for i, submission_path in enumerate(liste_submission_paths):
        print(submission_path)
        submission = pd.read_csv(submission_path)
        submission = submission.drop('ID', axis = 1)
        if rank:
            submission = submission.apply(lambda x: x.rank(), axis=1)
        submission = submission.to_numpy()
        if softmax:
            submission = softmax(submission)
        ensemble_submission += liste_poids[i] * submission
    ensemble_submission = ensemble_submission / len(liste_submission_paths)
    solution = pd.DataFrame(ensemble_submission)
    solution["ID"] = solution.index
    solution = solution[["ID"] + [col for col in solution.columns if col != "ID"]]
    submission_path = "submissions/ensemble_submission" + str(len(liste_submission_paths)) + get_date_time_string() + ".csv"
    solution.to_csv(submission_path, index=False)
    print("Saved ensemble submission in", submission_path)