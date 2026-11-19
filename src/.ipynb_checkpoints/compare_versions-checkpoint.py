import mlflow
import pandas as pd

def compare():

    experiment = mlflow.get_experiment_by_name("KNN_Iris")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    print(runs[[
        "tags.mlflow.runName",
        "params.n_neighbors",
        "metrics.accuracy",
        "metrics.f1_score"
    ]])

    best = runs.sort_values(
        "metrics.f1_score",
        ascending=False
    ).iloc[0]

    print("\nBest Model:")
    print(best["tags.mlflow.runName"])

    return best