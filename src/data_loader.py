import pandas as pd
from sklearn.datasets import load_iris

def load_data():
    try:
        df = pd.read_csv("data/Iris.csv")
        df = df.drop(columns=["Id"], errors="ignore")
        df = df.drop_duplicates()
        X = df.drop("species", axis=1)
        y = df["species"]
    except FileNotFoundError:
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target).map(dict(enumerate(iris.target_names)))

    return X, y

