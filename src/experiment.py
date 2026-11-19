
import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def train_model(X_train, X_test, y_train, y_test, k=5, version="V1"):

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    mlflow.set_experiment("KNN_Iris")

    with mlflow.start_run(run_name=version):

        model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="macro")

        mlflow.log_param("n_neighbors", k)
        mlflow.log_param("metric", "euclidean")
        mlflow.log_param("version", version)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        joblib.dump(model, "models/knn_" + version + ".pkl")

        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=[
                "sklearn.metrics._dist_metrics.EuclideanDistance64",
                "sklearn.neighbors._kd_tree.KDTree"
            ]
        )

        cm = confusion_matrix(y_test, predictions)

        joblib.dump(
            cm,
            "outputs/confusion_matrix_" + version + ".pkl"
        )

        print("\n", version)
        print("Accuracy:", accuracy)
        print("F1 Score:", f1)
        print("Confusion Matrix:")
        print(cm)

    return model, accuracy, f1
