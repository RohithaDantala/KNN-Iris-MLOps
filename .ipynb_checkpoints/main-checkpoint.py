from src.data_loader import load_data
from src.preprocess import preprocess
from src.experiment import train_model
from src.compare_versions import compare

X, y = load_data()

X_train, X_test, y_train, y_test, scaler, pca, encoder = preprocess(X, y)

model1, acc1, f1_1 = train_model(
    X_train, X_test, y_train, y_test,
    k=5, version="V1"
)

model2, acc2, f1_2 = train_model(
    X_train, X_test, y_train, y_test,
    k=7, version="V2"
)

best = compare()

if f1_1 >= f1_2:
    best_model = model1
else:
    best_model = model2

import joblib

joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(pca, "models/pca.pkl")
joblib.dump(encoder, "models/encoder.pkl")

print("\nBest Model Saved Successfully")