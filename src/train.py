import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

def train_model():
    print("Starting Model Training...")
    X_train = np.load("data/processed/X_train_final.npy")
    y_train = np.load("data/processed/y_train.npy")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/random_forest_baseline.pkl")
    print("Model training complete and saved to disk!")

if __name__ == "__main__":
    train_model()
