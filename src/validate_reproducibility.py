import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def train_once(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return f1_score(y_test, model.predict(X_test))

def main():
    print("Running Reproducibility Validation...")
    X_train = np.load("data/processed/X_train_final.npy")
    X_test = np.load("data/processed/X_test_final.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    f1_1 = train_once(X_train, y_train, X_test, y_test)
    f1_2 = train_once(X_train, y_train, X_test, y_test)

    print(f"Execution 1 F1: {f1_1:.6f}")
    print(f"Execution 2 F1: {f1_2:.6f}")

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/reproducibility_report.txt", "w") as f:
        f.write(f"Execution 1 F1: {f1_1:.6f}\n")
        f.write(f"Execution 2 F1: {f1_2:.6f}\n")
        f.write(f"Reproducible: {f1_1 == f1_2}\n")

    if f1_1 == f1_2:
        print("SUCCESS: Pipeline is 100% reproducible. Report saved.")
    else:
        raise SystemExit("ERROR: Pipeline is not reproducible.")

if __name__ == "__main__":
    main()
