import argparse, yaml, os, mlflow, mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_experiment(cfg):
    # Enable autolog (you can comment if you prefer manual logging)
    mlflow.sklearn.autolog(disable=True)  # we'll log manually for pedagogy

    # Data
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.get("test_size", 0.2), random_state=cfg.get("random_state", 42), stratify=y
    )

    with mlflow.start_run(run_name="sklearn_logreg"):
        # Log params
        mlflow.log_params({
            "solver": cfg.get("solver", "lbfgs"),
            "C": cfg.get("C", 1.0),
            "max_iter": cfg.get("max_iter", 100),
            "test_size": cfg.get("test_size", 0.2),
            "random_state": cfg.get("random_state", 42),
        })

        # Model
        model = LogisticRegression(
            solver=cfg.get("solver", "lbfgs"),
            C=float(cfg.get("C", 1.0)),
            max_iter=int(cfg.get("max_iter", 100))
        )
        model.fit(X_train, y_train)

        # Eval
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred)

        # Log metrics
        mlflow.log_metrics({"accuracy": acc, "f1_macro": f1})

        # Log artifacts
        import matplotlib.pyplot as plt
        import io

        plt.figure()
        import itertools
        classes = iris.target_names
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion matrix")
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, ha="right")
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

        fig_path = "confusion_matrix.png"
        plt.savefig(fig_path, bbox_inches="tight")
        mlflow.log_artifact(fig_path)

        # Log model
        mlflow.sklearn.log_model(model, "model", input_example=X_test.head(2))

        print(f"Run finished. accuracy={acc:.4f} f1_macro={f1:.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="params.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_experiment(cfg)

if __name__ == "__main__":
    main()
