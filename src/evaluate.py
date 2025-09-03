import argparse, yaml, mlflow
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from src.pipeline import build_pipeline
from sklearn.model_selection import train_test_split

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def main(cfg_path):
    cfg = load_config(cfg_path)
    df = pd.read_csv(cfg["data"]["csv_path"])
    X = df.drop(columns=[cfg["data"]["target"]])
    y = df[cfg["data"]["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["data"]["test_size"], random_state=cfg["data"]["random_state"], stratify=y
    )

    model_uri = "models:/ChurnClassifier/Staging"  # if using registry
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception:
        print("No model in registry, training a fresh one...")
        model = build_pipeline(cfg["features"]["numeric"], cfg["features"]["categorical"], cfg["model"]["type"])
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    with mlflow.start_run():
        mlflow.log_metric("roc_auc_test", roc_auc)
        # ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (AUC={roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig("roc.png")
        mlflow.log_artifact("roc.png")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.savefig("cm.png")
        mlflow.log_artifact("cm.png")

        print("ROC AUC:", roc_auc)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
