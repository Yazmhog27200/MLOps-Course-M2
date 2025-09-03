import mlflow, mlflow.sklearn, yaml, argparse
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from src.pipeline import build_pipeline

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def main(cfg_path):
    cfg = load_config(cfg_path)

    df = pd.read_csv(cfg["data"]["csv_path"])
    X = df.drop(columns=[cfg["data"]["target"]])
    y = df[cfg["data"]["target"]]

    pipe = build_pipeline(cfg["features"]["numeric"], cfg["features"]["categorical"], cfg["model"]["type"])

    cv = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=cfg["data"]["random_state"])
    grid = GridSearchCV(pipe, cfg["model"]["params"], cv=cv, scoring=cfg["cv"]["scoring"], n_jobs=-1)

    mlflow.sklearn.autolog()

    with mlflow.start_run():
        grid.fit(X, y)

        best_score = grid.best_score_
        mlflow.log_metric("best_cv_score", best_score)
        print("Best CV score:", best_score)
        print("Best params:", grid.best_params_)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
