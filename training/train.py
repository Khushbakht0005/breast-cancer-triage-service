"""Train candidate pipelines, select by CV malignant recall, write artifacts.

In this dataset target 0 = malignant (the positive class), 1 = benign.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MALIGNANT = 0


def build_candidates(params: dict, seed: int) -> dict:
    m = params["models"]
    lr = m["logistic_regression"]
    rf = m["random_forest"]
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=lr["C"], max_iter=lr["max_iter"], class_weight=lr["class_weight"], random_state=seed)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=rf["n_estimators"], max_depth=rf["max_depth"],
                min_samples_leaf=rf["min_samples_leaf"], class_weight=rf["class_weight"], random_state=seed, n_jobs=1)),
        ]),
    }


def evaluate_test(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    mal_idx = list(model.classes_).index(MALIGNANT)
    mal_proba = model.predict_proba(X_test)[:, mal_idx]
    return {
        "malignant_recall": float(recall_score(y_test, pred, pos_label=MALIGNANT)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score((y_test == MALIGNANT).astype(int), mal_proba)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="data/raw/breast_cancer.csv")
    p.add_argument("--output-dir", default="artifacts")
    p.add_argument("--model-version", default="1.0.0")
    p.add_argument("--params", default="params.yaml")
    args = p.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)
    seed = params["seed"]

    df = pd.read_csv(args.data_path)
    feature_names = [c for c in df.columns if c != "target"]
    X = df[feature_names].values  # numpy: the served model gets plain arrays
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], stratify=y, random_state=seed)

    cv = StratifiedKFold(n_splits=params["cv_folds"], shuffle=True, random_state=seed)
    scoring = {
        "recall": make_scorer(recall_score, pos_label=MALIGNANT),
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",  # AUC is identical whichever class is called positive
    }

    candidates = build_candidates(params, seed)
    cv_results = {}
    for name, pipe in candidates.items():
        r = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        cv_results[name] = {
            "malignant_recall": float(r["test_recall"].mean()),
            "accuracy": float(r["test_accuracy"].mean()),
            "roc_auc": float(r["test_roc_auc"].mean()),
        }

    best_name = max(cv_results, key=lambda n: cv_results[n]["malignant_recall"])
    best = candidates[best_name]
    best.fit(X_train, y_train)

    test_results = {name: None for name in candidates}
    for name, pipe in candidates.items():
        if name != best_name:
            pipe.fit(X_train, y_train)
        test_results[name] = evaluate_test(pipe, X_test, y_test)
    selected_test = test_results[best_name]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, out / "model.joblib")

    card = {
        "model_version": args.model_version,
        "model_family": best_name,
        "positive_class": "malignant (target 0)",
        "feature_names": feature_names,
        "metrics": {"test": selected_test, "cv_mean": cv_results[best_name]},
        "seed": seed,
        "sklearn_version": sklearn.__version__,
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2))
    metrics = {"selected_model": best_name, "cv": cv_results,
               "test_all_models": test_results, "test_selected": selected_test}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"{'model':<20}{'cv_recall':>10}{'cv_acc':>9}{'cv_auc':>9}{'t_recall':>10}{'t_acc':>9}{'t_auc':>9}")
    for n in candidates:
        c, t = cv_results[n], test_results[n]
        mark = "  <- selected" if n == best_name else ""
        print(f"{n:<20}{c['malignant_recall']:>10.4f}{c['accuracy']:>9.4f}{c['roc_auc']:>9.4f}"
              f"{t['malignant_recall']:>10.4f}{t['accuracy']:>9.4f}{t['roc_auc']:>9.4f}{mark}")
    print(f"Saved model + card to {out} (version {args.model_version})")

    if selected_test["malignant_recall"] < params["min_test_recall"]:
        print(f"FAIL: test malignant recall {selected_test['malignant_recall']:.4f} "
              f"< {params['min_test_recall']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
