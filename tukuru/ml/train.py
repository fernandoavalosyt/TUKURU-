"""Entrena una referencia tabular local; nunca activa el modelo automaticamente."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .dataset import group_cases, labeled_rows
from .features import FEATURE_VERSION
from .model import FEATURE_NAMES, feature_vector


def split_for_validation(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deduplica episodios de 120 s y reserva origenes completos para evaluacion."""
    episodes = group_cases(rows)
    if any(len({row["label"] for row in episode}) > 1 for episode in episodes):
        raise ValueError("Hay episodios con etiquetas humanas distintas; resuelva la revision antes de entrenar")
    deduplicated = [episode[-1] for episode in episodes]
    sessions = sorted({row["session_id"] for row in deduplicated},
                      key=lambda session: min(r["occurred_at"] for r in deduplicated if r["session_id"] == session))
    if len(sessions) < 4:
        raise ValueError("Se requieren al menos 4 sesiones independientes; un video repetido no cuenta como sesiones distintas")
    holdout = set(sessions[-max(1, len(sessions) // 4):])
    train = [row for row in deduplicated if row["session_id"] not in holdout]
    test = [row for row in deduplicated if row["session_id"] in holdout]
    for name, part in (("entrenamiento", train), ("evaluacion", test)):
        counts = Counter(row["label"] for row in part)
        if min(counts.get(0, 0), counts.get(1, 0)) < (10 if name == "entrenamiento" else 3):
            raise ValueError(f"Faltan casos confirmados o descartados independientes en {name}: {dict(counts)}")
    return train, test


def fit(rows: list[dict]) -> dict:
    try:
        import sklearn
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, precision_score, recall_score
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("Instale requirements-ml.txt para entrenar") from exc

    train, test = split_for_validation(rows)
    x_train = np.stack([feature_vector(row) for row in train])
    x_test = np.stack([feature_vector(row) for row in test])
    y_train = np.asarray([row["label"] for row in train], dtype=np.int64)
    y_test = np.asarray([row["label"] for row in test], dtype=np.int64)
    scaler = StandardScaler().fit(x_train)
    classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42).fit(
        scaler.transform(x_train), y_train)
    scores = classifier.predict_proba(scaler.transform(x_test))[:, 1]
    predictions = scores >= 0.5
    forest = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, class_weight="balanced_subsample",
                                    random_state=42, n_jobs=1).fit(x_train, y_train)
    forest_scores = forest.predict_proba(x_test)[:, 1]
    metrics = {
        "evaluacion_n": len(test), "positivos_n": int(y_test.sum()),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recobrado": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "average_precision": round(float(average_precision_score(y_test, scores)), 4),
        "precision_alertas_actuales": round(float(y_test.mean()), 4),
        "random_forest_referencia": {
            "precision": round(float(precision_score(y_test, forest_scores >= 0.5, zero_division=0)), 4),
            "recobrado": round(float(recall_score(y_test, forest_scores >= 0.5, zero_division=0)), 4),
            "average_precision": round(float(average_precision_score(y_test, forest_scores)), 4),
        },
    }
    return {"version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "algorithm": "logistic_regression", "feature_version": FEATURE_VERSION,
            "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
            "feature_names": FEATURE_NAMES, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coef": classifier.coef_[0].tolist(), "intercept": float(classifier.intercept_[0]),
            "training_n": len(train), "training_sessions": len({r["session_id"] for r in train}),
            "evaluation_sessions": len({r["session_id"] for r in test}), "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--db", type=Path, default=root / "data/incidents.sqlite3")
    parser.add_argument("--out", type=Path, default=root / "data/models_ml/candidato.json")
    args = parser.parse_args()
    if not args.db.is_file():
        parser.error("No hay base de incidentes; registre y revise casos primero")
    try:
        artifact = fit(labeled_rows(args.db))
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, args.out)
    print(f"Candidato guardado en {args.out}; no se activo en Tukuru")
    print(json.dumps(artifact["metrics"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
