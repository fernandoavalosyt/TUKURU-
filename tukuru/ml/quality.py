"""Audita etiquetas, origenes independientes y fuga potencial sin entrenar modelos."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .dataset import group_cases, labeled_rows
from .features import FEATURE_VERSION
from .novelty import reference_readiness
from .repository import IncidentRepository
from .train import split_for_validation


def audit(database: Path) -> dict:
    repo = IncidentRepository(database)
    summary = repo.quality_summary(FEATURE_VERSION)
    rows = labeled_rows(database)
    episodes = group_cases(rows)
    deduplicated = [episode[-1] for episode in episodes]
    conflicts = sum(1 for episode in episodes if len({row["label"] for row in episode}) > 1)
    labels = Counter(row["label"] for row in deduplicated)
    summary.update({"filas_etiquetadas_con_origen": len(rows), "duplicados_mismo_track_origen_120s": len(rows)-len(deduplicated),
                    "filas_tras_deduplicar": len(deduplicated), "clase_confirmada_tras_deduplicar": labels[1],
                    "clase_descartada_tras_deduplicar": labels[0], "episodios_con_etiquetas_conflictivas": conflicts})
    origins = {row["session_id"] for row in deduplicated}
    summary["grupos_origen_etiquetados"] = len(origins)
    summary["independencia_verificada"] = False
    novelty_reference, novelty_reason = reference_readiness(rows)
    summary["referencias_normales_para_novedad"] = len(novelty_reference)
    summary["grupos_origen_normales"] = len({row["session_id"] for row in novelty_reference})
    summary["prioridad_novedad_disponible"] = novelty_reason is None
    if novelty_reason:
        summary["motivo_novedad_desactivada"] = novelty_reason
    try:
        train, evaluation = split_for_validation(rows)
        summary["apto_para_evaluar"] = True
        summary["filas_entrenamiento_deduplicadas"] = len(train)
        summary["filas_evaluacion_deduplicadas"] = len(evaluation)
    except ValueError as exc:
        summary["apto_para_evaluar"] = False
        summary["motivo_no_apto"] = str(exc)
    summary["entrenamiento_automatico"] = False
    summary["no_concluyentes_entrenables"] = 0
    return summary


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=root / "data/incidents.sqlite3")
    args = parser.parse_args()
    if not args.db.is_file():
        parser.error(f"No existe {args.db}")
    print(json.dumps(audit(args.db), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
