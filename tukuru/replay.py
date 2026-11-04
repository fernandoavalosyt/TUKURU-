"""Reproduce un video con la heuristica de Tukuru (sin camara, sin Gemini) para calibrar.

    python -m tukuru.replay video.mp4 --zones zonas.json --out salida.mp4

Imprime la linea de tiempo de eventos (alcance, ocultamiento, salida, cambios de nivel)
y, con --out, escribe el video anotado con zonas, esqueleto y nivel de cada persona.
Con --gemini cada posible ocultamiento se revisa de verdad con Gemini (el clip de recortes se envia
a la API con la clave de .env; --save-clips guarda copia de lo enviado). Sin --gemini decide solo el
respaldo local.
Con --dashboard el video sale como el panel web: video en vivo, tabla de personas y alertas
con evidencia (se usa el AlertManager real, en una carpeta temporal).
`zonas.json` tiene el mismo formato que el campo "Zonas" del panel.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from dotenv import load_dotenv

from .alerts import AlertManager
from .clips import select_clip, store_crop
from .config import ROOT, Settings, _validate_zones
from .dashboard import THUMB, render_dashboard
from .decision import apply_verdict, decide
from .gemini import GeminiJudge
from .heuristics import TrackState, update_track
from .optimizer import FrameOptimizer
from .render import draw_tracks
from .vision import PoseTracker, ensure_model
from .zones import ZoneSet


def run(video: Path, zones: Optional[dict], fps: float, model: str, out: Optional[Path], settings: Settings,
        max_seconds: Optional[float] = None, dashboard: bool = False, use_gemini: bool = False,
        save_clips: Optional[Path] = None, log=print, object_model: Optional[Path] = None,
        object_report: Optional[Path] = None) -> List[dict]:
    experiment = None
    if object_model is not None:
        from .objects import ObjectExperiment, SmallDetector

        experiment = ObjectExperiment(SmallDetector(object_model, settings.device))
    elif object_report is not None:
        raise ValueError("--object-report requiere --object-model")
    with tempfile.TemporaryDirectory(prefix="tukuru-replay-") as tmp:
        # En el replay el tiempo avanza mas rapido que el reloj real: sin cooldown entre consultas.
        replay_settings = replace(settings, data_dir=Path(tmp), shadow_mode=False, alert_webhook_url=None,
                                  gemini_cooldown=0.0, gemini_enabled=use_gemini and settings.gemini_enabled)
        timeline = _run(video, zones, fps, model, out, replay_settings, max_seconds, dashboard, use_gemini,
                        save_clips, log, experiment)
    if experiment is not None:
        report = experiment.report()
        report.update(video=str(video), model=str(object_model), timeline=timeline)
        if object_report is not None:
            object_report.parent.mkdir(parents=True, exist_ok=True)
            object_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"-- Objetos: {experiment.calls} inferencias, {experiment.seconds:.2f} s adicionales de detector")
        for event in experiment.events:
            log(f"   {event['t']:.2f}s persona {event['id']} mano {event['hand']}: "
                f"{event['label']} -> {event['state']}")
    return timeline


def _review(judge: GeminiJudge, ts: TrackState, now: float, settings: Settings, save_clips: Optional[Path],
            timeline: List[dict], log) -> None:
    """Pide el dictamen de Gemini sobre el ultimo ocultamiento y espera la respuesta (el replay no tiene prisa)."""
    event = ts.review_event
    if event is None or now < ts.review_due:
        return
    frames = select_clip(ts, now, event, settings.clip_seconds, settings.gemini_frames)
    if not frames:
        if now - event > 45.0:
            ts.review_event = None
        return
    ts.review_event = None
    context = {**ts.summary(now), "visible_seg": round(now - ts.first_seen)}
    done, box = threading.Event(), {}

    def callback(key, verdict, error):
        box["verdict"], box["error"] = verdict, error
        done.set()

    key = f"replay-{ts.track_id}-{event:.1f}"
    if not judge.review(key, frames, context, callback):
        log(f"   (Gemini no acepto la consulta de la persona {ts.track_id} a los {event:.1f} s)")
        return
    done.wait(120)
    verdict = box.get("verdict")
    if save_clips is not None:
        save_clips.mkdir(parents=True, exist_ok=True)
        for n, (age, jpeg) in enumerate(frames):
            (save_clips / f"p{ts.track_id}_{event:05.1f}s_{n:02d}.jpg").write_bytes(jpeg)
        (save_clips / f"p{ts.track_id}_{event:05.1f}s_dictamen.json").write_text(
            json.dumps({"contexto": context, "dictamen": verdict, "error": box.get("error")}, ensure_ascii=False,
                       indent=2), encoding="utf-8")
    if verdict is None:
        timeline.append({"t": round(now, 2), "id": ts.track_id, "kind": "gemini", "detail": f"error: {box.get('error')}",
                         "score": round(ts.score, 2)})
        return
    apply_verdict(ts, event, verdict, now, settings)
    timeline.append({"t": round(now, 2), "id": ts.track_id, "kind": "gemini",
                     "detail": f"{verdict['state']} {verdict['confidence']:.2f} {verdict['action']}",
                     "score": round(ts.score, 2), "reason": verdict["reasoning"]})


def _run(video: Path, zones: Optional[dict], fps: float, model: str, out: Optional[Path], settings: Settings,
         max_seconds: Optional[float], dashboard: bool, use_gemini: bool, save_clips: Optional[Path],
         log, experiment=None) -> List[dict]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"No se pudo abrir {video}")
    src_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(src_fps / fps))
    eff_fps = src_fps / stride
    tracker = PoseTracker(ensure_model(settings.models_dir, model), settings.device, settings.yolo_conf)
    optimizer = FrameOptimizer(settings.max_side)
    alerts = AlertManager(settings)
    judge = GeminiJudge(settings) if use_gemini else None
    if judge is not None and judge.disabled_reason:
        log(f"-- Gemini no disponible ({judge.disabled_reason}): decide el respaldo local")
        judge = None
    thumbs: Dict[str, np.ndarray] = {}
    zone_set: Optional[ZoneSet] = None
    tracks: Dict[int, TrackState] = {}
    writer = None
    timeline: List[dict] = []
    last_level: Dict[int, str] = {}
    seen_events: Dict[int, int] = {}
    index, started = 0, time.time()

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        index += 1
        if (index - 1) % stride:
            continue
        now = (index - 1) / src_fps
        if max_seconds and now > max_seconds:
            break
        small, scale = optimizer.process(frame)
        h, w = small.shape[:2]
        if zone_set is None:
            zone_set = ZoneSet(zones, w, h)
        people = [p for p in tracker.infer(small, settings.max_side) if (p.bbox[3] - p.bbox[1]) >= 0.10 * h]
        # Actualizar todas las poses antes de asociar objetos: permite rechazar propiedad ambigua.
        for person in people:
            ts = tracks.setdefault(person.track_id, TrackState(person.track_id, "replay", now, now))
            update_track(ts, person, zone_set, now, settings)
        if experiment is not None:
            experiment.step(frame, scale, list(tracks.values()), zone_set, now)
        for person in people:
            ts = tracks[person.track_id]
            if judge is not None:
                store_crop(ts, frame, scale, now, settings.gemini_crop_side,
                           optimizer.enhance if optimizer.is_dark else None)
                _review(judge, ts, now, settings, save_clips, timeline, log)
            else:
                ts.review_event = None                               # sin Gemini: decide el respaldo local
            alerts.evaluate(ts, decide(ts, now, settings), now, video.name,
                            lambda: draw_tracks(small, list(tracks.values()), video.name, eff_fps, zone_set))
            if ts.level != last_level.get(ts.track_id, "ok"):
                timeline.append({"t": round(now, 2), "id": ts.track_id, "kind": "nivel", "detail": ts.level,
                                 "score": round(ts.score, 2), "reason": ts.level_reason})
                last_level[ts.track_id] = ts.level
            for t, kind, weight in list(ts.events)[seen_events.get(ts.track_id, 0):]:
                timeline.append({"t": round(t, 2), "id": ts.track_id, "kind": kind, "detail": ts.conceal_place
                                 if kind == "ocultamiento" else "", "score": round(ts.score, 2)})
            seen_events[ts.track_id] = len(ts.events)
        if out is not None:
            visible = [t for t in tracks.values() if now - t.last_seen < 0.5]
            shown = draw_tracks(small, visible, video.name, eff_fps, zone_set)
            if experiment is not None:
                shown = experiment.draw(shown)
            if dashboard:
                records = alerts.recent(200)[::-1]
                for record in records:
                    if record["snapshot"] and record["id"] not in thumbs:
                        image = cv2.imread(str(alerts.dir / record["snapshot"]))
                        if image is not None:
                            thumbs[record["id"]] = cv2.resize(image, THUMB, interpolation=cv2.INTER_AREA)
                if judge is not None:
                    stats = judge.health()
                    gemini_text = f"Gemini: activo ({stats['calls']} consultas, {stats['tokens']} tokens)"
                else:
                    gemini_text = "Gemini: apagado (replay) - solo heuristica local"
                shown = render_dashboard(shown, tracks.values(), records, thumbs, now, video.name, eff_fps,
                                         gemini_text)
            else:
                cv2.putText(shown, f"t={now:5.1f}s", (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                            cv2.LINE_AA)
            if writer is None:
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), eff_fps,
                                         (shown.shape[1], shown.shape[0]))
            writer.write(shown)
    capture.release()
    if writer is not None:
        writer.release()
    if judge is not None:
        judge.shutdown()
    timeline.sort(key=lambda e: e["t"])
    for event in timeline:
        log(f"{event['t']:6.2f}s  persona {event['id']:<3} {event['kind']:<13} {event['detail']:<22} "
            f"punt={event['score']:.2f} {event.get('reason', '')}")
    log(f"-- {index} cuadros leidos en {time.time() - started:.0f} s ({eff_fps:.1f} fps de analisis)")
    return timeline


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce un video con la heuristica de Tukuru")
    parser.add_argument("video", type=Path)
    parser.add_argument("--zones", help="archivo JSON con las zonas (formato del panel)")
    parser.add_argument("--fps", type=float, default=8.0, help="cuadros por segundo de analisis")
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--out", type=Path, help="video anotado de salida (.mp4)")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--dashboard", action="store_true", help="el video sale como el panel web")
    parser.add_argument("--gemini", action="store_true", help="revisar los ocultamientos con Gemini (envia recortes)")
    parser.add_argument("--save-clips", type=Path, help="carpeta donde guardar lo enviado a Gemini y su dictamen")
    parser.add_argument("--object-model", type=Path, help="detector local opcional (.pt); solo observacion")
    parser.add_argument("--object-report", type=Path, help="JSON con detecciones, relaciones y tiempo del detector")
    args = parser.parse_args(argv)
    if args.object_report and not args.object_model:
        parser.error("--object-report requiere --object-model")
    if args.object_model and not args.object_model.is_file():
        parser.error(f"No existe el detector local: {args.object_model}")
    zones = _validate_zones(json.loads(Path(args.zones).read_text(encoding="utf-8"))) if args.zones else None
    load_dotenv(ROOT / ".env", override=False)
    settings = Settings.from_env()
    run(args.video, zones, args.fps, args.model, args.out, settings, args.max_seconds, args.dashboard, args.gemini,
        args.save_clips, object_model=args.object_model, object_report=args.object_report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
