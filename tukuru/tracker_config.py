"""Configuracion local de ByteTrack por duracion de oclusion."""

from __future__ import annotations

import os
import uuid
from pathlib import Path


def bytetrack_config(data_dir: Path, track_buffer_frames: int) -> str:
    """Escribe un YAML privado en data/ para no modificar los defaults del paquete."""
    if not 15 <= track_buffer_frames <= 240:
        raise ValueError("TRACK_BUFFER_FRAMES debe estar entre 15 y 240")
    directory = data_dir / "trackers"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"bytetrack_{track_buffer_frames}.yaml"
    if target.is_file():
        return str(target)
    temp = target.with_name(target.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    text = ("tracker_type: bytetrack\n"
            "track_high_thresh: 0.25\n"
            "track_low_thresh: 0.1\n"
            "new_track_thresh: 0.25\n"
            f"track_buffer: {track_buffer_frames}\n"
            "match_thresh: 0.8\n"
            "fuse_score: true\n")
    try:
        temp.write_text(text, encoding="utf-8")
        try:
            temp.replace(target)
        except OSError:
            if not target.is_file():
                raise
    finally:
        temp.unlink(missing_ok=True)
    return str(target)
