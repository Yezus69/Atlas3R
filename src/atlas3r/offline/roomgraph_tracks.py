"""Real track extraction for RoomGraph optimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from atlas3r.offline.frame_cache import FrameCacheResult, FrameRecord
from atlas3r.offline.keyframes import KeyframeRecord
from atlas3r.offline.run_manifest import write_json

RoomGraphTrackSource = Literal["auto", "cotracker", "opencv_lk"]


@dataclass(frozen=True)
class TrackObservation:
    frame_id: int
    keyframe_id: int
    x_px: float
    y_px: float
    confidence: float


@dataclass(frozen=True)
class RoomGraphTrack:
    track_id: int
    observations: tuple[TrackObservation, ...]


@dataclass(frozen=True)
class TrackBuildResult:
    status: str
    source: str
    tracks: tuple[RoomGraphTrack, ...]
    summary_path: str
    tracks_npz_path: str
    reason: str | None = None

    @property
    def track_count(self) -> int:
        return len(self.tracks)

    @property
    def observation_count(self) -> int:
        return sum(len(track.observations) for track in self.tracks)


def build_roomgraph_tracks(
    run_dir: str | Path,
    *,
    frame_cache: FrameCacheResult,
    keyframes: tuple[KeyframeRecord, ...],
    usable_frame_ids: set[int],
    checkpoint_path: str | None,
    device: str,
    max_keyframes: int,
    grid_size: int,
    max_tracks: int,
    requested_source: RoomGraphTrackSource = "auto",
) -> TrackBuildResult:
    root = Path(run_dir)
    (root / "roomgraph").mkdir(parents=True, exist_ok=True)
    selected = _selected_records(frame_cache, keyframes, usable_frame_ids, max_keyframes)
    reason: str | None = None
    if len(selected) < 2:
        return _write_result(root, "unavailable", "none", (), "fewer than two usable frames")
    tracks: tuple[RoomGraphTrack, ...] = ()
    source = requested_source
    if requested_source in {"auto", "cotracker"}:
        source = "cotracker"
        try:
            tracks = _build_cotracker_tracks(
                root,
                selected=selected,
                checkpoint_path=checkpoint_path,
                device=device,
                grid_size=grid_size,
                max_tracks=max_tracks,
            )
        except Exception as exc:
            reason = f"CoTracker failed: {exc}"
            if requested_source == "cotracker":
                tracks = ()
            else:
                source = "opencv_lk"
                tracks = _build_opencv_tracks(root, selected=selected, max_tracks=max_tracks)
    else:
        source = "opencv_lk"
        tracks = _build_opencv_tracks(root, selected=selected, max_tracks=max_tracks)
    status = "available" if tracks else "unavailable"
    if not tracks and reason is None:
        reason = "no long tracks survived filtering"
    return _write_result(root, status, source if tracks else "none", tracks, reason)


def _selected_records(
    frame_cache: FrameCacheResult,
    keyframes: tuple[KeyframeRecord, ...],
    usable_frame_ids: set[int],
    max_keyframes: int,
) -> list[tuple[FrameRecord, KeyframeRecord]]:
    records = {record.frame_id: record for record in frame_cache.records}
    selected: list[tuple[FrameRecord, KeyframeRecord]] = []
    for keyframe in sorted(keyframes, key=lambda item: item.keyframe_id):
        if keyframe.frame_id not in usable_frame_ids:
            continue
        record = records.get(keyframe.frame_id)
        if record is None:
            continue
        selected.append((record, keyframe))
        if len(selected) >= max(2, max_keyframes):
            break
    return selected


def _build_cotracker_tracks(
    root: Path,
    *,
    selected: list[tuple[FrameRecord, KeyframeRecord]],
    checkpoint_path: str | None,
    device: str,
    grid_size: int,
    max_tracks: int,
) -> tuple[RoomGraphTrack, ...]:
    checkpoint = _default_checkpoint() if checkpoint_path is None else Path(checkpoint_path)
    if checkpoint is None or not checkpoint.is_file():
        raise FileNotFoundError("CoTracker checkpoint is missing")
    import cv2
    import torch
    from cotracker.predictor import CoTrackerPredictor  # type: ignore[import-untyped]

    target_w = 384
    frames: list[NDArray[np.uint8]] = []
    scale_xy: list[tuple[float, float]] = []
    for record, _ in selected:
        rgb = _read_rgb(root / record.frame_path, cv2)
        target_h = max(64, int(round(target_w * rgb.shape[0] / max(1, rgb.shape[1]))))
        resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frames.append(resized)
        scale_xy.append((record.width / float(target_w), record.height / float(target_h)))
    video_np = np.stack(frames, axis=0).astype(np.float32)
    video = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None]
    run_device = device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    predictor = CoTrackerPredictor(checkpoint=str(checkpoint), offline=True, window_len=60).to(
        run_device
    )
    predictor.eval()
    with torch.no_grad():
        tracks, visible = predictor(video.to(run_device), grid_size=max(2, grid_size))
    xy = tracks[0].detach().cpu().numpy().astype(np.float32)
    vis = visible[0].detach().cpu().numpy().astype(bool)
    return _tracks_from_arrays(xy, vis, selected, scale_xy, max_tracks=max_tracks)


def _build_opencv_tracks(
    root: Path,
    *,
    selected: list[tuple[FrameRecord, KeyframeRecord]],
    max_tracks: int,
) -> tuple[RoomGraphTrack, ...]:
    import cv2

    grays = [_read_gray(root / record.frame_path, cv2) for record, _ in selected]
    points = cv2.goodFeaturesToTrack(
        grays[0],
        maxCorners=max_tracks,
        qualityLevel=0.01,
        minDistance=18,
        blockSize=7,
    )
    if points is None:
        return ()
    active = points.reshape((-1, 2)).astype(np.float32)
    tracks: list[list[TrackObservation]] = [
        [_observation(selected[0], float(point[0]), float(point[1]), 1.0)] for point in active
    ]
    for index in range(1, len(grays)):
        next_points, status, err = cv2.calcOpticalFlowPyrLK(
            grays[index - 1],
            grays[index],
            active.reshape((-1, 1, 2)),
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_points is None or status is None:
            break
        next_flat = next_points.reshape((-1, 2)).astype(np.float32)
        status_flat = status.reshape((-1,)).astype(bool)
        err_flat = np.zeros((len(next_flat),), dtype=np.float32) if err is None else err.reshape(-1)
        for track_index, ok in enumerate(status_flat.tolist()):
            if not ok:
                continue
            x, y = next_flat[track_index]
            if x < 0 or y < 0 or x >= selected[index][0].width or y >= selected[index][0].height:
                continue
            confidence = float(1.0 / (1.0 + max(0.0, float(err_flat[track_index])) / 20.0))
            tracks[track_index].append(
                _observation(selected[index], float(x), float(y), confidence)
            )
        active = next_flat
    return _filter_tracks(tracks, max_tracks=max_tracks)


def _tracks_from_arrays(
    xy: NDArray[np.float32],
    visible: NDArray[np.bool_],
    selected: list[tuple[FrameRecord, KeyframeRecord]],
    scale_xy: list[tuple[float, float]],
    *,
    max_tracks: int,
) -> tuple[RoomGraphTrack, ...]:
    tracks: list[list[TrackObservation]] = []
    for track_index in range(xy.shape[1]):
        observations: list[TrackObservation] = []
        for frame_index in range(xy.shape[0]):
            if not bool(visible[frame_index, track_index]):
                continue
            sx, sy = scale_xy[frame_index]
            x = float(xy[frame_index, track_index, 0] * sx)
            y = float(xy[frame_index, track_index, 1] * sy)
            record = selected[frame_index][0]
            if x < 0 or y < 0 or x >= record.width or y >= record.height:
                continue
            observations.append(_observation(selected[frame_index], x, y, 1.0))
        tracks.append(observations)
    return _filter_tracks(tracks, max_tracks=max_tracks)


def _filter_tracks(
    tracks: list[list[TrackObservation]], *, max_tracks: int
) -> tuple[RoomGraphTrack, ...]:
    scored: list[tuple[float, list[TrackObservation]]] = []
    for observations in tracks:
        if len(observations) < 6:
            continue
        xy = np.asarray([(obs.x_px, obs.y_px) for obs in observations], dtype=np.float32)
        motion = float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0)))
        if motion < 6.0:
            continue
        scored.append((motion * len(observations), observations))
    scored.sort(key=lambda item: item[0], reverse=True)
    kept = scored[: max(1, max_tracks)]
    return tuple(
        RoomGraphTrack(track_id=index, observations=tuple(observations))
        for index, (_, observations) in enumerate(kept)
    )


def _observation(
    selected: tuple[FrameRecord, KeyframeRecord], x: float, y: float, confidence: float
) -> TrackObservation:
    record, keyframe = selected
    return TrackObservation(
        frame_id=record.frame_id,
        keyframe_id=keyframe.keyframe_id,
        x_px=x,
        y_px=y,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
    )


def _read_rgb(path: Path, cv2: Any) -> NDArray[np.uint8]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"could not read frame image: {path}")
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def _read_gray(path: Path, cv2: Any) -> NDArray[np.uint8]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"could not read frame image: {path}")
    return np.asarray(image, dtype=np.uint8)


def _default_checkpoint() -> Path | None:
    path = Path.home() / ".cache" / "atlas3r" / "cotracker" / "scaled_offline.pth"
    return path if path.is_file() else None


def _write_result(
    root: Path,
    status: str,
    source: str,
    tracks: tuple[RoomGraphTrack, ...],
    reason: str | None,
) -> TrackBuildResult:
    track_ids: list[int] = []
    frame_ids: list[int] = []
    keyframe_ids: list[int] = []
    xy: list[tuple[float, float]] = []
    confidence: list[float] = []
    lengths = []
    for track in tracks:
        lengths.append(len(track.observations))
        for obs in track.observations:
            track_ids.append(track.track_id)
            frame_ids.append(obs.frame_id)
            keyframe_ids.append(obs.keyframe_id)
            xy.append((obs.x_px, obs.y_px))
            confidence.append(obs.confidence)
    np.savez_compressed(
        root / "roomgraph" / "tracks.npz",
        track_ids=np.asarray(track_ids, dtype=np.int32),
        frame_ids=np.asarray(frame_ids, dtype=np.int32),
        keyframe_ids=np.asarray(keyframe_ids, dtype=np.int32),
        xy_px=np.asarray(xy, dtype=np.float32).reshape((-1, 2)),
        confidence=np.asarray(confidence, dtype=np.float32),
    )
    write_json(
        root / "roomgraph" / "track_summary.json",
        {
            "format_name": "atlas3r_roomgraph_tracks",
            "format_version": 1,
            "status": status,
            "track_source": source,
            "track_count": len(tracks),
            "observation_count": len(track_ids),
            "mean_observations_per_track": float(np.mean(lengths)) if lengths else 0.0,
            "reason": reason,
        },
    )
    return TrackBuildResult(
        status=status,
        source=source,
        tracks=tracks,
        summary_path="roomgraph/track_summary.json",
        tracks_npz_path="roomgraph/tracks.npz",
        reason=reason,
    )
