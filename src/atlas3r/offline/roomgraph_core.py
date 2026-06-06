"""RoomGraph least-squares optimization core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from atlas3r.contracts.coordinates import validate_T_A_B
from atlas3r.offline.disagreement import DisagreementResult
from atlas3r.offline.map_consistency_core import (
    MapConsistencyOptimizerOptions,
    fit_depth_scale_bias,
    resize_nearest,
    resize_nearest_bool,
)
from atlas3r.offline.proposal_cache import ProposalCacheResult
from atlas3r.offline.roomgraph_metrics import (
    RoomGraphMetrics,
    compute_roomgraph_metrics,
    improvement,
)
from atlas3r.offline.roomgraph_tracks import RoomGraphTrack

RoomGraphVariant = Literal["depth_only", "pose_only", "joint"]


@dataclass(frozen=True)
class RoomGraphFrame:
    frame_id: int
    keyframe_id: int
    K: NDArray[np.float32]
    R_world_camera: NDArray[np.float32]
    center_prior: NDArray[np.float32]
    depth_pro_m: NDArray[np.float32]
    depth_valid: NDArray[np.bool_]
    vggt_depth_m: NDArray[np.float32]
    vggt_valid: NDArray[np.bool_]
    base_depth_scale: float
    base_depth_bias_m: float


@dataclass(frozen=True)
class RoomGraphObservation:
    track_index: int
    frame_index: int
    xy_px: tuple[float, float]
    depth_pro_m: float
    vggt_depth_m: float
    confidence: float


@dataclass(frozen=True)
class RoomGraphProblem:
    frames: tuple[RoomGraphFrame, ...]
    observations: tuple[RoomGraphObservation, ...]
    initial_points_world_m: NDArray[np.float32]

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def point_count(self) -> int:
        return int(self.initial_points_world_m.shape[0])


@dataclass(frozen=True)
class RoomGraphVariantResult:
    variant: RoomGraphVariant
    status: str
    centers_world_m: NDArray[np.float32]
    depth_scale: NDArray[np.float32]
    depth_bias_m: NDArray[np.float32]
    points_world_m: NDArray[np.float32]
    focal_scale: float
    before_metrics: RoomGraphMetrics
    after_metrics: RoomGraphMetrics
    improvement: dict[str, float]
    cost_initial: float
    cost_final: float
    iterations: int
    reason: str | None = None


def build_roomgraph_problem(
    *,
    proposal_cache: ProposalCacheResult,
    disagreement: DisagreementResult | None,
    tracks: tuple[RoomGraphTrack, ...],
) -> RoomGraphProblem:
    cameras = _dedupe_by_frame(proposal_cache.vggt_camera_records)
    depth_pro = _dedupe_by_frame(proposal_cache.depth_pro_depth_records)
    depth_pro_cameras = _dedupe_by_frame(proposal_cache.depth_pro_camera_records)
    vggt_depth = _dedupe_by_frame(proposal_cache.vggt_depth_records)
    frame_ids = sorted(
        set(cameras)
        .intersection(depth_pro)
        .intersection(vggt_depth)
        .intersection({obs.frame_id for track in tracks for obs in track.observations})
    )
    frames: list[RoomGraphFrame] = []
    frame_index_by_id: dict[int, int] = {}
    fit_options = MapConsistencyOptimizerOptions(
        enabled=True,
        min_overlap_pixels=128,
        max_iterations=5,
        depth_scale_min=0.25,
        depth_scale_max=4.0,
        depth_bias_max_m=2.0,
    )
    for frame_id in frame_ids:
        camera = cameras[frame_id]
        dp_record = depth_pro[frame_id]
        vg_record = vggt_depth[frame_id]
        depth = proposal_cache.depth_arrays[str(dp_record["depth_key"])].astype(np.float32)
        valid = proposal_cache.depth_arrays[str(dp_record["valid_mask_key"])] > 0.0
        vg_depth = proposal_cache.depth_arrays[str(vg_record["depth_key"])].astype(np.float32)
        vg_valid = proposal_cache.depth_arrays[str(vg_record["valid_mask_key"])] > 0.0
        depth_shape = (int(depth.shape[0]), int(depth.shape[1]))
        vg_depth_resized = resize_nearest(vg_depth, depth_shape)
        vg_valid_resized = resize_nearest_bool(vg_valid, depth_shape)
        estimate = fit_depth_scale_bias(
            vg_depth_resized,
            depth,
            vg_valid_resized & valid,
            options=fit_options,
            frame_id=frame_id,
            keyframe_id=int(str(camera["keyframe_index"])),
        )
        T_world_camera = validate_T_A_B(camera["T_world_camera"], "T_world_camera")
        K = _frame_intrinsics(camera, depth_pro_cameras.get(frame_id), depth_shape)
        frame_index_by_id[frame_id] = len(frames)
        frames.append(
            RoomGraphFrame(
                frame_id=frame_id,
                keyframe_id=int(str(camera["keyframe_index"])),
                K=K,
                R_world_camera=T_world_camera[:3, :3].astype(np.float32),
                center_prior=T_world_camera[:3, 3].astype(np.float32),
                depth_pro_m=depth,
                depth_valid=valid,
                vggt_depth_m=vg_depth_resized,
                vggt_valid=vg_valid_resized,
                base_depth_scale=estimate.scale,
                base_depth_bias_m=estimate.bias_m,
            )
        )
    observations, points = _build_observations(frames, frame_index_by_id, tracks)
    return RoomGraphProblem(tuple(frames), tuple(observations), points)


def optimize_roomgraph_variant(
    problem: RoomGraphProblem,
    *,
    variant: RoomGraphVariant,
    max_iterations: int,
) -> RoomGraphVariantResult:
    if problem.frame_count < 2 or problem.point_count == 0 or not problem.observations:
        empty = _initial_state(problem)
        metrics = compute_roomgraph_metrics(problem, *empty)
        return RoomGraphVariantResult(
            variant=variant,
            status="unavailable",
            centers_world_m=empty[0],
            depth_scale=empty[1],
            depth_bias_m=empty[2],
            points_world_m=empty[3],
            focal_scale=empty[4],
            before_metrics=metrics,
            after_metrics=metrics,
            improvement={},
            cost_initial=0.0,
            cost_final=0.0,
            iterations=0,
            reason="insufficient tracks or frames",
        )
    from scipy.optimize import least_squares  # type: ignore[import-untyped]

    layout = _Layout(problem, variant)
    x0 = layout.pack(*_initial_state(problem))
    lower, upper = layout.bounds()
    before = compute_roomgraph_metrics(problem, *_initial_state(problem))
    result = least_squares(
        lambda value: _residuals(problem, layout, value),
        x0,
        bounds=(lower, upper),
        max_nfev=max(20, max_iterations),
        loss="soft_l1",
        f_scale=1.0,
        verbose=0,
    )
    centers, scales, biases, points, focal = layout.unpack(result.x)
    after = compute_roomgraph_metrics(problem, centers, scales, biases, points, focal)
    gains = improvement(before, after)
    status = "improved" if any(value >= 0.01 for value in gains.values()) else "no_improvement"
    return RoomGraphVariantResult(
        variant=variant,
        status=status,
        centers_world_m=centers,
        depth_scale=scales,
        depth_bias_m=biases,
        points_world_m=points,
        focal_scale=focal,
        before_metrics=before,
        after_metrics=after,
        improvement=gains,
        cost_initial=float(np.sum(_residuals(problem, layout, x0) ** 2)),
        cost_final=float(result.cost * 2.0),
        iterations=int(result.nfev),
    )


class _Layout:
    def __init__(self, problem: RoomGraphProblem, variant: RoomGraphVariant):
        self.problem = problem
        self.variant = variant
        self.optimize_pose = variant in {"pose_only", "joint"}
        self.optimize_depth = variant in {"depth_only", "joint"}
        self.optimize_points = variant in {"pose_only", "joint"}

    def pack(
        self,
        centers: NDArray[np.float32],
        depth_scale: NDArray[np.float32],
        depth_bias_m: NDArray[np.float32],
        points: NDArray[np.float32],
        focal_scale: float,
    ) -> NDArray[np.float64]:
        values: list[NDArray[np.float64]] = []
        if self.optimize_pose:
            values.append(np.asarray([_initial_log_scale_guess(self.problem)], dtype=np.float64))
            values.append(np.zeros((self.problem.frame_count, 3), dtype=np.float64).reshape(-1))
            values.append(np.zeros((1,), dtype=np.float64))
        if self.optimize_depth:
            values.append(np.zeros((self.problem.frame_count,), dtype=np.float64))
            values.append(np.zeros((self.problem.frame_count,), dtype=np.float64))
        if self.optimize_points:
            values.append(points.astype(np.float64).reshape(-1))
        return np.concatenate(values) if values else np.zeros((0,), dtype=np.float64)

    def unpack(
        self, value: NDArray[np.float64]
    ) -> tuple[
        NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], float
    ]:
        cursor = 0
        centers, scales, biases, points, focal = _initial_state(self.problem)
        if self.optimize_pose:
            log_traj = float(value[cursor])
            cursor += 1
            deltas = value[cursor : cursor + self.problem.frame_count * 3].reshape((-1, 3))
            cursor += self.problem.frame_count * 3
            log_focal = float(value[cursor])
            cursor += 1
            prior = np.stack([frame.center_prior for frame in self.problem.frames]).astype(
                np.float64
            )
            anchor = prior[0]
            centers = (anchor[None, :] + np.exp(log_traj) * (prior - anchor) + deltas).astype(
                np.float32
            )
            focal = float(np.exp(log_focal))
        if self.optimize_depth:
            alpha = value[cursor : cursor + self.problem.frame_count]
            cursor += self.problem.frame_count
            beta = value[cursor : cursor + self.problem.frame_count]
            cursor += self.problem.frame_count
            base_scale = np.asarray([frame.base_depth_scale for frame in self.problem.frames])
            base_bias = np.asarray([frame.base_depth_bias_m for frame in self.problem.frames])
            scales = (base_scale * np.exp(alpha)).astype(np.float32)
            biases = (base_bias + beta).astype(np.float32)
        if self.optimize_points:
            count = self.problem.point_count * 3
            points = value[cursor : cursor + count].reshape((-1, 3)).astype(np.float32)
        return centers, scales, biases, points, focal

    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower: list[NDArray[np.float64]] = []
        upper: list[NDArray[np.float64]] = []
        if self.optimize_pose:
            lower.extend(
                [
                    np.asarray([-1.5], dtype=np.float64),
                    np.full((self.problem.frame_count * 3,), -2.0, dtype=np.float64),
                    np.asarray([-0.2], dtype=np.float64),
                ]
            )
            upper.extend(
                [
                    np.asarray([2.0], dtype=np.float64),
                    np.full((self.problem.frame_count * 3,), 2.0, dtype=np.float64),
                    np.asarray([0.2], dtype=np.float64),
                ]
            )
        if self.optimize_depth:
            lower.extend(
                [
                    np.full((self.problem.frame_count,), -0.7, dtype=np.float64),
                    np.full((self.problem.frame_count,), -1.0, dtype=np.float64),
                ]
            )
            upper.extend(
                [
                    np.full((self.problem.frame_count,), 0.7, dtype=np.float64),
                    np.full((self.problem.frame_count,), 1.0, dtype=np.float64),
                ]
            )
        if self.optimize_points:
            lower.append(np.full((self.problem.point_count * 3,), -np.inf, dtype=np.float64))
            upper.append(np.full((self.problem.point_count * 3,), np.inf, dtype=np.float64))
        if not lower:
            return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
        return np.concatenate(lower), np.concatenate(upper)


def _residuals(
    problem: RoomGraphProblem, layout: _Layout, value: NDArray[np.float64]
) -> NDArray[np.float64]:
    centers, scales, biases, points, focal = layout.unpack(value)
    residuals: list[float] = []
    for obs in problem.observations:
        frame = problem.frames[obs.frame_index]
        pc = frame.R_world_camera.T @ (points[obs.track_index] - centers[obs.frame_index])
        weight = np.sqrt(max(0.05, obs.confidence))
        if pc[2] <= 1e-5 or not np.all(np.isfinite(pc)):
            residuals.extend([100.0, 100.0, 100.0])
            continue
        K = frame.K.copy()
        K[0, 0] *= focal
        K[1, 1] *= focal
        pixel = (K @ pc)[:2] / max(float(pc[2]), 1e-6)
        target = scales[obs.frame_index] * obs.depth_pro_m + biases[obs.frame_index]
        residuals.append(float(weight * (pixel[0] - obs.xy_px[0]) / 4.0))
        residuals.append(float(weight * (pixel[1] - obs.xy_px[1]) / 4.0))
        residuals.append(float(weight * (pc[2] - target) / 0.08))
    prior_centers = np.stack([frame.center_prior for frame in problem.frames]).astype(np.float32)
    residuals.extend(((centers - prior_centers).reshape(-1) / 3.0).astype(float).tolist())
    if problem.frame_count > 1:
        smooth = np.diff(centers, axis=0) - np.diff(prior_centers, axis=0)
        residuals.extend((smooth.reshape(-1) / 1.0).astype(float).tolist())
    base_scale = np.asarray([frame.base_depth_scale for frame in problem.frames], dtype=np.float32)
    base_bias = np.asarray([frame.base_depth_bias_m for frame in problem.frames], dtype=np.float32)
    residuals.extend(
        (np.log(np.maximum(scales, 1e-6) / np.maximum(base_scale, 1e-6)) / 0.3).tolist()
    )
    residuals.extend(((biases - base_bias) / 0.3).astype(float).tolist())
    residuals.extend(
        ((points - problem.initial_points_world_m).reshape(-1) / 0.02).astype(float).tolist()
    )
    residuals.append(float(np.log(max(focal, 1e-6)) / 0.08))
    return np.asarray(residuals, dtype=np.float64)


def _initial_state(
    problem: RoomGraphProblem,
) -> tuple[
    NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], float
]:
    centers = np.stack([frame.center_prior for frame in problem.frames]).astype(np.float32)
    scale = np.asarray([frame.base_depth_scale for frame in problem.frames], dtype=np.float32)
    bias = np.asarray([frame.base_depth_bias_m for frame in problem.frames], dtype=np.float32)
    return centers, scale, bias, problem.initial_points_world_m.astype(np.float32), 1.0


def _initial_log_scale_guess(problem: RoomGraphProblem) -> float:
    if problem.frame_count < 2 or problem.point_count == 0:
        return 0.0
    centers, scales, biases, points, focal = _initial_state(problem)
    prior = centers.astype(np.float32)
    anchor = prior[0]
    best_scale = 1.0
    best_error = float("inf")
    for scale in np.geomspace(0.25, 6.0, 25):
        trial_centers = (anchor[None, :] + float(scale) * (prior - anchor)).astype(np.float32)
        metrics = compute_roomgraph_metrics(problem, trial_centers, scales, biases, points, focal)
        if metrics.reprojection_error_mean_px < best_error:
            best_error = metrics.reprojection_error_mean_px
            best_scale = float(scale)
    return float(np.log(best_scale))


def _build_observations(
    frames: list[RoomGraphFrame],
    frame_index_by_id: dict[int, int],
    tracks: tuple[RoomGraphTrack, ...],
) -> tuple[tuple[RoomGraphObservation, ...], NDArray[np.float32]]:
    observations: list[RoomGraphObservation] = []
    point_inits: list[NDArray[np.float32]] = []
    for track in tracks:
        track_obs: list[RoomGraphObservation] = []
        lifted: list[NDArray[np.float32]] = []
        next_index = len(point_inits)
        for obs in track.observations:
            frame_index = frame_index_by_id.get(obs.frame_id)
            if frame_index is None:
                continue
            frame = frames[frame_index]
            dp = _sample(frame.depth_pro_m, obs.x_px, obs.y_px)
            if dp is None or not _sample_valid(frame.depth_valid, obs.x_px, obs.y_px):
                continue
            vg = _sample(frame.vggt_depth_m, obs.x_px, obs.y_px)
            if vg is None or not _sample_valid(frame.vggt_valid, obs.x_px, obs.y_px):
                continue
            depth = frame.base_depth_scale * dp + frame.base_depth_bias_m
            if depth <= 0.0:
                continue
            lifted.append(_lift(frame, obs.x_px, obs.y_px, depth))
            track_obs.append(
                RoomGraphObservation(
                    track_index=next_index,
                    frame_index=frame_index,
                    xy_px=(obs.x_px, obs.y_px),
                    depth_pro_m=float(dp),
                    vggt_depth_m=float(vg),
                    confidence=obs.confidence,
                )
            )
        if len(track_obs) < 3 or not lifted:
            continue
        point_inits.append(np.median(np.stack(lifted), axis=0).astype(np.float32))
        observations.extend(track_obs)
    if point_inits:
        points = np.stack(point_inits).astype(np.float32)
    else:
        points = np.zeros((0, 3), dtype=np.float32)
    return tuple(observations), points


def _lift(frame: RoomGraphFrame, x: float, y: float, depth: float) -> NDArray[np.float32]:
    fx = float(frame.K[0, 0])
    fy = float(frame.K[1, 1])
    cx = float(frame.K[0, 2])
    cy = float(frame.K[1, 2])
    point_camera = np.asarray([(x - cx) * depth / fx, (y - cy) * depth / fy, depth], np.float32)
    return cast(
        NDArray[np.float32],
        (frame.R_world_camera @ point_camera + frame.center_prior).astype(np.float32),
    )


def _sample(array: NDArray[np.float32], x: float, y: float) -> float | None:
    height, width = array.shape
    u = int(round(np.clip(x, 0, width - 1)))
    v = int(round(np.clip(y, 0, height - 1)))
    value = float(array[v, u])
    return value if np.isfinite(value) and value > 0.0 else None


def _sample_valid(mask: NDArray[np.bool_], x: float, y: float) -> bool:
    height, width = mask.shape
    u = int(round(np.clip(x, 0, width - 1)))
    v = int(round(np.clip(y, 0, height - 1)))
    return bool(mask[v, u])


def _frame_intrinsics(
    vggt_camera: dict[str, object],
    depth_pro_camera: dict[str, object] | None,
    target_shape: tuple[int, int],
) -> NDArray[np.float32]:
    if depth_pro_camera is not None and depth_pro_camera.get("K") is not None:
        K = np.asarray(depth_pro_camera["K"], dtype=np.float32)
        if K.shape == (3, 3):
            return K
    K = np.asarray(vggt_camera["K"], dtype=np.float32).copy()
    shape = vggt_camera.get("depth_shape")
    if isinstance(shape, list) and len(shape) == 2:
        src_h, src_w = int(str(shape[0])), int(str(shape[1]))
    else:
        src_h, src_w = target_shape
    dst_h, dst_w = target_shape
    K[0, :] *= float(dst_w) / max(1.0, float(src_w))
    K[1, :] *= float(dst_h) / max(1.0, float(src_h))
    K[2, :] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return K.astype(np.float32)


def _dedupe_by_frame(records: tuple[dict[str, object], ...]) -> dict[int, dict[str, object]]:
    deduped: dict[int, dict[str, object]] = {}
    for record in records:
        deduped[int(str(record["frame_id"]))] = record
    return deduped
