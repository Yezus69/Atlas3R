"""Overlap-aware keyframe selector (GT-free policy; pre-registered parameters).

Replaces fixed-count / evenly-spaced keyframe selection with a policy derived
from multi-view first principles, NOT from ground-truth tuning:

    A multi-view geometry backbone needs adjacent keyframes to share most of
    their field of view (co-observation is the evidence mass everything else
    depends on), while still accumulating baseline. Both are controlled by the
    apparent motion between keyframes. So: walk the video accumulating median
    optical-flow displacement, and emit a keyframe every time the accumulated
    displacement reaches a fixed budget.

Pre-registered parameters (frozen BEFORE any ground-truth evaluation; the
rationale is overlap geometry, and they must never be re-tuned against GT
metrics -- that would overfit the teacher to the test scenes and silently
break it on no-GT internet video, the exact failure the acceptance gate
exists to prevent):

    FLOW_BUDGET_WIDTHS = 0.25   accumulated median flow between keyframes,
                                in image widths. Displacement ~= lost overlap:
                                0.25 keeps ~75% shared field between adjacent
                                keyframes -- the middle of the 60-80% overlap
                                range standard multi-view practice targets.
    MIN_GAP_FRAMES     = 3      never select within a blur burst
    MAX_KEYFRAMES      = 48     backbone view-capacity ceiling. Provenance:
                                the repo's PRE-EXISTING measured finding that
                                MapAnything degrades beyond ~48 views (kf sweep,
                                docs/band_obstacle_recall_evidence.md Phase 5;
                                "do not blast to 100-300 views"). This is a
                                settled model property applied scene-blind --
                                NOT a parameter of this experiment.
    First and last frame are always included (full trajectory coverage).

The selector is deterministic, uses ONLY raw frames (sparse Lucas-Kanade flow
on a fixed grid at reduced resolution; cv2, already a dependency), and records
per-gap accumulated flow so the selection is auditable. Scenes whose flow
cannot be computed yield an explicit error status -- never a silent fallback
to even spacing.

``numpy``/``cv2`` are imported lazily inside functions (repo convention).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

FLOW_BUDGET_WIDTHS = 0.25
MIN_GAP_FRAMES = 3
MAX_KEYFRAMES = 48
PROC_WIDTH = 320
GRID_X, GRID_Y = 16, 12


def _natural_key(path: Path) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def frame_files_for(asset_id: str, root: Path) -> list[Path]:
    candidates = [root / "data" / asset_id / "rgb", root / "data" / asset_id]
    for directory in candidates:
        if not directory.is_dir():
            continue
        files = sorted(
            [p for p in directory.iterdir()
             if p.suffix.lower() in (".png", ".jpg", ".jpeg")],
            key=_natural_key,
        )
        if files:
            return files
    return []


def median_flow_per_step(files: Sequence[Path]) -> list[float]:
    """Median sparse-LK displacement between consecutive frames, in image
    widths. A failed pair contributes its predecessor's value (recorded), so a
    single unreadable frame cannot zero the accumulator."""
    import cv2  # type: ignore
    import numpy as np

    grid = np.array(
        [[(x + 0.5) / GRID_X, (y + 0.5) / GRID_Y]
         for y in range(GRID_Y) for x in range(GRID_X)],
        dtype=np.float32,
    )
    flows: list[float] = []
    prev = None
    prev_flow = 0.0
    for path in files:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            if prev is not None:
                flows.append(prev_flow)
            continue
        height = int(img.shape[0] * PROC_WIDTH / img.shape[1])
        img = cv2.resize(img, (PROC_WIDTH, height))
        if prev is not None:
            pts = (grid * np.array([PROC_WIDTH, height], dtype=np.float32)).reshape(-1, 1, 2)
            nxt, status, _err = cv2.calcOpticalFlowPyrLK(prev, img, pts, None)
            ok = status.reshape(-1).astype(bool)
            if int(ok.sum()) >= 10:
                disp = np.linalg.norm((nxt - pts).reshape(-1, 2)[ok], axis=1)
                prev_flow = float(np.median(disp)) / PROC_WIDTH
            # else: keep prev_flow (recorded implicitly by repetition)
            flows.append(prev_flow)
        prev = img
    return flows


def select_keyframes(files: Sequence[Path], anchor_ids: Sequence[int] = ()) -> dict[str, Any]:
    """Apply the pre-registered policy. Returns selection + audit trail.

    ``anchor_ids`` (comparability amendment, recorded): frame ids that MUST be
    included so the candidate shares >=3 frames with the measured M2 packets --
    without them the camera/band3d comparisons cannot compute at all (measured
    on the first selector run: every comparison returned
    insufficient_overlap_for_sim3_band_comparison and no GT value was ever
    observed, so this amendment is measurement plumbing, not GT tuning).
    Anchors count against the cap; policy frames nearest an anchor are dropped
    first when over budget. No-GT scenes pass no anchors -- the policy is pure
    there."""
    n = len(files)
    if n < 2:
        return {"status": "error_too_few_frames", "n_frames": n}
    flows = median_flow_per_step(files)
    if not flows or all(f == 0.0 for f in flows):
        return {"status": "error_no_flow_computed", "n_frames": n}

    selected = [0]
    accumulated = 0.0
    gap_audit = []
    for i, step in enumerate(flows, start=1):
        accumulated += step
        if accumulated >= FLOW_BUDGET_WIDTHS and (i - selected[-1]) >= MIN_GAP_FRAMES:
            selected.append(i)
            gap_audit.append(round(accumulated, 4))
            accumulated = 0.0
    if selected[-1] != n - 1:
        if (n - 1 - selected[-1]) >= MIN_GAP_FRAMES or len(selected) == 1:
            selected.append(n - 1)
            gap_audit.append(round(accumulated, 4))
        else:
            selected[-1] = n - 1
    anchors = sorted({int(a) for a in anchor_ids if 0 <= int(a) < n})
    if anchors:
        selected = sorted(set(selected) | set(anchors))
    over_cap = len(selected) > MAX_KEYFRAMES
    if over_cap:
        import numpy as np

        keep = set(anchors) | {0, n - 1}
        policy_only = [s for s in selected if s not in keep]
        # Drop policy frames nearest to a kept frame until under the cap.
        while len(keep) + len(policy_only) > MAX_KEYFRAMES and policy_only:
            kept_arr = np.asarray(sorted(keep))
            dists = [int(np.min(np.abs(kept_arr - s))) for s in policy_only]
            policy_only.pop(int(np.argmin(dists)))
        selected = sorted(keep | set(policy_only))

    return {
        "status": "selected",
        "policy": "accumulated_median_flow_budget",
        "params": {
            "flow_budget_widths": FLOW_BUDGET_WIDTHS,
            "min_gap_frames": MIN_GAP_FRAMES,
            "max_keyframes": MAX_KEYFRAMES,
            "proc_width": PROC_WIDTH,
        },
        "params_provenance": (
            "pre-registered from overlap geometry (75% shared field between "
            "adjacent keyframes); NEVER tuned against ground-truth metrics"
        ),
        "n_frames": n,
        "n_selected": len(selected),
        "anchor_ids_included": anchors,
        "capped_at_max": over_cap,
        "frame_ids": selected,
        "accumulated_flow_per_gap": gap_audit,
        "total_flow_widths": round(float(sum(flows)), 3),
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--anchor-ids", default="",
                        help="comma-separated frame ids that must be included (measured-comparison anchors)")
    args = parser.parse_args(argv)

    root = Path.cwd()
    files = frame_files_for(args.asset, root)
    if not files:
        print(json.dumps({"status": "error_no_frames", "asset": args.asset}))
        return 1
    anchors = [int(x) for x in args.anchor_ids.split(",") if x.strip()]
    report = select_keyframes(files, anchor_ids=anchors)
    report["asset_id"] = args.asset
    out = root / "runs/_diag" / f"keyframe_selection_{args.asset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    compact = {k: v for k, v in report.items() if k not in ("frame_ids", "accumulated_flow_per_gap")}
    print(json.dumps(compact, indent=2))
    print(f"[keyframes] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
