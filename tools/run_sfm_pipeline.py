"""Production SfM pipeline: video frames -> loop-closed COLMAP model + MVS
verified-depth workspaces (production + disjoint-half A/B for the
perturbation-stability tier).

One tool for the chain that was previously hand-run per scene (measured
across Phases 8-14, docs/band_obstacle_recall_evidence.md):

  1. stage every Nth frame UNION selector keyframes UNION measured anchors
     (sparse keyframes alone starve SfM -- Phase 10);
  2. GPU SIFT -> sequential matching WITH loop detection (COLMAP >= May 2025
     auto-downloads its faiss vocab tree; parity with exhaustive at O(N) --
     Phase 14: 0.1399 m vs 0.1480 m on the fr1_room loop);
  3. incremental mapper (pinned measured intrinsics, BA refine off -- the
     oracle recipe; or SIMPLE_RADIAL self-calibration for unknown cameras,
     BA refine on -- the internet-video path) -> largest model, TXT;
  4. undistort -> photometric PatchMatch for ALL staged frames (the reusable
     substrate) -> geometric pass for KEYFRAMES only (~2 s/keyframe marginal,
     identical verified coverage to a full run -- Phase 11);
  5. A/B workspaces from disjoint temporal source halves (Phase 12) ->
     geometric passes, feeding run_mvs_depth_backend --stability-*.

Downstream (unchanged): tools/run_colmap_pose_backend.py on
<work>/sparse_txt, then tools/run_mvs_depth_backend.py with
--dense-workspace <work>/dense --stability-workspace-a/-b <work>/dense_pa|pb.

Honest failures: every step checks its output and stops with the exact
failed command; fragmented reconstructions report ALL models and proceed on
the largest (frames outside it are dropped downstream, never fabricated).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLMAP = ROOT / "external/colmap_cuda/bin/colmap.exe"


def run(cmd: list[str], log: list[dict], step: str) -> None:
    t0 = time.perf_counter()
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    log.append({"step": step, "seconds": round(time.perf_counter() - t0, 1),
                "returncode": proc.returncode})
    if proc.returncode != 0:
        raise RuntimeError(
            f"step '{step}' failed (rc={proc.returncode}):\n"
            f"  command: {' '.join(str(c) for c in cmd)}\n"
            f"  stderr tail: {proc.stderr[-2000:]}"
        )


def stage_frames(frames: list[Path], keyframe_ids: list[int], anchor_ids: list[int],
                 divisor: int, imgs_dir: Path) -> list[str]:
    imgs_dir.mkdir(parents=True, exist_ok=True)
    ids = set(keyframe_ids) | set(anchor_ids) | set(range(0, len(frames), divisor))
    staged = []
    for i in sorted(ids):
        if i >= len(frames):
            continue
        src = frames[i]
        dst = imgs_dir / src.name
        if not dst.exists():
            shutil.copy(src, dst)
        staged.append(src.name)
    return staged


def write_keyframe_cfg(dense: Path, ref_names: list[str], all_names: list[str],
                       parity: int | None) -> None:
    """patch-match.cfg: keyframe references; sources __auto__ (production) or
    the disjoint temporal half of the 20 nearest staged frames (A/B)."""
    name_idx = {n: i for i, n in enumerate(all_names)}
    lines = []
    for nm in ref_names:
        if nm not in name_idx:
            continue
        lines.append(nm)
        if parity is None:
            lines.append("__auto__, 20")
        else:
            i = name_idx[nm]
            ranked = sorted(range(len(all_names)), key=lambda j: abs(j - i))[1:21]
            lines.append(", ".join(all_names[j] for k, j in enumerate(ranked) if k % 2 == parity))
    (dense / "stereo/patch-match.cfg").write_text("\n".join(lines) + "\n")


def stage_ab_workspace(dense: Path, variant_dir: Path) -> None:
    """A/B workspace = images + sparse model + the photometric substrate."""
    if variant_dir.exists():
        shutil.rmtree(variant_dir)
    for sub in ("stereo/depth_maps", "stereo/normal_maps", "stereo/consistency_graphs"):
        (variant_dir / sub).mkdir(parents=True)
    shutil.copytree(dense / "images", variant_dir / "images")
    shutil.copytree(dense / "sparse", variant_dir / "sparse")
    for kind in ("depth_maps", "normal_maps"):
        for f in (dense / f"stereo/{kind}").glob("*.photometric.bin"):
            shutil.copy(f, variant_dir / f"stereo/{kind}" / f.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--frame-glob", default="*.png")
    parser.add_argument("--keyframes", default=None,
                        help="selector json (default runs/_diag/keyframe_selection_<asset>.json)")
    parser.add_argument("--anchor-ids", default="",
                        help="comma-separated measured-anchor frame ids (band-comparison requirement)")
    parser.add_argument("--stage-divisor", type=int, default=8,
                        help="stage every Nth frame in addition to keyframes+anchors")
    parser.add_argument("--camera-mode", choices=("pinned", "self-calibrate"), required=True)
    parser.add_argument("--camera-params", default=None,
                        help="fx,fy,cx,cy -- required with --camera-mode pinned (measured intrinsics)")
    parser.add_argument("--work-dir", default=None,
                        help="default runs/_diag/colmap_work/<asset>_sfm")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--seq-overlap", type=int, default=15)
    args = parser.parse_args()

    if args.camera_mode == "pinned" and not args.camera_params:
        print(json.dumps({"status": "error_pinned_needs_camera_params"}))
        return 1

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    work = Path(args.work_dir) if args.work_dir else (
        Path("runs/_diag/colmap_work") / f"{args.asset_id}_sfm")
    if not work.is_absolute():
        work = ROOT / work
    frames = sorted((ROOT / args.frames_dir).glob(args.frame_glob),
                    key=lambda p: p.name)
    if not frames:
        print(json.dumps({"status": "error_no_frames", "frames_dir": args.frames_dir}))
        return 1

    kf_path = Path(args.keyframes) if args.keyframes else (
        ROOT / "runs/_diag" / f"keyframe_selection_{args.asset_id}.json")
    if not kf_path.exists():
        print(json.dumps({
            "status": "error_no_keyframe_selection",
            "next_command": f"python -m atlas3r.keyframes --asset {args.asset_id} "
                            f"--anchor-ids <measured ids>"}))
        return 1
    sel = json.loads(kf_path.read_text(encoding="utf-8"))
    keyframe_ids = [int(i) for i in sel["frame_ids"]]
    anchor_ids = [int(x) for x in args.anchor_ids.split(",") if x.strip()]

    log: list[dict] = []
    staged = stage_frames(frames, keyframe_ids, anchor_ids, args.stage_divisor, work / "imgs")
    ref_names = sorted({frames[i].name for i in keyframe_ids + anchor_ids if i < len(frames)})

    db = work / "db.db"
    extractor = [COLMAP, "feature_extractor", "--database_path", db,
                 "--image_path", work / "imgs", "--ImageReader.single_camera", "1",
                 "--FeatureExtraction.use_gpu", "1"]
    if args.camera_mode == "pinned":
        extractor += ["--ImageReader.camera_model", "PINHOLE",
                      "--ImageReader.camera_params", args.camera_params]
    else:
        extractor += ["--ImageReader.camera_model", "SIMPLE_RADIAL"]
    run(extractor, log, "feature_extractor")

    run([COLMAP, "sequential_matcher", "--database_path", db,
         "--SequentialMatching.overlap", args.seq_overlap,
         "--SequentialMatching.quadratic_overlap", "1",
         "--SequentialMatching.loop_detection", "1",
         "--FeatureMatching.use_gpu", "1"], log, "sequential_matcher_loop_detection")

    sparse = work / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    mapper = [COLMAP, "mapper", "--database_path", db,
              "--image_path", work / "imgs", "--output_path", sparse]
    if args.camera_mode == "pinned":
        # Oracle recipe: measured intrinsics are evidence, never re-fit.
        mapper += ["--Mapper.ba_refine_focal_length", "0",
                   "--Mapper.ba_refine_principal_point", "0",
                   "--Mapper.ba_refine_extra_params", "0"]
    run(mapper, log, "mapper")

    models = sorted([d for d in sparse.iterdir() if d.is_dir()],
                    key=lambda d: (d / "images.bin").stat().st_size if (d / "images.bin").exists() else 0,
                    reverse=True)
    if not models:
        raise RuntimeError("mapper produced no model -- video may lack parallax/texture")
    model = models[0]
    run([COLMAP, "model_converter", "--input_path", model, "--output_path", model,
         "--output_type", "TXT"], log, "model_converter")
    registered = sum(1 for line in (model / "images.txt").read_text().splitlines()
                     if line.strip() and not line.startswith("#") and line.split()[0].isdigit())

    dense = work / "dense"
    run([COLMAP, "image_undistorter", "--image_path", work / "imgs",
         "--input_path", model, "--output_path", dense,
         "--output_type", "COLMAP", "--max_image_size", "2000"], log, "image_undistorter")

    # Photometric substrate for ALL staged frames (sources need it), then
    # geometric verification for keyframes only.
    run([COLMAP, "patch_match_stereo", "--workspace_path", dense,
         "--workspace_format", "COLMAP", "--PatchMatchStereo.max_image_size", "2000",
         "--PatchMatchStereo.geom_consistency", "false"], log, "patch_match_photometric_all")
    undist_names = sorted(p.name for p in (dense / "images").glob("*"))
    write_keyframe_cfg(dense, ref_names, undist_names, parity=None)
    run([COLMAP, "patch_match_stereo", "--workspace_path", dense,
         "--workspace_format", "COLMAP", "--PatchMatchStereo.max_image_size", "2000",
         "--PatchMatchStereo.geom_consistency", "true"], log, "patch_match_geometric_keyframes")

    for variant, parity in (("dense_pa", 0), ("dense_pb", 1)):
        vdir = work / variant
        stage_ab_workspace(dense, vdir)
        write_keyframe_cfg(vdir, ref_names, undist_names, parity=parity)
        run([COLMAP, "patch_match_stereo", "--workspace_path", vdir,
             "--workspace_format", "COLMAP", "--PatchMatchStereo.max_image_size", "2000",
             "--PatchMatchStereo.geom_consistency", "true"], log, f"patch_match_{variant}")

    report = {
        "status": "ok",
        "asset_id": args.asset_id,
        "camera_mode": args.camera_mode,
        "staged_frames": len(staged),
        "keyframes": len(keyframe_ids),
        "registered_in_largest_model": registered,
        "model_count": len(models),
        "sparse_model": str(model.relative_to(ROOT)),
        "dense_workspace": str(dense.relative_to(ROOT)),
        "stability_workspaces": [str((work / v).relative_to(ROOT)) for v in ("dense_pa", "dense_pb")],
        "steps": log,
        "next_commands": [
            f"python tools/run_colmap_pose_backend.py --asset-id {args.asset_id} "
            f"--colmap-model {model.relative_to(ROOT).as_posix()}",
            f"python tools/run_mvs_depth_backend.py --asset-id {args.asset_id} "
            f"--dense-workspace {dense.relative_to(ROOT).as_posix()} "
            f"--stability-workspace-a {(work / 'dense_pa').relative_to(ROOT).as_posix()} "
            f"--stability-workspace-b {(work / 'dense_pb').relative_to(ROOT).as_posix()}"
            + (" --source-sparse-model " + model.relative_to(ROOT).as_posix()
               if args.camera_mode == "self-calibrate" else ""),
        ],
    }
    (work / "sfm_pipeline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "steps"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
