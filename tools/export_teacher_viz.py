"""Export a single self-contained HTML honesty viewer for one teacher asset.

Packs EVERY teacher output for the asset (3D voxel occupancy, 2D channels,
camera trajectory, surface point cloud, mesh, full teacher report) verbatim
into tools/teacher_viz_template.html as base64 typed arrays. Zero network
dependencies; the result opens from file://.

Honesty rules: exact values (float32 cast only), no subsampling, no curation;
untouched voxels are omitted from the sparse pack ONLY because they are exactly
the unknown prior, which is embedded and surfaced in the UI.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import numpy as np  # noqa: E402

VOXEL_CHANNELS = [
    "P_free", "P_occupied_static", "P_movable_static", "P_dynamic", "P_unknown",
    "map_confidence",
    "evidence_occupied_static_count", "evidence_movable_count",
    "evidence_dynamic_count", "evidence_free_count",
    "evidence_verified_occupied_count", "evidence_verified_movable_count",
]
CLASS_ORDER = ["P_free", "P_occupied_static", "P_movable_static", "P_dynamic", "P_unknown"]
GRID2D_CHANNELS = [
    "P_free", "P_occupied_static", "P_movable_static", "P_dynamic", "P_unknown",
    "height_min_m", "height_max_m",
]
PLY_TYPE_MAP = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
    "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_ply(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Generic binary_little_endian PLY parser. Returns {element: {prop: array}}.

    List properties (e.g. face vertex_indices) require a uniform list length
    and are returned as a 2D array under the property name.
    """
    raw = path.read_bytes()
    end = raw.find(b"end_header\n")
    if end < 0:
        raise ValueError(f"{path}: no end_header")
    header = raw[:end].decode("ascii", errors="replace").splitlines()
    body = memoryview(raw)[end + len(b"end_header\n"):]

    if not any(line.strip() == "format binary_little_endian 1.0" for line in header):
        raise ValueError(f"{path}: only binary_little_endian 1.0 is supported")

    elements: list[tuple[str, int, list[tuple]]] = []
    for line in header:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "element":
            elements.append((parts[1], int(parts[2]), []))
        elif parts[0] == "property" and elements:
            if parts[1] == "list":
                elements[-1][2].append(("list", PLY_TYPE_MAP[parts[2]], PLY_TYPE_MAP[parts[3]], parts[4]))
            else:
                elements[-1][2].append(("scalar", PLY_TYPE_MAP[parts[1]], parts[2]))

    out: dict[str, dict[str, np.ndarray]] = {}
    offset = 0
    for name, count, props in elements:
        if all(p[0] == "scalar" for p in props):
            dt = np.dtype([(p[2], "<" + p[1]) for p in props])
            arr = np.frombuffer(body, dtype=dt, count=count, offset=offset)
            offset += dt.itemsize * count
            out[name] = {p[2]: arr[p[2]] for p in props}
        elif len(props) == 1 and props[0][0] == "list":
            _, cnt_t, item_t, pname = props[0]
            cnt_dt = np.dtype("<" + cnt_t)
            item_dt = np.dtype("<" + item_t)
            first_n = int(np.frombuffer(body, dtype=cnt_dt, count=1, offset=offset)[0])
            dt = np.dtype([("n", "<" + cnt_t), ("v", "<" + item_t, (first_n,))])
            arr = np.frombuffer(body, dtype=dt, count=count, offset=offset)
            if not (arr["n"] == first_n).all():
                # non-uniform list lengths: slow generic fallback
                vals, off = [], offset
                for _ in range(count):
                    n = int(np.frombuffer(body, dtype=cnt_dt, count=1, offset=off)[0])
                    off += cnt_dt.itemsize
                    vals.append(np.frombuffer(body, dtype=item_dt, count=n, offset=off).copy())
                    off += item_dt.itemsize * n
                offset = off
                out[name] = {pname: np.array(vals, dtype=object)}
            else:
                offset += dt.itemsize * count
                out[name] = {pname: arr["v"]}
        else:
            raise ValueError(f"{path}: unsupported mixed scalar/list element '{name}'")
    return out


def pack_voxel(npz_path: Path, sizes: dict[str, int]) -> dict:
    z = np.load(npz_path)
    shape = z["P_free"].shape
    probs = np.stack([z[k] for k in CLASS_ORDER], axis=0)
    argmax = np.argmax(probs, axis=0).astype(np.uint8)  # 0..4 in CLASS_ORDER
    class_counts = [int((argmax == c).sum()) for c in range(5)]

    evidence_any = np.zeros(shape, dtype=bool)
    for k in VOXEL_CHANNELS[6:]:
        evidence_any |= z[k] > 0
    touched = evidence_any | (argmax != 4)
    flat_idx = np.flatnonzero(touched.reshape(-1)).astype(np.uint32)  # C-order linear

    untouched = ~touched
    untouched_prior: dict[str, float] = {}
    for k in VOXEL_CHANNELS:
        vals = z[k][untouched]
        if vals.size == 0:
            untouched_prior[k] = None
        else:
            uniq = np.unique(vals)
            if uniq.size != 1:
                print(f"  WARNING: untouched voxels not uniform in {k} "
                      f"({uniq.size} distinct values); embedding the first", file=sys.stderr)
            untouched_prior[k] = float(uniq[0])

    channels = {}
    nbytes = flat_idx.nbytes
    for k in VOXEL_CHANNELS:
        a = z[k].reshape(-1)[flat_idx].astype(np.float32)
        channels[k] = b64(a)
        nbytes += a.nbytes
    cls_sparse = argmax.reshape(-1)[flat_idx]
    nbytes += cls_sparse.nbytes
    sizes["voxel_sparse"] = nbytes

    return {
        "shape": [int(s) for s in shape],
        "origin_world": [float(v) for v in z["origin_world"]],
        "voxel_size_m": float(z["voxel_size_m"]),
        "floor_axis": int(z["floor_axis"]),
        "band_min_m": float(z["band_min_m"]),
        "band_max_m": float(z["band_max_m"]),
        "grid_frame": str(z["grid_frame"]),
        "acceptance_category": str(z["acceptance_category"]),
        "confidence_calibration": str(z["confidence_calibration"]),
        "scale_uncertainty": float(z["scale_uncertainty"]),
        "n_touched": int(flat_idx.size),
        "n_total": int(np.prod(shape)),
        "class_counts": class_counts,
        "untouched_prior": untouched_prior,
        "indices_b64": b64(flat_idx),
        "argmax_class_b64": b64(cls_sparse),
        "channels": channels,
    }


def pack_grid2d(npz_path: Path, sizes: dict[str, int]) -> dict:
    z = np.load(npz_path)
    shape = z["P_free"].shape
    channels = {}
    nbytes = 0
    for k in GRID2D_CHANNELS:
        a = z[k].astype(np.float32)
        channels[k] = b64(a)
        nbytes += a.nbytes
    sizes["grid2d_dense"] = nbytes
    return {
        "shape": [int(s) for s in shape],
        "origin_world": [float(v) for v in z["origin_world"]],
        "resolution_m": float(z["resolution_m"]),
        "grid_frame": str(z["grid_frame"]),
        "scale_uncertainty": float(z["scale_uncertainty"]),
        "acceptance_status_weight": float(z["acceptance_status_weight"]),
        "channels": channels,
    }


def pack_cloud(ply_path: Path, sizes: dict[str, int]) -> dict:
    v = parse_ply(ply_path)["vertex"]
    pos = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    nrm = np.stack([v["nx"], v["ny"], v["nz"]], axis=1).astype(np.float32)
    col = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8)
    sizes["cloud"] = pos.nbytes + nrm.nbytes + col.nbytes
    return {
        "count": int(pos.shape[0]),
        "positions_b64": b64(pos),
        "normals_b64": b64(nrm),
        "colors_b64": b64(col),
    }


def pack_mesh(ply_path: Path, sizes: dict[str, int]) -> dict:
    if not ply_path.exists():
        sizes["mesh"] = 0
        return {"present": False, "note": f"{ply_path} did not exist at export time"}
    elems = parse_ply(ply_path)
    v = elems["vertex"]
    pos = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    nrm = np.stack([v["nx"], v["ny"], v["nz"]], axis=1).astype(np.float32)
    faces = elems["face"]["vertex_indices"]
    if faces.dtype == object:
        raise ValueError(f"{ply_path}: non-triangular faces are not supported")
    idx = faces.astype(np.uint32)
    sizes["mesh"] = pos.nbytes + nrm.nbytes + idx.nbytes
    return {
        "present": True,
        "vertex_count": int(pos.shape[0]),
        "face_count": int(idx.shape[0]),
        "positions_b64": b64(pos),
        "normals_b64": b64(nrm),
        "indices_b64": b64(idx),
    }


def git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else f"unavailable ({r.stderr.strip()})"
    except Exception as exc:  # noqa: BLE001 - provenance must tolerate any git failure
        return f"unavailable ({exc})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", default="runs/teacher", help="teacher run directory")
    ap.add_argument("--asset", default="phone_room", help="asset id")
    ap.add_argument("--out", default="runs/viz/phone_room_viewer.html", help="output HTML path")
    args = ap.parse_args()

    run_dir = (ROOT / args.run_dir) if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    asset_dir = run_dir / args.asset
    out_path = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src = {
        "voxel": asset_dir / "voxel_occupancy_3d.npz",
        "grid2d": asset_dir / "occupancy_channels.npz",
        "trajectory": asset_dir / "camera_trajectory.json",
        "cloud": asset_dir / "surface_point_cloud.ply",
        "mesh": asset_dir / "mesh.ply",
        "report": run_dir / f"{args.asset}_teacher_report.json",
    }
    for name, p in src.items():
        if name != "mesh" and not p.exists():
            raise SystemExit(f"missing required input {name}: {p}")

    sizes: dict[str, int] = {}
    print(f"packing voxel grid   {src['voxel']}")
    voxel = pack_voxel(src["voxel"], sizes)
    print(f"packing 2D channels  {src['grid2d']}")
    grid2d = pack_grid2d(src["grid2d"], sizes)
    print(f"packing point cloud  {src['cloud']}")
    cloud = pack_cloud(src["cloud"], sizes)
    print(f"packing mesh         {src['mesh']}")
    mesh = pack_mesh(src["mesh"], sizes)

    trajectory = json.loads(src["trajectory"].read_text(encoding="utf-8"))
    report = json.loads(src["report"].read_text(encoding="utf-8"))
    sizes["trajectory_json"] = src["trajectory"].stat().st_size
    sizes["report_json"] = src["report"].stat().st_size

    provenance = []
    for name, p in src.items():
        if not p.exists():
            provenance.append({"path": str(p.resolve()), "sha256": "ABSENT",
                               "mtime": "ABSENT", "bytes": 0})
            continue
        st = p.stat()
        provenance.append({
            "path": str(p.resolve()),
            "sha256": sha256_file(p),
            "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            "bytes": st.st_size,
        })

    payload = {
        "meta": {
            "asset_id": args.asset,
            "export_timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "provenance": provenance,
        },
        "voxel": voxel,
        "grid2d": grid2d,
        "trajectory": trajectory,
        "cloud": cloud,
        "mesh": mesh,
        "report": report,
        "sizes": sizes,
    }

    template_path = ROOT / "tools" / "teacher_viz_template.html"
    template = template_path.read_text(encoding="utf-8")
    placeholder = "__ATLAS3R_PAYLOAD__"
    if template.count(placeholder) != 1:
        raise SystemExit(f"template must contain exactly one {placeholder}")
    # escape "</" so arbitrary report strings cannot close the inline <script> tag
    payload_json = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    html = template.replace(placeholder, payload_json)
    out_path.write_text(html, encoding="utf-8")

    total = out_path.stat().st_size
    print(f"\nwrote {out_path}")
    print(f"total HTML size: {total:,} bytes ({total/1e6:.1f} MB)")
    print("per-section raw binary/json bytes (base64 adds ~33% to binary sections):")
    for k, v in sizes.items():
        print(f"  {k:>16}: {v:,}")
    print(f"voxels touched: {voxel['n_touched']:,} / {voxel['n_total']:,}")
    print(f"class counts (free/occ/movable/dynamic/unknown): {voxel['class_counts']}")


if __name__ == "__main__":
    main()
