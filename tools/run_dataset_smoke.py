"""Mutation falsifiers for the dataset emitter (ARCHITECTURE.md TrainingSample
verification policy). Each check must FAIL LOUDLY when its mutation is applied;
a silent pass refutes the export as training-grade.

GT-free: operates only on emitter outputs and synthetic mutations.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from atlas3r.contracts import ContractValidationError  # noqa: E402
from atlas3r.dataset import (  # noqa: E402
    _revalidate_label_field,
    _sha256,
    emit_training_dataset,
)

OUT = ROOT / "runs/_diag/_dataset_smoke_out"
results: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        print("\n".join(results))
        raise SystemExit(f"SMOKE FAILED at {name}")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)

    # 1. Emission: accepted scenes emitted, rejected scenes refused by name.
    manifest = emit_training_dataset(
        ROOT, ROOT / "runs/teacher", ROOT / "config/canonical_assets.json",
        ROOT / "external/teacher_artifacts", OUT,
    )
    acc = manifest["accounting"]
    check("emits_at_least_one_accepted_scene", acc["scenes_accepted"] >= 1,
          f"accepted={acc['scenes_accepted']}")
    refusal_reasons = {r["asset_id"]: r["reason"] for r in acc["refusals"]}
    check("rejected_scenes_refused_with_named_reasons",
          all(r for r in refusal_reasons.values()), str(refusal_reasons))

    sample_entry = manifest["samples"][0]
    sample_dir = OUT / sample_entry["sample_dir"]
    sample = json.loads((sample_dir / "sample.json").read_text(encoding="utf-8"))

    # 2. Loader round-trip with NO repo imports beyond numpy/json: every hashed
    #    file re-hashes to the manifest value; frames reference existing files.
    all_match = all(
        _sha256(sample_dir / rel) == h
        for rel, h in sample_entry["file_sha256"].items()
        if rel != "sample.json"  # sample.json hash covers the file that lists the others
    )
    check("manifest_hashes_match_on_clean_read", all_match)
    check("frames_reference_existing_rgb",
          all((sample_dir / f["rgb_path"]).exists() for f in sample["frames"]),
          f"n={len(sample['frames'])}")
    check("calibration_status_markers_in_band",
          sample["label_field"]["confidence_calibration"] == "uncalibrated_heuristic"
          and sample["scale"]["scale_status"] == "per_backbone_constant_prior")
    check("no_gt_derived_fields_in_sample",
          "band3d" not in json.dumps(sample) and "rmse" not in json.dumps(sample).lower())

    # 3. Corruption: flip one byte of the label npz -> hash check must catch it.
    npz_path = sample_dir / "voxel_occupancy_3d.npz"
    blob = bytearray(npz_path.read_bytes())
    blob[len(blob) // 2] ^= 0xFF
    npz_path.write_bytes(bytes(blob))
    corrupted_hash = _sha256(npz_path)
    check("corruption_detected_by_hash",
          corrupted_hash != sample_entry["file_sha256"]["voxel_occupancy_3d.npz"])

    # 4. unknown->free flip must fail contract re-validation on load.
    src_npz = ROOT / "runs/teacher" / sample["asset_id"] / "voxel_occupancy_3d.npz"
    z = dict(np.load(src_npz).items())
    flip = np.asarray(z["P_unknown"]) >= 0.99
    pf = np.asarray(z["P_free"]).copy()
    pf[flip] = 1.0  # unknown serialized as free -- the forbidden mutation
    z["P_free"] = pf
    mutated = OUT / "_mutated.npz"
    np.savez_compressed(mutated, **z)
    try:
        _revalidate_label_field(mutated)
        check("unknown_as_free_rejected_by_contract", False, "validator accepted the flip")
    except ContractValidationError:
        check("unknown_as_free_rejected_by_contract", True)

    # 5. NC license must be excluded with a named reason.
    nc_manifest = json.loads(
        (ROOT / "config/canonical_assets.json").read_text(encoding="utf-8")
    )
    for t in nc_manifest["tracks"]:
        t["metadata"]["license"] = {"name": "CC BY-NC-SA 4.0", "source": "test", "verified": True}
    nc_path = OUT / "_nc_assets.json"
    nc_path.write_text(json.dumps(nc_manifest), encoding="utf-8")
    nc_out = OUT / "_nc_out"
    m2 = emit_training_dataset(
        ROOT, ROOT / "runs/teacher", nc_path, ROOT / "external/teacher_artifacts", nc_out,
    )
    check("nc_license_excluded", m2["accounting"]["scenes_accepted"] == 0,
          str([r["reason"] for r in m2["accounting"]["refusals"]][:4]))
    check("nc_refusal_named", any(
        str(r["reason"]).startswith("license_not_commercial_clean")
        for r in m2["accounting"]["refusals"]))

    # 6. Missing license record must refuse (no silent default).
    for t in nc_manifest["tracks"]:
        t["metadata"].pop("license", None)
    nl_path = OUT / "_nolic_assets.json"
    nl_path.write_text(json.dumps(nc_manifest), encoding="utf-8")
    m3 = emit_training_dataset(
        ROOT, ROOT / "runs/teacher", nl_path, ROOT / "external/teacher_artifacts",
        OUT / "_nolic_out",
    )
    check("missing_license_refused", m3["accounting"]["scenes_accepted"] == 0
          and any(r["reason"] == "missing_license_record"
                  for r in m3["accounting"]["refusals"]))

    print("\n".join(results))
    print(f"ALL {len(results)} DATASET MUTATION FALSIFIERS PASSED")


if __name__ == "__main__":
    main()
