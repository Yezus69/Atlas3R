# Branch Prune Suggestions

## Local Branches Deleted

- `codex/core-smgt-small-v2-measured-pseudo`
- `codex/core-smgt-tiny-generalization-gauntlet`
- `codex/core-smgt-tiny-student-map`
- `codex/overnight-realdata-tum-rgbd`
- `codex/phase5a-real-multiview-forge-temporal`
- `codex/phase5b-teacher-signal-forge-mapping-bridge`
- `codex/phase5b1-repo-slim-teacher-hardening`
- `codex/phase5c-external-teacher-runner-bootstrap`
- `codex/phase5d-teacher-weighted-temporal-mapping-training`
- `codex/phase5e-streaming-student-map-runtime`
- `codex/phase5f-real-depthpro-mixed-training-runtime`
- `codex/phase5g-multisequence-pose-odometry`
- `codex/phase5g1-multisequence-tum-generalization`
- `codex/phase5h-vggt-pose-pointmap-teacher`
- `codex/phase6a-product-slice-mapper-recording-mesh`
- `codex/phase6b-real-capture-incremental-mapper`
- `codex/phase6c-true-incremental-tsdf-backend`
- `codex/phase6d-sparse-block-tsdf-live-replay`
- `codex/phase6e-live-replay-scheduler`
- `codex/phase6f-live-mesh-chunks`
- `codex/phase6g-rgb-teacher-map`
- `codex/phase6h-teacher-stitch-cache`

## Local Branches Kept

- `codex/offline-world-builder-repo-reset`: current branch.
- `main`: protected branch.
- `codex/phase4d-tum-eval-v2-training`: kept because `git branch -d` refused
  deletion due to upstream merge state even though it is merged to `HEAD`.

## Remote Branches That Appear Stale

Remote branches were not deleted. Review before running any of these:

```bash
git push origin --delete codex/core-smgt-small-v2-measured-pseudo
git push origin --delete codex/core-smgt-tiny-generalization-gauntlet
git push origin --delete codex/core-smgt-tiny-student-map
git push origin --delete codex/overnight-realdata-tum-rgbd
git push origin --delete codex/phase4d-tum-eval-v2-training
git push origin --delete codex/phase5a-real-multiview-forge-temporal
git push origin --delete codex/phase5b-teacher-signal-forge-mapping-bridge
git push origin --delete codex/phase5b1-repo-slim-teacher-hardening
git push origin --delete codex/phase5c-external-teacher-runner-bootstrap
git push origin --delete codex/phase5d-teacher-weighted-temporal-mapping-training
git push origin --delete codex/phase5e-streaming-student-map-runtime
git push origin --delete codex/phase5f-real-depthpro-mixed-training-runtime
git push origin --delete codex/phase5g-multisequence-pose-odometry
git push origin --delete codex/phase5g1-multisequence-tum-generalization
git push origin --delete codex/phase5h-vggt-pose-pointmap-teacher
git push origin --delete codex/phase6a-product-slice-mapper-recording-mesh
git push origin --delete codex/phase6b-real-capture-incremental-mapper
git push origin --delete codex/phase6c-true-incremental-tsdf-backend
git push origin --delete codex/phase6d-sparse-block-tsdf-live-replay
git push origin --delete codex/phase6e-live-replay-scheduler
git push origin --delete codex/phase6f-live-mesh-chunks
git push origin --delete codex/phase6g-rgb-teacher-map
git push origin --delete codex/phase6h-teacher-stitch-cache
```
