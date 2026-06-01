# 05 — Loss Functions

## Total objective

Use a weighted multi-task objective:

```text
L_total =
    w_depth   L_depth
  + w_normal  L_normal
  + w_point   L_pointmap
  + w_pose    L_pose
  + w_intr    L_intrinsics
  + w_match   L_matching
  + w_mv      L_multiview
  + w_map     L_map
  + w_obj     L_object
  + w_dyn     L_dynamic
  + w_unc     L_uncertainty
  + w_distill L_distillation
  + w_reg     L_regularization
```

Initial weights should be config-driven. Start conservative:

```yaml
loss_weights:
  depth: 1.0
  normal: 0.2
  pointmap: 1.0
  pose: 0.5
  intrinsics: 0.1
  matching: 0.2
  multiview: 0.5
  map: 0.2
  object: 0.5
  dynamic: 0.2
  uncertainty: 0.05
  distillation: 0.5
  regularization: 0.01
```

Tune after individual losses decrease on overfit tests.

## Depth loss

For metric ground truth depth `D` and prediction `D_hat`:

```text
L_depth_metric = mean_valid( Charbonnier(log(D_hat) - log(D)) )
```

Add SILog-style scale-sensitive term for metric datasets:

```text
e_i = log(D_hat_i) - log(D_i)
L_silog = mean(e_i^2) - lambda * mean(e_i)^2
```

For non-metric or teacher-only data, use scale-and-shift aligned inverse-depth loss:

```text
find a,b minimizing || a * invD_hat + b - invD_teacher ||
L_depth_aligned = robust_l1(a * invD_hat + b - invD_teacher)
```

Boundary-aware gradient loss:

```text
L_depth_grad = |∂x logD_hat - ∂x logD| + |∂y logD_hat - ∂y logD|
```

Mask invalid, sky, reflective, dynamic, and occluded pixels as appropriate.

## Normal loss

For unit normals:

```text
L_normal = mean_valid(1 - dot(n_hat, n_gt))
```

Add normal-depth consistency by deriving normals from predicted depth and matching the normal head.

## Pointmap loss

Prediction: `P_hat_world[t, y, x]`.

Metric ground truth:

```text
L_point_metric = mean_valid( Charbonnier(P_hat_world - P_world_gt) )
```

For up-to-scale datasets, align by Sim(3) before loss:

```text
S* = argmin_Sim3 || S(P_hat) - P_teacher ||
L_point_sim3 = mean_valid( robust_l1(S*(P_hat) - P_teacher) )
```

Local camera pointmap consistency:

```text
P_hat_cam = T_camera_world * P_hat_world
z(P_hat_cam) should match D_hat
ray(P_hat_cam) should match pixel ray from K
```

## Pose loss

Relative pose between frames `i,j`:

```text
T_ij_hat = inv(T_world_camera_i_hat) @ T_world_camera_j_hat
T_ij_gt  = inv(T_world_camera_i_gt)  @ T_world_camera_j_gt
```

Rotation geodesic:

```text
L_rot = acos(clamp((trace(R_err)-1)/2, -1, 1))
```

Translation:

```text
L_trans = robust_l1(t_ij_hat - t_ij_gt) / scene_scale
```

Combined:

```text
L_pose = alpha_rot * L_rot + alpha_trans * L_trans
```

For unmetric monocular sequences, use Sim(3)-aligned trajectory loss and do not pretend scale is measured.

Trajectory smoothness regularizer:

```text
L_accel = || log_SE3(T_{t-1}^{-1}T_t) - log_SE3(T_t^{-1}T_{t+1}) ||_1
```

Use a low weight; over-smoothing hurts sharp camera motion.

## Intrinsics and distortion loss

If GT intrinsics are known:

```text
L_focal = | log(fx_hat/fx_gt) | + | log(fy_hat/fy_gt) |
L_pp    = |cx_hat-cx_gt|/W + |cy_hat-cy_gt|/H
```

Distortion loss:

```text
L_distortion = robust_l1(k_hat - k_gt) + robust_l1(p_hat - p_gt)
```

If GT unknown, supervise from teachers and multi-view reprojection only.

## Dense matching loss

Given teacher/GT correspondences `u_i <-> u_j`:

```text
L_match_coord = robust_l1(flow_hat(u_i) - (u_j - u_i))
L_match_conf  = BCE(match_conf_hat, match_inlier)
```

Epipolar consistency:

```text
L_epi = | x_j^T F_ij x_i |
```

Use robust losses and ignore dynamic/occluded areas.

## Multi-view photometric and feature loss

Warp frame `j` into frame `i` with predicted depth and pose.

```text
L_photo = rho( I_i - warp(I_j, D_i, T_i, T_j, K_i, K_j) )
```

Use a mixture of L1 and SSIM-like loss if implemented safely. Also use DINO feature consistency:

```text
L_feat = 1 - cosine(F_i, warp(F_j))
```

Mask occlusions using z-buffer checks and dynamic masks.

## Map loss

For a clip, integrate predictions into a temporary TSDF/surfel map and render back.

Depth render loss:

```text
L_map_depth = robust_l1(D_hat_frame - D_rendered_from_map)
```

Silhouette loss:

```text
L_sil = BCE(valid_surface_render, valid_depth_mask)
```

TSDF consistency:

```text
L_tsdf = mean_samples( | SDF_hat(x_surface) | + max(0, margin - sign_consistency) )
```

Mesh/point cloud GT loss when available:

```text
L_chamfer = CD(samples(mesh_hat), samples(mesh_gt))
L_fscore_aux = soft F-score style thresholded occupancy loss
```

## Object losses

2D mask loss with Hungarian matching:

```text
L_mask = Dice(mask_hat, mask_gt) + BCE(mask_hat, mask_gt)
```

Track embedding contrastive loss:

```text
positive: same object across frames
negative: different object instances
L_embed = supervised_contrastive(object_embedding)
```

3D instance loss:

```text
L_obj3d = Dice(voxel_object_probs, voxel_object_gt) + CE(object_id_logits, object_id_gt)
```

Open-vocabulary distillation:

```text
L_text = 1 - cosine(object_embedding, text_or_DINO_teacher_embedding)
```

## Dynamic losses

Static/dynamic classification:

```text
L_dynamic_mask = focal_BCE(dynamic_hat, dynamic_gt_or_teacher)
```

Rigid object motion where GT/pseudo labels exist:

```text
L_obj_motion = SE3_geodesic(T_world_object_hat, T_world_object_teacher)
```

Static-map exclusion penalty:

```text
L_static_poison = penalty if dynamic pixels are integrated into static TSDF in differentiable map training
```

## Uncertainty loss

Use heteroscedastic negative log-likelihood for geometry:

```text
L_depth_nll = |D_hat-D| / sigma + log(sigma)
L_point_nll = ||P_hat-P||_1 / sigma_p + log(sigma_p)
L_pose_nll = r_pose^T Sigma_pose^{-1} r_pose + logdet(Sigma_pose)
```

Add calibration loss:

- bin predictions by confidence;
- compare predicted uncertainty with empirical error;
- penalize overconfident wrong predictions more than underconfident correct predictions.

## Distillation losses

Teacher distillation should be confidence-weighted.

```text
L_distill_depth = c_teacher * robust_l1(D_student - D_teacher)
L_distill_point = c_teacher * robust_l1(P_student - P_teacher)
L_distill_feat  = 1 - cosine(F_student_proj, F_teacher)
L_distill_mask  = Dice/BCE against SAM3 masks
```

Do not force the student to copy low-confidence teacher predictions. If teachers disagree, lower target confidence or train with a mixture distribution.

## Regularization

- edge-aware depth smoothness, masked at image edges;
- normal smoothness on planar regions;
- pose acceleration with low weight;
- memory-token dropout to prevent over-reliance;
- keyframe dropout;
- object-track dropout;
- scale-prior dropout so the network handles unknown intrinsics honestly.

