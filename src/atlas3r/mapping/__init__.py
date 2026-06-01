"""TSDF/surfel fusion, meshing, and map chunks."""

from atlas3r.mapping.cpu_tsdf import (
    TSDFCubeRoomSmokeResult,
    TSDFSurface,
    TSDFVolume,
    evaluate_surface_against_synthetic_cube_room,
    extract_tsdf_surface,
    integrate_synthetic_cube_room_scene,
    run_tsdf_cube_room_smoke,
    write_tsdf_cube_room_smoke,
)
from atlas3r.mapping.mesh_sidecar import (
    DEFAULT_MAX_SURFACE_SAMPLES,
    MESH_SIDECAR_FILENAME,
    MESH_SIDECAR_FORMAT_NAME,
    MESH_SIDECAR_FORMAT_VERSION,
    load_tsdf_surface_artifacts,
    load_tsdf_surface_mesh_sidecar,
    mesh_chunk_from_tsdf_surface,
    tsdf_surface_mesh_sidecar_record,
    write_tsdf_surface_mesh_sidecar,
    write_tsdf_surface_mesh_sidecar_from_artifacts,
)
from atlas3r.mapping.teacher_cache_replay import (
    TeacherCacheTSDFReplayFrame,
    TeacherCacheTSDFReplayResult,
    load_teacher_cache_tsdf_replay_frames,
    run_teacher_cache_tsdf_replay,
    write_teacher_cache_tsdf_replay,
)

__all__ = [
    "TSDFCubeRoomSmokeResult",
    "TSDFSurface",
    "TSDFVolume",
    "TeacherCacheTSDFReplayFrame",
    "TeacherCacheTSDFReplayResult",
    "DEFAULT_MAX_SURFACE_SAMPLES",
    "MESH_SIDECAR_FILENAME",
    "MESH_SIDECAR_FORMAT_NAME",
    "MESH_SIDECAR_FORMAT_VERSION",
    "evaluate_surface_against_synthetic_cube_room",
    "extract_tsdf_surface",
    "integrate_synthetic_cube_room_scene",
    "load_teacher_cache_tsdf_replay_frames",
    "load_tsdf_surface_artifacts",
    "load_tsdf_surface_mesh_sidecar",
    "mesh_chunk_from_tsdf_surface",
    "run_teacher_cache_tsdf_replay",
    "run_tsdf_cube_room_smoke",
    "tsdf_surface_mesh_sidecar_record",
    "write_teacher_cache_tsdf_replay",
    "write_tsdf_surface_mesh_sidecar",
    "write_tsdf_surface_mesh_sidecar_from_artifacts",
    "write_tsdf_cube_room_smoke",
]
