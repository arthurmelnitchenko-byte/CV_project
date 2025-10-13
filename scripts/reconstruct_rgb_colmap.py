import subprocess, shutil, sys
from pathlib import Path
import numpy as np
import open3d as o3d

IMAGES = Path("../raw/test/camera_color_image_raw")
WORK   = Path("work_rgb")
DB     = WORK/"colmap.db"
SPARSE = WORK/"sparse"
DENSE  = WORK/"dense"
PLY_OUT= WORK/"dense_fused.ply"
MESH_OUT = WORK/"mesh_poisson.ply"

CAMERA_MODEL = "PINHOLE"   # ou SIMPLE_RADIAL si photos smartphone sans EXIF
USE_GPU = "1"              # "0" pour CPU

def run(cmd): print("[RUN]", " ".join(cmd)); subprocess.check_call(cmd)

def ensure_clean(p: Path):
    if p.exists(): shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def main():
    assert IMAGES.exists() and any(IMAGES.iterdir()), f"Put images in {IMAGES}"
    ensure_clean(WORK); SPARSE.mkdir(exist_ok=True)

    # 1) Features
    run(["colmap","feature_extractor",
         "--database_path",str(DB),
         "--image_path",str(IMAGES),
         "--ImageReader.camera_model",CAMERA_MODEL,
         "--SiftExtraction.use_gpu",USE_GPU])

    # 2) Matching (sequential pour “tour autour”)
    run(["colmap","sequential_matcher",
         "--database_path",str(DB),
         "--SequentialMatching.overlap","5",
         "--SiftMatching.use_gpu",USE_GPU])

    # 3) Mapping (SfM)
    run(["colmap","mapper",
         "--database_path",str(DB),
         "--image_path",str(IMAGES),
         "--output_path",str(SPARSE)])

    # 4) Undistort pour MVS
    out_model = SPARSE/"0"
    ensure_clean(DENSE)
    run(["colmap","image_undistorter",
         "--image_path",str(IMAGES),
         "--input_path",str(out_model),
         "--output_path",str(DENSE),
         "--output_type","COLMAP"])

    # 5) Multi-view stereo (depth maps)
    run(["colmap","patch_match_stereo",
         "--workspace_path",str(DENSE),
         "--workspace_format","COLMAP",
         "--PatchMatchStereo.geom_consistency","true"])

    # 6) Fusion en nuage dense
    run(["colmap","stereo_fusion",
         "--workspace_path",str(DENSE),
         "--workspace_format","COLMAP",
         "--input_type","geometric",
         "--output_path",str(PLY_OUT)])

    # 7) Post-process Open3D (nettoyage + Poisson)
    pcd = o3d.io.read_point_cloud(str(PLY_OUT))
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.voxel_down_sample(0.002)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(50)

    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
    dens = np.asarray(dens); keep = dens > np.quantile(dens, 0.02)
    mesh = mesh.select_by_index(np.where(keep)[0])
    mesh.remove_degenerate_triangles(); mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices(); mesh.remove_non_manifold_edges()
    mesh = mesh.filter_smooth_simple(1); mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(str(MESH_OUT), mesh)
    print("[OK] Dense cloud:", PLY_OUT); print("[OK] Mesh:", MESH_OUT)
    o3d.visualization.draw_geometries([mesh])

if __name__ == "__main__":
    main()
