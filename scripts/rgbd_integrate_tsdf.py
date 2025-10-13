# scripts/rgbd_integrate_tsdf.py
import sys
import glob
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d


# -------------------- CONFIG --------------------
COLOR_DIR = Path("../raw/test/camera_color_image_raw")
DEPTH_DIR = Path("../raw/test/camera_depth_image_raw")
OUT_DIR   = Path("out")

# Camera intrinsics from your message
WIDTH, HEIGHT = 640, 400
K = np.array([[306.000244140625, 0.0,               318.4753112792969],
              [0.0,               306.1123352050781, 201.36949157714844],
              [0.0,               0.0,               1.0]], dtype=np.float64)

# Color distortion (OpenCV CALIB_RATIONAL_MODEL with 8 coeffs)
D_color = np.array([
    -1.2477725744247437, 0.8747861981391907,
    -9.421713184565306e-05, -0.00014916047803126276,
    -0.2381284087896347, -1.2307056188583374,
     0.8520383238792419,  -0.2296648770570755
], dtype=np.float64)

# Depth distortion all zeros -> no undistort for depth
D_depth = np.zeros(8, dtype=np.float64)

# If depth is in millimeters (uint16), keep 1000.0; if already in meters (float32), set 1.0
DEPTH_SCALE = 1000.0
DEPTH_TRUNC = 3.0       # max distance (m) integrated into TSDF (tune to your scene)
TSDF_VOXEL  = 0.005     # 5 mm (increase if scene is larger or depth is noisy)
TSDF_SDF_TRUNC = TSDF_VOXEL * 5.0

# Odometry flavor
USE_HYBRID_ODOM = True  # use Hybrid term if available
PRINT_EVERY = 20
# ------------------------------------------------


def sorted_images(folder: Path, exts=("*.png", "*.jpg", "*.jpeg")):
    files = []
    for e in exts:
        files += glob.glob(str(folder / e))
    files = sorted(files)
    return files


def to_o3d_intrinsics(K_mat: np.ndarray, width: int, height: int):
    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(width, height, float(K_mat[0, 0]), float(K_mat[1, 1]),
                        float(K_mat[0, 2]), float(K_mat[1, 2]))
    return intr


def rgbd_from_np(color_bgr_u8: np.ndarray, depth_img: np.ndarray,
                 depth_scale: float, depth_trunc: float) -> o3d.geometry.RGBDImage:
    color_rgb = cv2.cvtColor(color_bgr_u8, cv2.COLOR_BGR2RGB)
    color_o3d = o3d.geometry.Image(color_rgb)

    # Keep depth as uint16 (mm) or float32 (m); Open3D will use depth_scale appropriately
    if depth_img.dtype == np.uint16:
        depth_o3d = o3d.geometry.Image(depth_img)
    elif depth_img.dtype == np.float32:
        depth_o3d = o3d.geometry.Image(depth_img)
    else:
        # Best-effort cast to uint16
        depth_o3d = o3d.geometry.Image(depth_img.astype(np.uint16))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=depth_scale, depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )
    return rgbd


def main():
    assert COLOR_DIR.exists(), f"Missing {COLOR_DIR}"
    assert DEPTH_DIR.exists(), f"Missing {DEPTH_DIR}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    color_paths = sorted_images(COLOR_DIR)
    depth_paths = sorted_images(DEPTH_DIR)

    if len(color_paths) == 0 or len(depth_paths) == 0:
        print("No images found. Put files into the specified folders.")
        sys.exit(1)

    n = min(len(color_paths), len(depth_paths))
    print(f"[INFO] Paires candidates (appariage par index) : {n}")
    print(f"  ex: {Path(color_paths[0]).name}  <->  {Path(depth_paths[0]).name}")

    # Read one sample color to get actual size
    sample_color = cv2.imread(color_paths[0], cv2.IMREAD_COLOR)
    if sample_color is None:
        raise RuntimeError("Failed to read sample color image.")

    h, w = sample_color.shape[:2]
    if (w, h) != (WIDTH, HEIGHT):
        print(f"[WARN] Image size is {w}x{h}, expected {WIDTH}x{HEIGHT}. Using actual size.")

    # Compute optimal new camera matrix and undistort maps (fast remap each frame)
    K_prime, _ = cv2.getOptimalNewCameraMatrix(K, D_color, (w, h), alpha=0, newImgSize=(w, h))
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D_color, R=None, newCameraMatrix=K_prime, size=(w, h), m1type=cv2.CV_16SC2
    )

    intr_o3d = to_o3d_intrinsics(K_prime, w, h)

    # TSDF volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=TSDF_VOXEL,
        sdf_trunc=TSDF_SDF_TRUNC,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    # Odometry options (version-compatible)
    odo_option = o3d.pipelines.odometry.OdometryOption()
    # Helpful constraints for tabletop scenes
    try:
        odo_option.max_depth_diff = 0.07
        odo_option.min_depth = 0.1
        odo_option.max_depth = float(DEPTH_TRUNC)
    except Exception:
        pass

    if hasattr(o3d.pipelines.odometry, "RGBDOdometryJacobianFromHybridTerm") and USE_HYBRID_ODOM:
        odo_method = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
    else:
        odo_method = o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm()

    # Pose chain (world_T_cam)
    poses = []
    T_curr = np.eye(4, dtype=np.float64)

    prev_rgbd = None

    for i in range(n):
        color_path = color_paths[i]
        depth_path = depth_paths[i]

        color_bgr = cv2.imread(color_path, cv2.IMREAD_COLOR)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if color_bgr is None or depth is None:
            print(f"[WARN] Skip pair {i}: failed to read files.")
            continue

        # Undistort color with precomputed maps
        color_undist = cv2.remap(color_bgr, map1, map2, interpolation=cv2.INTER_LINEAR)

        # Ensure same size
        if depth.shape[:2] != color_undist.shape[:2]:
            depth = cv2.resize(depth, (color_undist.shape[1], color_undist.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

        rgbd = rgbd_from_np(color_undist, depth, DEPTH_SCALE, DEPTH_TRUNC)

        if prev_rgbd is None:
            poses.append(T_curr.copy())
            volume.integrate(rgbd, intr_o3d, np.linalg.inv(T_curr))
            prev_rgbd = rgbd
            continue

        # Frame-to-frame odometry (prev -> curr)
        try:
            success, T_delta, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                rgbd_source=prev_rgbd,
                rgbd_target=rgbd,
                pinhole_camera_intrinsic=intr_o3d,
                odo_init=np.eye(4),
                jacobian=odo_method,
                option=odo_option
            )
        except TypeError:
            # Older Open3D signatures may differ slightly; fallback without named args
            success, T_delta, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                prev_rgbd, rgbd, intr_o3d, np.eye(4), odo_method, odo_option
            )

        if not success:
            print(f"[WARN] Odometry failed at pair {i}. Re-using last pose.")
        else:
            # world_T_curr = world_T_prev * inv(T_delta)  (since T_delta maps curr->prev or prev->curr depending on API;
            # Open3D returns transformation from source to target. We used source=prev, target=curr => T_prev_to_curr.
            # We want world_T_curr = world_T_prev @ T_prev_to_curr
            T_curr = T_curr @ T_delta

        poses.append(T_curr.copy())
        volume.integrate(rgbd, intr_o3d, np.linalg.inv(T_curr))
        prev_rgbd = rgbd

        if (i + 1) % PRINT_EVERY == 0:
            print(f"[INFO] Integrated {i + 1}/{n} frames")

    print(f"[INFO] Integrated total frames: {len(poses)}")

    # Extract geometry
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    pcd = volume.extract_point_cloud()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mesh_path = OUT_DIR / "tsdf_mesh.ply"
    pcd_path  = OUT_DIR / "tsdf_points.ply"

    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    o3d.io.write_point_cloud(str(pcd_path), pcd)

    print(f"[OK] Saved mesh  → {mesh_path}")
    print(f"[OK] Saved cloud → {pcd_path}")

    print("[INFO] Visualizing...")
    o3d.visualization.draw_geometries([mesh])


if __name__ == "__main__":
    main()
