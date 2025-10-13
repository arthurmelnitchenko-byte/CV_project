import glob
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d

DEPTH_DIR = Path("../raw/test/camera_depth_image_raw")
OUT_DIR   = Path("out_depth")
DEPTH_SCALE = 1000.0   # 1000 si uint16 en mm; 1.0 si float32 en m
DEPTH_TRUNC = 3.0
VOXEL = 0.01           # 1 cm (augmente si très bruité)

def sorted_images(folder, exts=("*.png","*.jpg","*.tiff")):
    files=[]
    for e in exts: files+=glob.glob(str(folder/e))
    return sorted(files)

def depth_to_pcd(depth, intr: o3d.camera.PinholeCameraIntrinsic):
    # depth: np.ndarray (uint16 mm ou float32 m)
    if depth.dtype==np.uint16: dimg = o3d.geometry.Image(depth)
    else: dimg = o3d.geometry.Image(depth.astype(np.float32))
    fake = o3d.geometry.Image(np.zeros((*depth.shape,3),np.uint8)) # couleur noire
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        fake, dimg, depth_scale=DEPTH_SCALE, depth_trunc=DEPTH_TRUNC, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
    return pcd

def main():
    assert DEPTH_DIR.exists(), f"Missing {DEPTH_DIR}"
    paths = sorted_images(DEPTH_DIR)
    assert paths, "No depth images found."

    # Lire la première pour dimensions (focal approx si inconnue)
    d0 = cv2.imread(paths[0], cv2.IMREAD_UNCHANGED)
    h,w = d0.shape[:2]
    # intrinsics “raisonnables” si tu n’as pas K : fx=fy= 575 pour VGA approx; sinon adapte
    fx=fy= 575.0 * (w/640.0)
    cx,cy = (w-1)/2.0, (h-1)/2.0
    intr = o3d.camera.PinholeCameraIntrinsic(w,h,fx,fy,cx,cy)

    # TSDF
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL, sdf_trunc=VOXEL*5,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.None)

    # Odométrie via ICP (point-to-plane) sur PCD downsamplés
    pose = np.eye(4)
    p_prev = depth_to_pcd(d0, intr)
    p_prev.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL*6, max_nn=30))
    vol.integrate(
        o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.zeros((h,w,3),np.uint8)),
            o3d.geometry.Image(d0 if d0.dtype==np.uint16 else d0.astype(np.float32)),
            depth_scale=DEPTH_SCALE, depth_trunc=DEPTH_TRUNC, convert_rgb_to_intensity=False),
        intr, np.linalg.inv(pose))

    for i,pth in enumerate(paths[1:], start=1):
        d = cv2.imread(pth, cv2.IMREAD_UNCHANGED)
        if d is None: continue
        p_cur = depth_to_pcd(d, intr)
        p_cur.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL*6, max_nn=30))

        # ICP point-to-plane
        reg = o3d.pipelines.registration.registration_icp(
            p_cur.voxel_down_sample(VOXEL*2), p_prev.voxel_down_sample(VOXEL*2),
            max_correspondence_distance=VOXEL*5,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        pose = pose @ reg.transformation  # world_T_cur

        # Integrate
        vol.integrate(
            o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.zeros((h,w,3),np.uint8)),
                o3d.geometry.Image(d if d.dtype==np.uint16 else d.astype(np.float32)),
                depth_scale=DEPTH_SCALE, depth_trunc=DEPTH_TRUNC, convert_rgb_to_intensity=False),
            intr, np.linalg.inv(pose))

        p_prev = p_cur
        if i % 10 == 0: print(f"[INFO] Integrated {i+1}/{len(paths)}")

    mesh = vol.extract_triangle_mesh(); mesh.compute_vertex_normals()
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    o3d.io.write_triangle_mesh(str(OUT_DIR/"tsdf_depth_only_mesh.ply"), mesh)
    print("[OK] Saved:", OUT_DIR/"tsdf_depth_only_mesh.ply")
    o3d.visualization.draw_geometries([mesh])

if __name__ == "__main__":
    main()
