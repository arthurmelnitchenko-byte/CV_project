import open3d as o3d
import numpy as np
import glob
import pickle

# Datasets
color_files = sorted(glob.glob('sync_color_image/*.png'))
depth_files = sorted(glob.glob('sync_depth_image/*.png'))

# Load transformation matrices
with open('Transformations_reduced.pkl', 'rb') as file:
    T = pickle.load(file)

# Intrinsics
fx, fy, cx, cy = 306.000244140625, 306.1123352050781, 318.4753112792969, 201.36949157714844
intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 400, fx, fy, cx, cy)

def create_point_cloud(color_file, depth_file):
    """Generate a point cloud from a color and depth image pair."""
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.io.read_image(color_file),
        o3d.io.read_image(depth_file),
        depth_scale=1000.0,
        convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    return pcd

# Initialize point cloud (first point cloud)
merged_pt_cloud = create_point_cloud(color_files[0], depth_files[0])
previous_pt_cloud = merged_pt_cloud

# Loop over the first 5 images using the reconstructed trajectory
for i in range(1, 5):
    # Generate next point cloud
    new_pt_cloud = create_point_cloud(color_files[i], depth_files[i])
    points = np.asarray(new_pt_cloud.points)

    # Extract rotation and translation from transformation matrix
    R = T[i][:3, :3]
    t = T[i][:3, 3]

    # Apply transformation to all points at once
    points = (R @ points.T).T + t
    new_pt_cloud.points = o3d.utility.Vector3dVector(points)

    # Merge point clouds
    merged_pt_cloud += new_pt_cloud
    previous_pt_cloud = new_pt_cloud
    print(f"Image {i+1}/{len(color_files)} processed")

# Visualize the merged point cloud
o3d.visualization.draw_geometries([merged_pt_cloud])

# ICP-based alignment
threshold = 0.02
merged_pt_cloud_ICP = create_point_cloud(color_files[0], depth_files[0])
global_transform = np.eye(4)
previous_pt_cloud_ICP = merged_pt_cloud_ICP

# Loop over the first 5 images for ICP alignment
for i in range(1, 5):
    # Generate next point cloud
    new_pt_cloud = create_point_cloud(color_files[i], depth_files[i])

    # Perform ICP registration
    icp_registration = o3d.pipelines.registration.registration_icp(
        new_pt_cloud, previous_pt_cloud_ICP, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    # Update global transformation
    global_transform = global_transform @ icp_registration.transformation
    new_pt_cloud.transform(global_transform)

    # Merge point clouds
    merged_pt_cloud_ICP += new_pt_cloud
    previous_pt_cloud_ICP = new_pt_cloud
    print(f"Image {i+1}/{len(color_files)} processed")

# Visualize the ICP-merged point cloud
o3d.visualization.draw_geometries([merged_pt_cloud_ICP])
