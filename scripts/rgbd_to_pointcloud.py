#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# creating one 3d frame

import os, re, glob, sys
from typing import Tuple, List, Dict
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    import open3d as o3d
    HAVE_O3D = True
except Exception:
    HAVE_O3D = False
    print("[WARN] open3d non dispo: on sauvegardera un PLY via fallback numpy.")

# ========== CONFIG ==========
ROOT_DIR = os.path.join("..","raw", "test")
COLOR_DIR = os.path.join(ROOT_DIR, "camera_color_image_raw")
DEPTH_DIR = os.path.join(ROOT_DIR, "camera_depth_image_raw")
COLOR_INFO_DIR = os.path.join(ROOT_DIR, "camera_color_camera_info")
DEPTH_INFO_DIR = os.path.join(ROOT_DIR, "camera_depth_camera_info")

# si plusieurs .txt, on prend le premier
N_FRAMES = 1           # commence simple: 1 paire
SUBSAMPLE = 4          # afficher 1 point sur 4 en scatter pour fluidité
OUT_PLY = "scene_rgbd_safe.ply"
# ===========================

TS_RE = re.compile(r"(\d{10,})$")

def pick_first_txt(folder: str) -> str:
    cands = sorted(glob.glob(os.path.join(folder, "*.txt")))
    return cands[0] if cands else None

def parse_cam_info_txt(path: str) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    def _arr(name, length=None):
        m = re.search(rf"\b{name}\s*:\s*\[([^\]]+)\]", txt)
        if not m: raise ValueError(f"{name} manquant dans {path}")
        arr = [float(x.strip()) for x in m.group(1).split(",")]
        if length and len(arr)!=length: raise ValueError(f"{name} len {len(arr)} != {length}")
        return arr
    def _int(name):
        m = re.search(rf"\b{name}\s*:\s*([0-9]+)", txt)
        if not m: raise ValueError(f"{name} manquant dans {path}")
        return int(m.group(1))
    K = np.array(_arr("K", 9), np.float32).reshape(3,3)
    D = np.array(_arr("D"), np.float32)
    w, h = _int("width"), _int("height")
    return K, D, (w, h)

def list_images(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.png")))

def ts_from_path(p: str) -> int:
    stem = os.path.splitext(os.path.basename(p))[0]
    m = TS_RE.search(stem)
    return int(m.group(1)) if m else -1

def pair_by_timestamp(colors: List[str], depths: List[str]) -> List[Tuple[str,str]]:
    m = {ts_from_path(d): d for d in depths}
    pairs = []
    for c in colors:
        t = ts_from_path(c)
        if t in m: pairs.append((c, m[t]))
    return pairs

def read_depth_as_meters(path: str) -> np.ndarray:
    depth_raw = cv.imread(path, cv.IMREAD_UNCHANGED)
    if depth_raw is None: raise FileNotFoundError(path)
    if depth_raw.ndim == 3:
        depth_raw = cv.cvtColor(depth_raw, cv.COLOR_BGR2GRAY)
    depth = depth_raw.astype(np.float32)
    finite = depth[np.isfinite(depth)]
    mx = float(np.nanmax(finite)) if finite.size else 0.0
    if mx > 50.0: depth /= 1000.0
    return depth

def undistort_color(img: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    if D is None or D.size == 0 or np.allclose(D, 0): return img
    return cv.undistort(img, K, D)

def backproject_numpy(depth_m: np.ndarray, K: np.ndarray, color_bgr: np.ndarray=None):
    h, w = depth_m.shape
    fx, fy = float(K[0,0]), float(K[1,1]); cx, cy = float(K[0,2]), float(K[1,2])
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    Z = depth_m
    valid = (Z > 0) & np.isfinite(Z)
    X = (us - cx) * Z / fx
    Y = (vs - cy) * Z / fy
    pts = np.stack([X[valid], Y[valid], Z[valid]], axis=1)  # (N,3)
    cols = None
    if color_bgr is not None and color_bgr.shape[:2] == depth_m.shape:
        color_rgb = cv.cvtColor(color_bgr, cv.COLOR_BGR2RGB)
        cols = color_rgb.reshape(-1,3)[valid.reshape(-1)]
    return pts, cols  # cols uint8

def save_ply_numpy(path: str, pts: np.ndarray, cols: np.ndarray=None):
    # PLY ASCII simple
    n = pts.shape[0]
    with open(path, "w") as f:
        if cols is None:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("end_header\n")
            for p in pts:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        else:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for p, c in zip(pts, cols):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")

def main():
    color_info = pick_first_txt(COLOR_INFO_DIR)
    depth_info = pick_first_txt(DEPTH_INFO_DIR)
    if not color_info or not depth_info:
        print("[ERR] camera_info manquants"); sys.exit(1)

    Kc, Dc, (wc, hc) = parse_cam_info_txt(color_info)
    Kd, Dd, (wd, hd) = parse_cam_info_txt(depth_info)
    print("[INFO] K_color:\n", Kc)
    print("[INFO] K_depth:\n", Kd)

    colors = list_images(COLOR_DIR)
    depths = list_images(DEPTH_DIR)
    if not colors or not depths:
        print("[ERR] images manquantes"); sys.exit(1)

    pairs = pair_by_timestamp(colors, depths)
    if not pairs:
        print("[WARN] pas de timestamps communs -> appairage par index")
        n = min(len(colors), len(depths))
        pairs = list(zip(colors[:n], depths[:n]))

    pairs = pairs[:max(1, N_FRAMES)]
    print(f"[INFO] 1ère paire: {os.path.basename(pairs[0][0])} <-> {os.path.basename(pairs[0][1])}")

    # === une seule frame pour debug ===
    ci, di = pairs[0]
    color = cv.imread(ci, cv.IMREAD_COLOR)
    depth = read_depth_as_meters(di)
    if depth.shape[:2] != color.shape[:2]:
        depth = cv.resize(depth, (color.shape[1], color.shape[0]), interpolation=cv.INTER_NEAREST)

    color_ud = undistort_color(color, Kc, Dc)

    d_valid = depth[(depth > 0) & np.isfinite(depth)]
    print(f"[INFO] depth min/max (m): {np.min(d_valid):.3f} / {np.max(d_valid):.3f}  (Nvalid={d_valid.size})")

    pts, cols = backproject_numpy(depth, Kd, color_ud)
    print(f"[INFO] points: {pts.shape[0]}")

    # Sauvegarde PLY
    if HAVE_O3D:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        if cols is not None:
            pcd.colors = o3d.utility.Vector3dVector((cols/255.0).astype(np.float32))
        o3d.io.write_point_cloud(OUT_PLY, pcd)
    else:
        save_ply_numpy(OUT_PLY, pts, cols)
    print(f"[OK] PLY écrit: {OUT_PLY}")

    # Scatter 3D rapide (downsample pour fluidité)
    if pts.shape[0] > 0:
        step = max(1, SUBSAMPLE)
        P = pts[::step]
        C = None if cols is None else (cols[::step] / 255.0)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        if C is not None:
            ax.scatter(P[:,0], P[:,1], P[:,2], s=1, c=C)
        else:
            ax.scatter(P[:,0], P[:,1], P[:,2], s=1)
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
        ax.set_title("RGB-D point cloud (single frame)")
        plt.tight_layout()
        plt.show()
    else:
        print("[WARN] aucun point valide à afficher.")

if __name__ == "__main__":
    main()
