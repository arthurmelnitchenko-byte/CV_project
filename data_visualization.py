import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# === Paramètres de la caméra ===
K = np.array([
    [306.000244140625, 0.0, 318.4753112792969],
    [0.0, 306.1123352050781, 201.36949157714844],
    [0.0, 0.0, 1.0]
])

# === Chemin vers dossier d'images ===
image_folder = "raw/test/camera_color_image_raw"

# Lister et trier les images
image_files = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

def get_matched_points(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return None, None

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]

    if len(good_matches) < 10:
        return None, None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    return pts1, pts2

# === Calcul des poses entre images ===
Rarray = []
Tarray = []

for i in range(len(image_files) - 1):
    img1 = cv2.imread(os.path.join(image_folder, image_files[i]))
    img2 = cv2.imread(os.path.join(image_folder, image_files[i + 1]))

    pts1, pts2 = get_matched_points(img1, img2)
    if pts1 is None:
        print(f"[!] Pas assez de correspondances entre {image_files[i]} et {image_files[i+1]}")
        continue

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)
    if E is None:
        print(f"[!] Échec de findEssentialMat entre {image_files[i]} et {image_files[i+1]}")
        continue

    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    Rarray.append(R)
    Tarray.append(t)

# === Reconstruction de la trajectoire et direction ===
positions = [np.array([0, 0, 0])]
directions = [np.array([0, 0, 1])]  # Direction initiale = Z camera

pose = np.eye(4)

for R, t in zip(Rarray, Tarray):
    T_mat = np.eye(4)
    T_mat[:3, :3] = R
    T_mat[:3, 3] = t.ravel()

    # Mise à jour de la pose globale (chaînage)
    pose = T_mat @ pose

    cam_position = pose[:3, 3]
    cam_direction = pose[:3, 2]  # Axe Z de la caméra dans le monde

    positions.append(cam_position)
    directions.append(cam_direction)

# === Affichage matplotlib 3D avec directions ===
positions = np.array(positions)
directions = np.array(directions)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o', color='blue', label="Trajectoire")

# Ajouter les flèches de direction (axe Z caméra)
for pos, dir_vec in zip(positions, directions):
    ax.quiver(pos[0], pos[1], pos[2], dir_vec[0], dir_vec[1], dir_vec[2],
              length=4, normalize=True, color='orange')

# Repère global
ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='X')
ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='Y')
ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='Z')

ax.set_title("Trajectoire + orientation de la caméra")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.tight_layout()
plt.show()
cv2.destroyAllWindows()
