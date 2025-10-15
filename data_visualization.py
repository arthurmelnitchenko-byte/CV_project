import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# === Paramètres de la caméra (à ajuster si tu les connais) ===
# D: [-1.2477725744247437, 0.8747861981391907, -9.421713184565306e-05, -0.00014916047803126276, -0.2381284087896347, -1.2307056188583374, 0.8520383238792419, -0.2296648770570755]
K= np.array([[306.000244140625, 0.0, 318.4753112792969],[ 0.0, 306.1123352050781, 201.36949157714844],[ 0.0, 0.0, 1.0]])
# R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
# P: [306.000244140625, 0.0, 318.4753112792969, 0.0, 0.0, 306.1123352050781, 201.36949157714844, 0.0, 0.0, 0.0, 1.0, 0.0]
# binning_x: 0
# focal_length = 306
# principal_point = (318, 201)  # ex: centre de l'image
# K = np.array([[focal_length, 0, principal_point[0]],
#               [0, focal_length, principal_point[1]],
#               [0, 0, 1]])

# === Chemin vers dossier d'images ===
image_folder = "raw/test/camera_color_image_raw"

# Lister et trier les images
image_files = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

import cv2
import numpy as np
import cv2
import numpy as np


def get_matched_points(img1, img2):
    # Convertir les images en niveaux de gris
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Créer le détecteur SIFT
    sift = cv2.SIFT_create()

    # Détection et calcul des descripteurs
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Vérification des descripteurs
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return None, None

    # Création du matcher FLANN adapté aux descripteurs float32 de SIFT
    index_params = dict(algorithm=1, trees=5)  # KDTree pour SIFT
    search_params = dict(checks=50)           # Recherche plus rapide

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Matcher les descripteurs avec knn (k=2 pour ratio test de Lowe)
    matches = flann.knnMatch(des1, des2, k=2)

    # Appliquer le ratio test pour garder les bons matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8* n.distance:
            good_matches.append(m)

    # Vérification du nombre de bons matches
    if len(good_matches) < 10:
        return None, None

    # Extraire les points correspondants
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    return pts1, pts2


# Boucle sur les paires d'images
Rarray=[]
Tarray=[]
for i in range(len(image_files) - 1):
    img1 = cv2.imread(os.path.join(image_folder, image_files[i]))
    img2 = cv2.imread(os.path.join(image_folder, image_files[i + 1]))
    

    pts1, pts2 = get_matched_points(img1, img2)

    if pts1 is None or len(pts1) < 8:
        print(f"[!] Pas assez de points pour {image_files[i]} -> {image_files[i+1]}")
        continue

    # === Étape 1 : Trouver la matrice essentielle ===
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)

    if E is None:
        print(f"[!] Échec de findEssentialMat entre {image_files[i]} et {image_files[i+1]}")
        continue

    # === Étape 2 : Récupérer la pose (R, t) ===
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    # cv2.imshow("Image Viewer", img1)
    # key = cv2.waitKey(100)
    # if key == 27:  # Appuyer sur ÉCHAP pour quitter
    #     break
    Rarray.append(R)
    Tarray.append(t)

print("Rotation (R):\n array size:\n", Rarray[0], len(Rarray))
print("Translation (T):\n ", Tarray[0])


positions = [np.array([0, 0, 0])]  # position initiale

# Matrice de pose (cumulative) — identité au départ
pose = np.eye(4)

for R, t in zip(Rarray, Tarray):
    # Construire une matrice de transformation homogène 4x4
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()

    # Mettre à jour la pose globale (chaînage)
    pose = pose @ np.linalg.inv(T)  # Inverse car on récupère le mouvement inverse de la caméra

    # Extraire position caméra actuelle (colonne 4)
    cam_position = pose[:3, 3]
    positions.append(cam_position)

# === Visualiser en 3D ===
positions = np.array(positions)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o', color='blue', label="Trajectoire")

# Repère d’axe
ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='X')
ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='Y')
ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='Z')

ax.set_title("Trajectoire de la caméra")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()
cv2.destroyAllWindows()
