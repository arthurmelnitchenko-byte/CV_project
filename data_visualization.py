import cv2
import os
import numpy as np

# === Paramètres de la caméra (à ajuster si tu les connais) ===
# D: [-1.2477725744247437, 0.8747861981391907, -9.421713184565306e-05, -0.00014916047803126276, -0.2381284087896347, -1.2307056188583374, 0.8520383238792419, -0.2296648770570755]
# K: [306.000244140625, 0.0, 318.4753112792969, 0.0, 306.1123352050781, 201.36949157714844, 0.0, 0.0, 1.0]
# R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
# P: [306.000244140625, 0.0, 318.4753112792969, 0.0, 0.0, 306.1123352050781, 201.36949157714844, 0.0, 0.0, 0.0, 1.0, 0.0]
# binning_x: 0
focal_length = 306
principal_point = (318.5, 201.4)  # ex: centre de l'image
K = np.array([[focal_length, 0, principal_point[0]],
              [0, focal_length, principal_point[1]],
              [0, 0, 1]])

# === Chemin vers dossier d'images ===
image_folder = "raw/test/camera_color_image_raw"

# Lister et trier les images
image_files = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

# Fonction pour détecter les points SIFT/ORB + matcher
def get_matched_points(img1, img2):
    # Convertir en niveau de gris
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Détecteur de points (ORB ici car gratuit)
    orb = cv2.ORB_create(2000)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return None, None

    # Matcher les descripteurs
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Trier les meilleurs matches
    matches = sorted(matches, key=lambda x: x.distance)[:100]

    # Extraire les points correspondants
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

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

    Rarray.append(R)
    Tarray.append(t)

print("Rotation (R):\n array size:\n", Rarray[1], len(Rarray))
cv2.destroyAllWindows()
