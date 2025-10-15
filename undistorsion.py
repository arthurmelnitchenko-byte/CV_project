import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# === Paramètres de la caméra (à ajuster si tu les connais) ===
D = np.array([-1.2477725744247437, 0.8747861981391907, -9.421713184565306e-05, -0.00014916047803126276, -0.2381284087896347, -1.2307056188583374, 0.8520383238792419, -0.2296648770570755])

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


def undistort_rational(image, camera_matrix, D):
    """
    Corrige la distorsion d'une image en utilisant un modèle rationnel polynomial.

    Args:
        image: Image d'entrée (format numpy array, BGR).
        camera_matrix: Matrice des paramètres intrinsèques de la caméra (3x3).
        D: Vecteur des 8 coefficients de distorsion rationnelle polynomiale.

    Returns:
        Image corrigée (format numpy array, BGR).
    """
    h, w = image.shape[:2]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]

    # Créer une image vide pour le résultat
    undistorted_image = np.zeros_like(image)

    # Parcourir chaque pixel de l'image
    for y in range(h):
        for x in range(w):
            # Normaliser les coordonnées
            x_norm = (x - cx) / fx
            y_norm = (y - cy) / fy
            r2 = x_norm**2 + y_norm**2

            # Extraire les coefficients
            k1, k2, p1, p2, k3, k4, k5, k6 = D

            # Appliquer le modèle rationnel polynomial
            numerator = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
            denominator = 1 + k4 * r2 + k5 * r2**2 + k6 * r2**3
            x_undistorted = x_norm * (numerator / denominator) + 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm**2)
            y_undistorted = y_norm * (numerator / denominator) + p1 * (r2 + 2 * y_norm**2) + 2 * p2 * x_norm * y_norm

            # Reprojection dans l'image
            x_u = x_undistorted * fx + cx
            y_u = y_undistorted * fy + cy

            # Vérifier que les coordonnées sont dans les limites
            if 0 <= x_u < w and 0 <= y_u < h:
                # Interpolation bilinéaire pour éviter les artefacts
                x0, y0 = int(np.floor(x_u)), int(np.floor(y_u))
                x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
                a = x_u - x0
                b = y_u - y0
                for c in range(image.shape[2]):
                    value = (1 - a) * (1 - b) * image[y0, x0, c] + \
                            a * (1 - b) * image[y0, x1, c] + \
                            (1 - a) * b * image[y1, x0, c] + \
                            a * b * image[y1, x1, c]
                    undistorted_image[y, x, c] = value

    return undistorted_image


# Boucle sur les paires d'images
Rarray=[]
Tarray=[]
for i in range(len(image_files) - 1):
    img = cv2.imread(os.path.join(image_folder, image_files[i]))

    undist = undistort_rational(img,K,D)

    cv2.imwrite(f"image{i}.png", undist)
    print(f'\rimage {i}\r')