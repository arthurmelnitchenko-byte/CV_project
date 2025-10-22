import numpy as np
import pickle
import glob

# Load file lists
color_raw_files = sorted(glob.glob('raw/test/camera_color_image_raw/*.png'))
color_sync_files = sorted(glob.glob('sync_color_image/*.png'))

# Load transformation matrices
with open('Transformations.pkl', 'rb') as file:  # 'rb' for reading in binary mode
    T = pickle.load(file)

# Filter transformations to only include those with matching filenames
T_adapted = []
for i in range(len(color_raw_files)):
    for j in range(len(color_sync_files)):
        # Extract indices from filenames
        index1 = color_raw_files[i].split('_')[-1].split('.')[0]  # Remove file extension
        index2 = color_sync_files[j].split('_')[-1].split('.')[0]  # Remove file extension
        if index1 == index2:
            T_adapted.append(T[i])
            break  # No need to check other files once a match is found

# Save the reduced transformation matrices
with open('Transformations_reduced.pkl', 'wb') as file:  # 'wb' for writing in binary mode
    pickle.dump(T_adapted, file)

