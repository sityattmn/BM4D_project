# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import bm4d
import numpy as np
import scipy.io as sp
import matplotlib.pyplot as plt

################################################################# 2nd test : ################################################################# 


# Charger l'image Matlab 3D
mat_data = sp.loadmat('C:/Users/Portatil PC 7/Documents/Test_BM4D/rodillaBUENO_image3D.mat')
image_data = mat_data['image3D']
image_data = np.abs(image_data)  # prendre la magnitude pour avoir une image réelle
image_data = image_data.astype(np.float)  # conversion en float

# Stocker chaque plan de l'image dans une liste
image_planes = []
for i in range(image_data.shape[0]):
    image_planes.append(image_data[i,:,:])

# Appliquer BM4D à chaque plan de l'image
denoised_planes = []
for i in range(len(image_planes)):
    denoised_planes.append(bm4d.bm4d(image_planes[i], sigma_psd=0.05, stage_arg=bm4d.BM4DStages.ALL_STAGES))

# Empiler les plans débruités dans un tableau 3D
denoised_image = np.stack(denoised_planes, axis=0)

# Afficher les images sur la même figure
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# Image originale
axs[0].imshow(image_data[14,:,:], cmap='gray')
axs[0].set_title('Image originale')

# Image débruitée avec BM4D
axs[1].imshow(denoised_image[14,:,:], cmap='gray')
axs[1].set_title('Image débruitée avec BM4D')

plt.show()
    
    
################################################################# 1st test : ################################################################# 
    
    

# Prétraitement de l'image
noise_sigma = 10  # Niveau de bruit
image_data = image_data.astype(np.float32) / 255.0  # Conversion en float32
noisy_data = image_data + np.random.normal(scale=noise_sigma/255.0, size=image_data.shape)  # Ajout de bruit gaussien

# Débruitage de l'image
sigma = noise_sigma / 255.0  # Ecart-type du bruit
denoised_data = bm4d.bm4d(noisy_data, sigma)

# Affichage de l'image originale et de l'image débruitée
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

axes[0].imshow(image_data[:, :, 0], cmap='gray')
axes[0].set_title('Image originale')
axes[0].axis('off')

axes[1].imshow(denoised_data[:, :, 0], cmap='gray')
axes[1].set_title('Image débruitée')
axes[1].axis('off')

plt.show()
