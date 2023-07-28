# -*- coding: utf-8 -*-
"""
Created on Tue May  9 12:25:51 2023

@author: Portatil PC 7
"""
import os
import numpy as np
import bm4d
import scipy.io as sp
from skimage import feature, segmentation, exposure
import matplotlib.pyplot as plt

# Charger l'image Matlab 3D
mat_data = sp.loadmat('C:/Users/Portatil PC 7/Documents/12_Guillermo/Foreleg/Left/mat/RAREprotocols_T1_TRA.2023.05.31.12.03.48.164.mat')
#'C:/Users/Portatil PC 7/Documents/12_Guillermo/Foreleg/Left/mat/RAREprotocols_T1_TRA.2023.05.31.12.03.48.164.mat'
image_data = mat_data['image3D']
image_data = np.abs(image_data)  # prendre la magnitude pour avoir une image réelle
image_data = image_data.astype(np.float)  # conversion en float

# Estimate the noise level using the MAD method
# sigma_psd = np.median(image_data)/0.6745

noise = np.std(image_data)
sigma_psd = noise / np.sqrt(2 * np.log(image_data.size))

# med = np.median(image_data)
# mad = np.median(np.abs(image_data - med))
# sigma_psd = 1.4826 * mad

print(f"Noise level (MAD): {sigma_psd:.2e}")

profile = bm4d.BM4DProfile()  # Profil par défaut
stage_arg = bm4d.BM4DStages.ALL_STAGES  # Effectuer à la fois le seuillage dur et le filtrage Wiener
blockmatches = (False, False)  # Ne pas utiliser les blockmatches précédents


# Débruitage de l'image
denoised_image = bm4d.bm4d(image_data, sigma_psd= sigma_psd, profile=profile, stage_arg=stage_arg, blockmatches=blockmatches)


# Rapport signal sur bruit 
snr = np.mean(image_data) / np.std(image_data)
print(snr)

slice_index = 10

# Apply intensity normalization to the selected slice
normalized_image3D = exposure.rescale_intensity(image_data[slice_index,:,:])
normalized_imageDenoised = exposure.rescale_intensity(denoised_image[slice_index,:,:])

# Apply contour detection (Canny edge detection) to the selected slice
edges_original = feature.canny(normalized_image3D, sigma=0.1)
edges_denoised = feature.canny(normalized_imageDenoised, sigma=0.1)

# Generate marked boundaries for the original and denoised images
marked_original = segmentation.mark_boundaries(normalized_image3D, edges_original)
marked_denoised = segmentation.mark_boundaries(normalized_imageDenoised, edges_denoised)


# Afficher les images sur la même figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))

# Image originale
axs[0, 0].imshow(image_data[slice_index, :, :], cmap='gray')
axs[0, 0].set_title('Image originale')

# Image débruitée avec BM4D
axs[0, 1].imshow(denoised_image[slice_index, :, :], cmap='gray')
axs[0, 1].set_title('Image débruitée avec BM4D')
 
# axs[1].text(0, -10, f"Sigma_psd: {sigma_psd:.2e}", fontsize=10, color='w')
# axs[1].text(0, -30, f"SNR: {snr:.2f}", fontsize=10, color='w')

axs[1, 0].imshow(marked_original, cmap='gray')
axs[1, 0].set_title('Contour Detection - Original Image')

axs[1, 1].imshow(marked_denoised, cmap='gray')
axs[1, 1].set_title('Contour Detection - Denoised Image')

plt.show()

# # Fonction pour afficher les tranches de l'image avec un slider
# def show_slices(slice_index):
#     fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

#     # Image originale
#     axs[0].imshow(image_data[slice_index,:,:], cmap='gray')
#     axs[0].set_title('Image originale')

#     # Image débruitée avec BM4D
#     axs[1].imshow(denoised_image[slice_index,:,:], cmap='gray')
#     axs[1].set_title('Image débruitée avec BM4D')

#     plt.show()

# # Créer le slider interactif
# interact(show_slices, slice_index=IntSlider(min=0, max=image_data.shape[0]-1, step=1, value=0))

