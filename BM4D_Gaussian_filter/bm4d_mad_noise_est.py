# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 17:11:20 2023

@author: S.Mouhamadi
"""

import bm4d
import numpy as np
import scipy.io as sp
import matplotlib.pyplot as plt

# Charger l'image Matlab 3D
mat_data = sp.loadmat('C:/Users/Portatil PC 7/Documents/12_Guillermo/Foreleg/Left/mat/RAREprotocols_T1_TRA.2023.05.31.12.03.48.164.mat')
image_data = mat_data['image3D']
image_data = np.abs(image_data)  # prendre la magnitude pour avoir une image réelle
image_data = image_data.astype(np.float64)  # conversion en float

# Estimate the noise level using the MAD method and Universal Thresholding

med = np.median(image_data)
mad = np.median(np.abs(image_data - med))
noise = 1.4826 * mad
sigma_psd = noise / np.sqrt(2 * np.log(image_data.size))




print(f"Noise level (Histogram): {sigma_psd:.2e}")

# Display the histogram
# plt.figure()
# plt.plot(bin_centers, histogram, label='Image Histogram')
# plt.plot(bin_centers[:len(background_hist)], background_hist, label='background_hist')
# plt.plot(bin_centers[:len(background_hist)], np.exp(np.polyval(fit_params, bin_centers[:len(background_hist)])), label='Polynomial Fit', linestyle='dashed')
# plt.xlabel('Pixel Value')
# plt.ylabel('Occurrences')
# plt.title('Image Histogram with Polynomial Fit')
# plt.legend()

profile = bm4d.BM4DProfile()  # Profil par défaut
stage_arg = bm4d.BM4DStages.ALL_STAGES  # Effectuer à la fois le seuillage dur et le filtrage Wiener
blockmatches = (False, False)  # Ne pas utiliser les blockmatches précédents

# Débruitage de l'image
denoised_image = bm4d.bm4d(image_data, sigma_psd=sigma_psd, profile=profile, stage_arg=stage_arg, blockmatches=blockmatches)

# Calculate SNR in decibels
snr_db = 10 * np.log10(np.mean(image_data ** 2) / np.mean((image_data - denoised_image) ** 2))

print(f"SNR: {snr_db:.2f} dB")

# Afficher les images sur la même figure
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# Image originale
axs[0].imshow(image_data[10, :, :], cmap='gray')
axs[0].set_title('Original Image')

# Image débruitée avec BM4D
axs[1].imshow(denoised_image[10, :, :], cmap='gray')
axs[1].set_title('BM4D Denoised Image')

axs[1].text(0, -10, f"Sigma_psd: {sigma_psd:.2e}", fontsize=10, color='w')
axs[1].text(0, -30, f"SNR: {snr_db:.2f}", fontsize=10, color='w')

plt.show()
