# -*- coding: utf-8 -*-
"""
Created on Tue May  9 12:25:51 2023

@author: S.Mouhamadi
"""

import numpy as np
import scipy.io as sp
import bm4d
import matplotlib.pyplot as plt


def do_bm4d_gauss_filter(mat_file_path):
    # Load the MATLAB file
    mat_data = sp.loadmat(mat_file_path)
    image_data = np.abs(mat_data['image3D'])  # Update the indexing here
    image_data = image_data.astype(np.float64)

    # Estimate the noise level 
    noise = np.std(image_data)
    sigma_psd = noise / np.sqrt(2 * np.log(image_data.size))
    
    med = np.median(image_data)
    mad = np.median(np.abs(image_data - med))
    noise = 1.4826 * mad
    sigma_psd = noise / np.sqrt(2 * np.log(image_data.size))
    
    histogram, bins = np.histogram(image_data.flatten(), bins='auto')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    background_hist = histogram[:int(len(histogram) * 0.1)]  # Prendre la partie de l'histogramme correspondant au bruit
    fit_params = np.polyfit(bin_centers[:len(background_hist)], np.log(background_hist), deg=2)
    sigma_psd = np.sqrt(-1 / fit_params[0])

    profile = bm4d.BM4DProfile()  # Default profile
    stage_arg = bm4d.BM4DStages.ALL_STAGES  # Perform both hard thresholding and Wiener filtering
    blockmatches = (False, False)  # Do not use previous blockmatches

    # Apply BM4D to the image
    denoised_image = bm4d.bm4d(image_data, sigma_psd=sigma_psd, profile=profile, stage_arg=stage_arg, blockmatches=blockmatches)

    # Update the filtered image in the MATLAB file
    mat_data['imageDenoised'] = denoised_image

    # Save the filtered MATLAB file
    sp.savemat(mat_file_path, mat_data)
    
    # Store the original image and the denoised image
    original_image = np.abs(mat_data['image3D'])
    denoised_image = np.abs(mat_data['imageDenoised'])

    return mat_data, original_image, denoised_image

# Test
mat_file_path = "C:/Users/Portatil PC 7/Documents/Data/Calves_ImageD/RAREprotocols_T1_TRA.2023.05.11.12.47.42.304_ImageD.mat"

mat_data, original_image, denoised_image = do_bm4d_gauss_filter(mat_file_path)

def display_images(original_image, denoised_image, slice_index):
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Display the original image
    axs[0].imshow(original_image[slice_index, :, :], cmap='gray')
    axs[0].set_title('Original Image')

    # Display the denoised image by BM4D
    axs[1].imshow(denoised_image[slice_index, :, :], cmap='gray')
    axs[1].set_title('Denoised Image (Python BM4D filter)')

    # Display the figure
    plt.show()
    
slice_index = 10    
display_images(original_image, denoised_image, slice_index)