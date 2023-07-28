# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:31:24 2023

@author: S.Mouhamadi
"""

from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio
from scipy.stats import pearsonr
from skimage import io
from skimage.transform import resize

# Load the original image and the denoised images using the new BM4D code and the BM4D MATLAB code
image_original = io.imread('C:/Users/Portatil PC 7/Documents/Test_BM4D/image_originale2.png')
image_denoised_new_code = io.imread('C:/Users/Portatil PC 7/Documents/Test_BM4D/denoised_image.png')
image_denoised_matlab = io.imread('C:/Users/Portatil PC 7/Documents/Test_BM4D/denoised_image_mat.png')

# Resize the images to have the same size
image_original = resize(image_original, (767, 1054, 4), anti_aliasing=True)
image_denoised_new_code = resize(image_denoised_new_code, (767, 1054, 4), anti_aliasing=True)
image_denoised_matlab = resize(image_denoised_matlab, (767, 1054, 4), anti_aliasing=True)

# Calculate the Root Mean Squared Error (RMSE) between the original image and the new BM4D denoised image
rmse_new_code = mean_squared_error(image_original, image_denoised_new_code)

# Calculate the RMSE between the original image and the BM4D MATLAB denoised image
rmse_matlab = mean_squared_error(image_original, image_denoised_matlab)

# Calculate the Structural Similarity Index (SSIM) between the original image and the new BM4D denoised image
# Use a window size of 7x7 and specify the data range based on the denoised image intensity values
ssim_new_code = structural_similarity(image_original, image_denoised_new_code, win_size=7, data_range=image_denoised_new_code.max() - image_denoised_new_code.min(), channel_axis=-1)

# Calculate the SSIM between the original image and the BM4D MATLAB denoised image
ssim_matlab = structural_similarity(image_original, image_denoised_matlab, win_size=7, data_range=image_denoised_matlab.max() - image_denoised_matlab.min(), channel_axis=-1)

# Calculate the Peak Signal-to-Noise Ratio (PSNR) between the original image and the new BM4D denoised image
# Specify the data range based on the denoised image intensity values
psnr_new_code = peak_signal_noise_ratio(image_original, image_denoised_new_code, data_range=image_denoised_new_code.max() - image_denoised_new_code.min())

# Calculate the PSNR between the original image and the BM4D MATLAB denoised image
psnr_matlab = peak_signal_noise_ratio(image_original, image_denoised_matlab, data_range=image_denoised_matlab.max() - image_denoised_matlab.min())

# Calculate the Pearson correlation coefficient between the flattened original image and the new BM4D denoised image
# The second value returned by pearsonr() is the p-value
correlation_new_code, p_value_new_code = pearsonr(image_original.flatten(), image_denoised_new_code.flatten())

# Calculate the correlation between the flattened original image and the BM4D MATLAB denoised image
correlation_matlab, p_value_matlab = pearsonr(image_original.flatten(), image_denoised_matlab.flatten())

# Display the performance results
print("Performance of the new BM4D code:")
print(f"RMSE: {rmse_new_code:.2f}")
print(f"SSIM: {ssim_new_code:.4f}")
print(f"PSNR: {psnr_new_code:.2f}")
print(f"Correlation: {correlation_new_code:.4f}")
print(f"p-value: {p_value_new_code:.4f}")

print("\nPerformance of the BM4D MATLAB code:")
print(f"RMSE: {rmse_matlab:.2f}")
print(f"SSIM: {ssim_matlab:.4f}")
print(f"PSNR: {psnr_matlab:.2f}")
print(f"Correlation: {correlation_matlab:.4f}")
print(f"p-value: {p_value_matlab:.4f}")
