# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 12:56:17 2023

@author: S.Mouhamadi
"""

import numpy as np
import scipy.io as sp
import bm4d
import matplotlib.pyplot as plt
import skimage
import skimage.metrics as metrics
from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio
import scipy.stats as stats
import numpy.fft as fft
import glob
import os
from skimage.transform import resize



def do_bm4d_gauss_filter(mat_file_path):
    # Load the MATLAB file
    mat_data = sp.loadmat(mat_file_path)
    
    # Check if the 'imageDenoised' variable already exists
    if 'imageDenoised' in mat_data:
        denoised_image = mat_data['imageDenoised']
    else:
        # Perform BM4D filtering
        
        image_data = np.abs(mat_data['image3D'])  # Update the indexing here
        image_data = image_data.astype(np.float64)

        # Estimate the noise level with Universal Thresholding
        noise = np.std(image_data)
        sigma_psd = noise / np.sqrt(2 * np.log(image_data.size))
        
        # Estimate the noise level with MAD method 
        # med = np.median(image_data)
        # mad = np.median(np.abs(image_data - med)) 
        # sigma_psd = 1.4826 * mad

        profile = bm4d.BM4DProfile()  # Default profile
        stage_arg = bm4d.BM4DStages.ALL_STAGES  # Perform both hard thresholding and Wiener filtering
        blockmatches = (False, False)  # Do not use previous blockmatches

        # Apply BM4D to the image
        denoised_image = bm4d.bm4d(image_data, sigma_psd=sigma_psd, profile=profile, stage_arg=stage_arg, blockmatches=blockmatches)

        # Update the filtered image in the MATLAB file
        mat_data['imageDenoised'] = denoised_image

        # Save the filtered MATLAB file
        sp.savemat(mat_file_path, mat_data)
    
    # Store the images
    original_image = np.abs(mat_data['image3D'])
    denoised_image = np.abs(mat_data['imageDenoised'])


    return mat_data, original_image, denoised_image

# Test
mat_file_path = "C:/Users/Portatil PC 7/Documents/Data/Calves_ImageD/RAREprotocols_T1_TRA.2023.05.31.15.53.20.677_ImageD.mat"

mat_data, original_image, denoised_image = do_bm4d_gauss_filter(mat_file_path)


def load_imagenR_from_rawData(file_path):
    file_name = os.path.basename(file_path)

    if 'RAREprotocols_T1_TRA' in file_path:
        file_name_parts = file_name.split('_')
        dimensions_part = file_name_parts[3]  # Part containing the dimensions
        dimensions = list(dimensions_part.split())
        # Extract dimensions differently for filenames with 'RAREprotocols_T1_TRA'
        dim1 = int(dimensions[0])
        dim2 = int(dimensions[1])
        dim3 = int(dimensions[2])
        
    else:
        file_name_parts = file_name.split('_')
        dimensions_part = file_name_parts[1]  # Part containing the dimensions
        dimensions = list(dimensions_part.split())

        dim1 = int(dimensions[0])
        dim2 = int(dimensions[1])
        dim3 = int(dimensions[2])

    imagenR = np.loadtxt(file_path, delimiter=',')
    imagenR = imagenR.reshape(dim1, dim2, dim3)  # Redimensionner la matrice à la taille spécifiée en 3D
    
    # Load the text file containing the matrix
    # imagenR = np.loadtxt(file_path, delimiter=',')
    # imagenR = np.reshape(imagenR, [imagenR.shape[0], int(imagenR.shape[1]/imagenR.shape[0]), imagenR.shape[0]])
    
    # # Horizontal flip
    imagenR = np.fliplr(imagenR)
    # # Rotate 90 degrees counter-clockwise
    imagenR = np.rot90(imagenR, k=-1, axes=(1, 2))
    # imagenR = np.transpose(imagenR, (0, 1, 2))
    
    return imagenR

# Test
file_path = 'C:/Users/Portatil PC 7/Documents/Data/ImagenR/RAREprotocols_T1_TRA.2023.05.31.15.53.20.677_18  100  100_ImagenR.txt'
imagenR = load_imagenR_from_rawData(file_path)

def display_images(original_image, denoised_image, imagenR, slice_index, original_filename):
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Comparison for {original_filename}')

    # Display the original image
    axs[0].imshow(original_image[slice_index, :, :], cmap='gray')
    axs[0].set_title('Original Image')

    # Display the denoised image by BM4D
    axs[1].imshow(denoised_image[slice_index, :, :], cmap='gray')
    axs[1].set_title('Denoised Image (Python BM4D filter)')

    # Display the imagenR image
    axs[2].imshow(imagenR[slice_index, :, :], cmap='gray')
    axs[2].set_title('Matlab Denoised Image')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Display the figure
    plt.show()


def calculate_mse(image1, image2):
    # diff = image1 - image2
    # mse = np.mean(np.square(diff))
    rmse = mean_squared_error(image1, image2)
    return rmse

#PNSR is in dB and estimate the quality of an image
def calculate_psnr(image1,image2, rmse):
    if rmse == 0:
        return float('inf')
    max_value = max(image1.max(), image2.max())
    psnr = 20 * np.log10(max_value/np.sqrt(rmse))
    return psnr
    # psnr = skimage.metrics.peak_signal_noise_ratio(image1, image2,'' ,data_range=None)
    # return psnr

def calculate_ssim(image1, image2):
    ssim_score = metrics.structural_similarity(image1, image2)
    return ssim_score


def calculate_csc(image1, image2):
    image1_vector = image1.flatten()
    image2_vector = image2.flatten()
    csc_score = stats.pearsonr(image1_vector, image2_vector)[0]
    return csc_score


def calculate_cop(image1, image2):
    image1_fft = fft.fft2(image1)
    image2_fft = fft.fft2(image2)
    cop_score = np.abs(np.mean(image1_fft * np.conj(image2_fft)) / np.sqrt(np.mean(np.abs(image1_fft) ** 2) * np.mean(np.abs(image2_fft) ** 2)))
    return cop_score


import pandas as pd

def compare_performance(imageD_directory, imagenR_directory, slice_index):
    imageD_files = glob.glob(os.path.join(imageD_directory, '*ImageD*.mat'))
    imagenR_files = glob.glob(os.path.join(imagenR_directory, '*ImagenR*.txt'))
    
    # Create a dictionary to store performance values
    performance_data = {
        'Image': [],  # Original image filename
        'Filter Type': [],  # Filter type: Python BM4D (imageD) or MATLAB BM4D (imagenR)
        'RMSE': [],  # Root Mean Squared Error
        'PSNR': [],  # Peak Signal-to-Noise Ratio
        'SSIM': [],  # Structural Similarity Index
        'CSC': [],  # Cross-Scale Correlation
        'COP': []   # Coefficient of Phase
    }
    
    for imageD_file, imagenR_file in zip(imageD_files, imagenR_files):
        # Load the original image
        original_data = sp.loadmat(imageD_file)
        original_image = np.abs(original_data['image3D'])
        original_image = original_image.astype(np.float64)

        # Load the denoised image from imageD directory
        imageD_data = sp.loadmat(imageD_file)
        denoised_image_python = np.abs(imageD_data['imageDenoised'])

        # Load the denoised image from imagenR directory
        denoised_image_matlab = load_imagenR_from_rawData(imagenR_file)
        if denoised_image_python.shape != denoised_image_matlab.shape:
            denoised_image_matlab = resize(denoised_image_matlab, denoised_image_python.shape, anti_aliasing=True)

        # Calculate performance metrics for Python BM4D
        rmse_denoised_python = calculate_mse(original_image, denoised_image_python)
        psnr_denoised_python = calculate_psnr(original_image, denoised_image_python,rmse_denoised_python)
        ssim_denoised_python = calculate_ssim(original_image, denoised_image_python)
        csc_denoised_python = calculate_csc(original_image, denoised_image_python)
        cop_denoised_python = calculate_cop(original_image, denoised_image_python)

        # Calculate performance metrics for MATLAB BM4D
        rmse_denoised_matlab = calculate_mse(original_image, denoised_image_matlab)
        psnr_denoised_matlab = calculate_psnr(original_image, denoised_image_matlab, rmse_denoised_matlab)
        ssim_denoised_matlab = calculate_ssim(original_image, denoised_image_matlab)
        csc_denoised_matlab = calculate_csc(original_image, denoised_image_matlab)
        cop_denoised_matlab = calculate_cop(original_image, denoised_image_matlab)

        # Add performance values to the dictionary
        performance_data['Image'].append(os.path.basename(imageD_file))
        performance_data['Filter Type'].append('Python BM4D (imageD)')
        performance_data['RMSE'].append(rmse_denoised_python)
        performance_data['PSNR'].append(psnr_denoised_python)
        performance_data['SSIM'].append(ssim_denoised_python)
        performance_data['CSC'].append(csc_denoised_python)
        performance_data['COP'].append(cop_denoised_python)

        performance_data['Image'].append(os.path.basename(imageD_file))
        performance_data['Filter Type'].append('MATLAB BM4D (imagenR)')
        performance_data['RMSE'].append(rmse_denoised_matlab)
        performance_data['PSNR'].append(psnr_denoised_matlab)
        performance_data['SSIM'].append(ssim_denoised_matlab)
        performance_data['CSC'].append(csc_denoised_matlab)
        performance_data['COP'].append(cop_denoised_matlab)
        
        # Get the original filename without extension
        original_filename = os.path.splitext(os.path.basename(imageD_file))[0]
        
        # Display the images for comparison
        display_images(original_image, denoised_image_python, denoised_image_matlab, slice_index, original_filename)

    # Create a DataFrame from the performance dictionary
    result_df = pd.DataFrame(performance_data)

    # Save the result DataFrame as a CSV file
    result_file_path = os.path.join('C:/Users/Portatil PC 7/Documents/Data', 'performance_results.csv')
    result_df.to_csv(result_file_path, index=False)
    

    return result_df


input_directory = 'C:/Users/Portatil PC 7/Documents/Data/Calves/'
output_directory = 'C:/Users/Portatil PC 7/Documents/Data/Calves_ImageD/'
imagenR_directory = 'C:/Users/Portatil PC 7/Documents/Data/Calves_ImagenR/'

slice_index = 12
# Compare the performance and display the images
compare_performance(output_directory, imagenR_directory, slice_index)