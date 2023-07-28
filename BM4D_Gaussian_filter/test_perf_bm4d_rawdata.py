# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 12:24:37 2023

@author: S.Mouhamadi
"""

import numpy as np
import scipy.io as sp
import bm4d
import matplotlib.pyplot as plt
import skimage.metrics as metrics
from skimage.metrics import mean_squared_error
import glob
import os
import pandas as pd


def do_bm4d_gauss_filter(mat_file_path):
    # Load the MATLAB file
    mat_data = sp.loadmat(mat_file_path)
    image_data = np.abs(mat_data['image3D'])  # Update the indexing here
    image_data = image_data.astype(np.float64)

    # Estimate the noise level using the MAD method
    med = np.median(image_data)
    mad = np.median(np.abs(image_data - med))
    noise = 1.4826 * mad
    sigma_psd = noise / np.sqrt(2 * np.log(image_data.size))
    
    # Estimate the noise level using the Histogram method
    # histogram, bins = np.histogram(image_data.flatten(), bins='auto')
    # bin_centers = (bins[:-1] + bins[1:]) / 2
    # background_hist = histogram[:int(len(histogram) * 0.1)]  #select the noisy part or the histogramm
    # fit_params = np.polyfit(bin_centers[:len(background_hist)], np.log(background_hist), deg=1)
    # sigma_psd = np.sqrt(-1 / fit_params[0])

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


def load_imagenR_from_rawData(file_path):
    file_name = os.path.basename(file_path)

    if 'RAREprotocols_T1_TRA' in file_name:
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
    
    
    # imagenR = np.transpose(imagenR, (0, 2, 1)) #for joint image
    
    # # Horizontal flip
    # imagenR = np.fliplr(imagenR)
    # # Rotate 90 degrees counter-clockwise
    # imagenR = np.rot90(imagenR, k=1, axes=(2, 1))
    
   

    
    # # Horizontal flip
    # imagenR = np.flipud(imagenR)
    # # Rotate 90 degrees counter-clockwise
    # imagenR = np.rot90(imagenR, k=-1, axes=(2, 1))
    
        
    return imagenR


def display_images(original_image, denoised_image, imagenR, slice_index):
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Display the original image
    axs[0].imshow(original_image[slice_index, :, :], cmap='gray')
    axs[0].set_title('Original Image')

    # Display the denoised image by BM4D
    axs[1].imshow(denoised_image[slice_index, :, :], cmap='gray')
    axs[1].set_title('Denoised Image (Python BM4D filter)')

    # Display the imagenR image
    axs[2].imshow(imagenR[slice_index, :, :], cmap='gray')
    axs[2].set_title('Matlab Denoised Image')

    # Display the figure
    plt.show()


def calculate_rmse(image1, image2):
    # diff = image1 - image2
    # mse = np.mean(diff ** 2)
    # return mse
    rmse = mean_squared_error(image1, image2)
    return rmse


def calculate_psnr(rmse):
    if rmse == 0:
        return float('inf')
    max_value = 0.1
    psnr = 20 * np.log10(max_value/rmse)
    return psnr



def calculate_ssim(image1, image2):
    ssim = metrics.structural_similarity(image1, image2) 
    return ssim



def compare_performance(imageD_directory, imagenR_directory):
    imageD_files = glob.glob(os.path.join(imageD_directory, '*ImageD*.mat'))
    imagenR_files = glob.glob(os.path.join(imagenR_directory, '*ImagenR*.txt'))

    performance_data = {'Filter Type': [], 'MSE': [], 'PSNR': [], 'SSIM': []}

    for imageD_file, imagenR_file in zip( imageD_files, imagenR_files):
        # Load the original image
        original_data = sp.loadmat(imageD_file)
        original_image = np.abs(original_data['image3D'])
        original_image = original_image.astype(np.float64)

        # Load the denoised image from imageD directory
        imageD_data = sp.loadmat(imageD_file)
        denoised_image_python = np.abs(imageD_data['imageDenoised'])

        # Load the denoised image from imagenR directory
        denoised_image_matlab = load_imagenR_from_rawData(imagenR_file)

        # Compare the performance
        for i in range(original_image.shape[0]):
            rmse_python = calculate_rmse(original_image[i, :, :], denoised_image_python[i, :, :])
            psnr_python = calculate_psnr(rmse_python)
            ssim_python = calculate_ssim(original_image[i, :, :], denoised_image_python[i, :, :])

            # denoised_image_matlab = np.transpose(denoised_image_matlab, (0, 2, 1))  # Permute axes if necessary
            rmse_matlab = calculate_rmse(original_image[i, :, :], denoised_image_matlab[i, :, :])

            psnr_matlab = calculate_psnr(rmse_matlab)
            ssim_matlab = calculate_ssim(original_image[i, :, :], denoised_image_matlab[i, :, :])

            performance_data['Filter Type'].append('Python BM4D (imageD)')
            performance_data['MSE'].append(rmse_python)
            performance_data['PSNR'].append(psnr_python)
            performance_data['SSIM'].append(ssim_python)

            performance_data['Filter Type'].append('MATLAB BM4D (imagenR)')
            performance_data['MSE'].append(rmse_matlab)
            performance_data['PSNR'].append(psnr_matlab)
            performance_data['SSIM'].append(ssim_matlab)
            
            # # Display the images
            # fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            # axs[0].imshow(original_image[i, :, :], cmap='gray')
            # axs[0].set_title('Original Image')

            # axs[1].imshow(denoised_image_python[i, :, :], cmap='gray')
            # axs[1].set_title('Denoised Image (Python BM4D)')

            # axs[2].imshow(denoised_image_matlab[i, :, :], cmap='gray')
            # axs[2].set_title('Denoised Image (MATLAB BM4D)')

            # plt.tight_layout()
            # plt.show()
            
    result_df = pd.DataFrame(performance_data)

    # Save the result DataFrame as a CSV file
    result_file_path = os.path.join('C:/Users/Portatil PC 7/Documents/Data', 'performance_results_calves2.csv') #adapt the name of the csv file 
    result_df.to_csv(result_file_path, index=False)

    return result_df


def process_files(input_directory, output_directory):
    file_paths = glob.glob(os.path.join(input_directory, '*RARE*.mat'))
    # result_data = []  # List to store the results for all files
    for file_path in file_paths:
        mat_data, original_image, denoised_image = do_bm4d_gauss_filter(file_path)

        # Generate the output file path
        file_name = os.path.basename(file_path)
        file_name, extension = os.path.splitext(file_name)
        output_file_name = file_name + "_ImageD" + extension
        output_file_path = os.path.join(output_directory, output_file_name)

        # Save the denoised image as a MATLAB file in the output directory
        sp.savemat(output_file_path, mat_data)

        #     # Load imagenR from the corresponding directory
        # imagenR_file_paths = glob.glob(os.path.join(imagenR_directory, file_name))
        # for imagenR_file_path in imagenR_file_paths:     
        #     imagenR = load_imagenR_from_rawData(imagenR_file_path)

    #         # Compare the performance and store the results
    #         result_df = compare_performance(original_image, denoised_image, imagenR)
    #         result_data.append(result_df)
            
    # # Concatenate all the result DataFrames
    # result_df = pd.concat(result_data, keys=file_paths)

    # # Reset the index
    # result_df.reset_index(level=0, inplace=True)
    # result_df.rename(columns={'level_0': 'File Path'}, inplace=True)

    # # Save the result DataFrame as a CSV file
    # result_file_path = os.path.join('C:/Users/Portatil PC 7/Documents/Data', 'performance_results.csv')
    # result_df.to_csv(result_file_path, index=False)

    # return result_df
    
# def save_nifti(imageD_directory):
#     imageD_files = glob.glob(os.path.join(imageD_directory, '*ImageD*.mat'))
#     for imageD_file in imageD_files:
#         imageD_data = sp.loadmat(imageD_file)
#         imageD_data = imageD_data['imageDenoised']
#         nifti_file = nib.Nifti1Image(imageD_data, affine=np.eye(4))
#         output_file = os.path.join('C:/Users/Portatil PC 7/Documents/Data/Nifti Files/', os.path.basename(imageD_file) + '.nii.gz')
#         nib.save(nifti_file,output_file)


# Specify the input directory, output directory, and imagenR directory

            #Calf Image
input_directory = 'C:/Users/Portatil PC 7/Documents/Data/Calves/'
output_directory = 'C:/Users/Portatil PC 7/Documents/Data/Calves_ImageD/'
imagenR_directory = 'C:/Users/Portatil PC 7/Documents/Data/Calves_ImagenR/'

            #Joint Image
# input_directory = 'C:/Users/Portatil PC 7/Documents/Data/Joints/'
# output_directory = 'C:/Users/Portatil PC 7/Documents/Data/Joints_ImageD/'
# imagenR_directory = 'C:/Users/Portatil PC 7/Documents/Data/Joints_ImagenR/'

    # Process the files to apply python BM4D filter
# process_files(input_directory, output_directory)

    # Test of Compariison 
compare_performance(output_directory, imagenR_directory)

#Useless
# save_nifti(imageD_directory)