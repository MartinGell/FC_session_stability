
import numpy as np
import time
import os
import h5py
import nibabel as nb
from nilearn import connectome

# Author: Martin Gell 
# Purpose: transform dconn file into hdf5 file (shrinks dconn from ~32 GB to ~8 GB)
# Last Updated: 4 Feb 2025


def dconn_to_hdf5(dconn_in):
    """
    Converts a dense connectome (.dconn.nii) file to HDF5 format.
    Importantly, we only save the upper triangle of the conn matrix without the diagonal.

    Parameters:
    dconn_in (str): The base name of the dense connectome file.

    The output HDF5 file will have the following datasets:
    - 'data': The upper triangle of the connectivity matrix.
    - 'diagonal_value': The value of the diagonal element of the connectivity matrix.
    - 'n_grayordinates': The number of grayordinates (nodes) in the connectivity matrix.
    """

    # start timer
    start = time.time()

    # load data and get upper triangle
    print('loading...')
    nii = nb.load(dconn_in)
    mat = nii.get_fdata()

    print(f'{dconn_in}')
    
    # turn to float16 to reduce size
    mat = mat.astype(np.float16)

    # extract only upper triangle of conn mat
    upper = connectome.sym_matrix_to_vec(mat, discard_diagonal = True)
    diagonal_value = mat[0,0]

    # save out as hdf5
    # -should take about 25 seconds
    # strip dconn_in of any file extensions
    file_out = os.path.splitext(dconn_in)[0]
    hdf5_out = f'{file_out}.h5'

    print('Saving...')
    h = h5py.File(hdf5_out, 'w')
    dset = h.create_dataset('data', data=upper)
    dset = h.create_dataset('diagonal_value', data=np.atleast_1d(diagonal_value))
    dset = h.create_dataset('n_grayordinates', data=np.atleast_1d(mat.shape[0]))
    h.close() # important to close the file!

    # print out file path and end timer
    print(f'{hdf5_out}')

    end = time.time()
    print(f'Exporting dconn as HDF5 took {end - start} seconds')




def hdf5_to_dconn(hdf5_in, dconn_ref):
    """
    Converts an HDF5 file containing connectome data to a CIFTI-2 dense connectivity (dconn.nii) file,
    using the header information from the reference dconn file.

    Parameters:
    hdf5_in (str): Path to the input HDF5 file.
    dconn_ref (str): Path to a reference dconn file to copy header information from.
            This should be ideally the exact type of dconn that you want reconstructed.
    """

    # Open the HDF5 file for reading
    print('loading...')
    hdf5_file = h5py.File(hdf5_in,'r')

    print(f'{hdf5_in}')

    # Read the upper triangular part of the connectome matrix
    upper = hdf5_file["data"][0:len(hdf5_file["data"])]

    # Get the info about diagonal
    diag_len =   hdf5_file["n_grayordinates"][0]
    diag_value = hdf5_file["diagonal_value"][0]

    # Reconstruct the full symmetric matrix
    print('reconstructing matrix...')
    mat = connectome.vec_to_sym_matrix(upper, diagonal=np.repeat(np.nan,diag_len))
    np.fill_diagonal(mat, diag_value)

    # Convert the matrix to float32
    mat = mat.astype(np.float32)

    # Load the reference dconn file to copy header information
    nii = nb.load(f'{dconn_ref}')

    # Create a new CIFTI-2 image with the reconstructed matrix and the copied header information
    file_out = os.path.splitext(hdf5_in)[0]
    file_out = os.path.splitext(file_out)[0]
    dconn_out = f'{file_out}_NEW.dconn.nii'

    print('Saving...')
    new_img = nb.Cifti2Image(mat, header=nii.header,
                        nifti_header=nii.nifti_header)
    new_img.to_filename(f'{dconn_out}')

    print(f'{dconn_out}')

    # Close the HDF5 file
    hdf5_file.close() # important to close the file!

