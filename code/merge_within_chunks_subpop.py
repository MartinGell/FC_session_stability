
import h5py
import pandas as pd
import numpy as np
import glob
import nibabel as nb
from nilearn import connectome

results_dir = '/home/btervocl/shared/projects/martin_FC_stability/res/subpop/mat_parts'
outdir = f'/home/btervocl/shared/projects/martin_FC_stability/res/subpop'

# Find all matching HDF5 files
# h5_files = sorted(glob.glob(f'{results_dir}/within_sub_var_rows_*.h5'))
# h5_files.pop(1)
# h5_files.pop(5)
# h5_files.pop(4)
h5_files = [
    '/home/btervocl/shared/projects/martin_FC_stability/res/subpop/mat_parts/within_sub_var_rows_0_to_40000001.h5',
    '/home/btervocl/shared/projects/martin_FC_stability/res/subpop/mat_parts/within_sub_var_rows_40000001_to_190000001.h5',
    '/home/btervocl/shared/projects/martin_FC_stability/res/subpop/mat_parts/within_sub_var_rows_190000001_to_290000001.h5',
    '/home/btervocl/shared/projects/martin_FC_stability/res/subpop/mat_parts/within_sub_var_rows_290000001_to_390000001.h5',
    '/home/btervocl/shared/projects/martin_FC_stability/res/subpop/mat_parts/within_sub_var_rows_390000001_to_490000001.h5',
    '/home/btervocl/shared/projects/martin_FC_stability/res/subpop/mat_parts/within_sub_var_rows_490000001_to_600000001.h5',
    '/home/btervocl/shared/projects/martin_FC_stability/res/subpop/mat_parts/within_sub_var_rows_600000001_to_700000001.h5',
    '/home/btervocl/shared/projects/martin_FC_stability/res/subpop/mat_parts/within_sub_var_rows_700000001_to_850000001.h5',
    '/home/btervocl/shared/projects/martin_FC_stability/res/subpop/mat_parts/within_sub_var_rows_850000001_to_1000000001.h5',
    '/home/btervocl/shared/projects/martin_FC_stability/res/subpop/mat_parts/within_sub_var_rows_1000000001_to_1150000001.h5',
    '/home/btervocl/shared/projects/martin_FC_stability/res/subpop/mat_parts/within_sub_var_rows_1150000001_to_1300000001.h5',
    '/home/btervocl/shared/projects/martin_FC_stability/res/subpop/mat_parts/within_sub_var_rows_1300000001_to_1450000001.h5',
    '/home/btervocl/shared/projects/martin_FC_stability/res/subpop/mat_parts/within_sub_var_rows_1450000001_to_1600000001.h5',
    '/home/btervocl/shared/projects/martin_FC_stability/res/subpop/mat_parts/within_sub_var_rows_1600000001_to_1750000001.h5',
    '/home/btervocl/shared/projects/martin_FC_stability/res/subpop/mat_parts/within_sub_var_rows_1750000001_to_1900000001.h5'
]

# Initialize lists for within sub vars and connections
var_list = []

# Load the first file fully
with h5py.File(h5_files[0], 'r') as h5file:
    vars = h5file['within_sub_var'][()]
    var_list.append(vars)
    prev_last_conn_idx = h5file['connection'][-1].decode('utf-8')
    prev_last_conn_idx = int(prev_last_conn_idx.split('_')[1])

# Process remaining files
for fname in h5_files[1:]:
    with h5py.File(fname, 'r') as h5file:
        first_conn = h5file['connection'][0].decode('utf-8')
        last_conn = h5file['connection'][-1].decode('utf-8')

        first_conn_idx = int(first_conn.split('_')[1])

        print(f'last files last index: {prev_last_conn_idx}')
        print(f'this files first index: {first_conn_idx}')


        if first_conn_idx != prev_last_conn_idx + 1:
            print(f"Non-consecutive files: {prev_last_conn_idx} -> {first_conn_idx} in {fname}")
            print("Aborting to avoid data misalignment.")
            break

        # Append within sub vars only
        vars = h5file['within_sub_var'][()]
        var_list.append(vars)

        # Optionally store connection labels from just this file’s edges
        prev_last_conn_idx = int(last_conn.split('_')[1])
        # Don't load full connections array — just reconstruct them later if needed

# Concatenate and save
all_vars = np.concatenate(var_list)


#save the FC matrix
ref_sub = '/home/btervocl/shared/projects/martin_FC_stability/data/subpop/sub-1003001/ses-2/'
dconn = nb.load(f'{ref_sub}sub-1003001_ses-2_task-restMENORDICtrimmed_space-fsLR_den-91k_desc-denoised_bold_FD_02_smoothed_1.7mm.dconn.nii')
h5dconn = h5py.File(f'{ref_sub}sub-1003001_ses-2_task-restMENORDICtrimmed_space-fsLR_den-91k_desc-denoised_bold_FD_02_smoothed_1.7mm.dconn.h5', 'r')

flat_dconn_len = 4166156121 #len(h5dconn['data'][()]) # 4166156121
mat_dconn_len = h5dconn['n_grayordinates'][0] # 91282

var_vec = all_vars

rows_to_add = flat_dconn_len - var_vec.shape[0]
pad = np.zeros(rows_to_add, dtype=var_vec.dtype)
var_padded = np.concatenate([var_vec, pad])

var_mat = connectome.vec_to_sym_matrix(var_padded, diagonal=np.repeat(np.nan,int(mat_dconn_len)))
np.fill_diagonal(var_mat, 0)

new_img = nb.Cifti2Image(var_mat, header=dconn.header,
                         nifti_header=dconn.nifti_header)

new_img.to_filename(f'/home/btervocl/shared/projects/martin_FC_stability/res/subpop/img/subpop_full_within_sub_var.dconn.nii')


# Create a nodewise summary (akadscalar)
dscalar = nb.load(f'{ref_sub}sub-1003001_ses-combined_task-restMENORDICtrimmed_run-08_space-fsLR_den-91k_stat-reho_boldmap.dscalar.nii')
#save the ICC matrix
median_vec = np.nanmedian(var_mat, axis=0)
# zero_indices = np.where(median_vec == 0)[0]
# median_vec[299] = np.mean(np.append(median_vec[298],median_vec[300]))
# zero_indices = np.where(median_vec == 0)[0]
#median_vec[zero_indices] = 0.42

new_img = nb.Cifti2Image(median_vec.reshape(1, int(mat_dconn_len)), header=dscalar.header,
                         nifti_header=dscalar.nifti_header)
new_img.to_filename(f'/home/btervocl/shared/projects/martin_FC_stability/res/subpop/img/subpop_full_within_sub_var_median.dscalar.nii')