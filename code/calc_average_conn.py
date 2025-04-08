
import numpy as np
import pandas as pd
import glob

import matplotlib.pyplot as plt

import nibabel as nb
from nilearn import connectome


feature = '4S1056Parcels' #


indir = '/home/btervocl/shared/projects/martin_SNR/input/subpop'
#indir = '/Users/mgell/Work/SNR/input/subpop'
outdir = '/home/btervocl/shared/projects/martin_SNR/res/subpop'

subs = glob.glob(f'{indir}/*')
file_paths = glob.glob(f"{indir}/sub-*/ses-*/*{feature}*.pconn.nii")


data = []

for file_i in sorted(file_paths):

    # extract subject and session IDs from the file path
    parts = file_i.split("/")
    sub_id = parts[-3].split("-")[1]
    ses_id = parts[-2]
    
    print(f'Loading: {sub_id}, {ses_id}')

    # load data and get upper triangle
    nii = nb.load(file_i)
    dat = nii.get_fdata()
    print(dat.shape)
    upper = connectome.sym_matrix_to_vec(dat, discard_diagonal = True)
    
    # save
    data.append([*upper])


mean_upper = np.mean(np.array(data), axis=0)
mean_mat = connectome.vec_to_sym_matrix(mean_upper,diagonal=np.repeat(np.nan,len(dat)))
np.fill_diagonal(mean_mat, 1)

# Plot stuff
atlas = pd.read_table('/home/btervocl/shared/projects/martin_SNR/text_files/atlas-4S1056Parcels_dseg.tsv')
df = pd.read_csv(f'/home/btervocl/shared/projects/martin_SNR/res/subpop/subpop_icc_variances_{feature}.csv')
df['mean_fc'] = mean_upper

# Get the sorting order from the DataFrame column (based on 'SortOrder')
insort = atlas['7_network_index'].values.argsort()

# Generate example data

# Create a scatter plot
plt.figure(figsize=(8, 6))  # Adjust figure size for better readability
plt.scatter(df['between_sub_var'], df['mean_fc'], color='blue', edgecolor='k', alpha=0.7)

# Customize the plot
plt.xlabel('Between subject variance', fontsize=12)
plt.ylabel('Mean connectivity across subjects and sessions', fontsize=12)
#plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
#plt.legend(fontsize=10)

# Adjust tick parameters
plt.tick_params(axis='both', which='major', labelsize=10)

#plt.show()

file2save = f"{outdir}/plots/scatter_{feature}_BW_mea_FC.png"
print(f'saving: {file2save}')
plt.savefig(f'{file2save}', dpi=300, bbox_inches='tight')
plt.close()



# Create a scatter plot
plt.figure(figsize=(8, 6))  # Adjust figure size for better readability
plt.scatter(df['icc'], df['mean_fc'], color='blue', edgecolor='k', alpha=0.7)

# Customize the plot
plt.xlabel('ICC', fontsize=12)
plt.ylabel('Mean connectivity across subjects and sessions', fontsize=12)
#plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
#plt.legend(fontsize=10)

# Adjust tick parameters
plt.tick_params(axis='both', which='major', labelsize=10)

#plt.show()

file2save = f"{outdir}/plots/scatter_{feature}_icc_mea_FC.png"
print(f'saving: {file2save}')
plt.savefig(f'{file2save}', dpi=300, bbox_inches='tight')
plt.close()



# FC Mat
#indsort = atlas['7_network_index']

#display = plotting.plot_matrix(m_weights_mat)
#sorted = mean_mat[np.ix_(indsort-1,indsort.T-1)]

# Apply the sorting order to both axes (rows and columns)
sorted = mean_mat[insort, :][:, insort]

sorted[:,113] = 0
sorted[113,:] = 0

sorted[776,:] = 0
sorted[:,776] = 0

rows_to_remove = [113, 776]
sorted = np.delete(sorted, rows_to_remove, axis=0)
sorted = np.delete(sorted, rows_to_remove, axis=1)


cmap_custom = plt.cm.RdBu_r

plt.figure(figsize=(7, 7))
plt.imshow(sorted, origin='lower', cmap=cmap_custom, vmin=-1, vmax=1)
cbar = plt.colorbar(fraction=0.046)
plt.show()

file2save = f"{outdir}/plots/mean_FC_all_subs_{feature}_sorted.png"
print(f'saving: {file2save}')
plt.savefig(f'{file2save}', dpi=300)
plt.close()

d = []
d.append([*connectome.sym_matrix_to_vec(sorted, discard_diagonal = True)])




# BW
plt.rcParams.update({'font.size': 15})

icc_mat = connectome.vec_to_sym_matrix(df['between_sub_var'],diagonal=np.repeat(np.nan,len(dat)))
np.fill_diagonal(icc_mat, 1)

sorted = icc_mat[insort, :][:, insort]

rows_to_remove = [113, 776]
sorted = np.delete(sorted, rows_to_remove, axis=0)
sorted = np.delete(sorted, rows_to_remove, axis=1)
d.append([*connectome.sym_matrix_to_vec(sorted, discard_diagonal = True)])


cmap_custom = plt.cm.YlGnBu

plt.figure(figsize=(7, 7))
plt.imshow(sorted, origin='lower', cmap=cmap_custom, vmin=0, vmax=0.05)
cbar = plt.colorbar(fraction=0.046)
plt.show()

#file2save = parts[10].split('task-')[1].split('_den')  # just want this: 'restMENORDICtrimmed_space-fsLR_seg-4S956Parcels'    ##### possibly just 4S956Parcels????
file2save = f"{outdir}/plots/BW_{feature}_sorted.png"
print(f'saving: {file2save}')
plt.savefig(f'{file2save}', dpi=300)
plt.close()


#WV
icc_mat = connectome.vec_to_sym_matrix(df['within_sub_var'],diagonal=np.repeat(np.nan,len(dat)))
np.fill_diagonal(icc_mat, 1)

sorted = icc_mat[insort, :][:, insort]

rows_to_remove = [113, 776]
sorted = np.delete(sorted, rows_to_remove, axis=0)
sorted = np.delete(sorted, rows_to_remove, axis=1)
d.append([*connectome.sym_matrix_to_vec(sorted, discard_diagonal = True)])

cmap_custom = plt.cm.YlGnBu

plt.figure(figsize=(7, 7))
plt.imshow(sorted, origin='lower', cmap=cmap_custom, vmin=0, vmax=0.01)
cbar = plt.colorbar(fraction=0.046)
plt.show()

#file2save = parts[10].split('task-')[1].split('_den')  # just want this: 'restMENORDICtrimmed_space-fsLR_seg-4S956Parcels'    ##### possibly just 4S956Parcels????
file2save = f"{outdir}/plots/WH_{feature}_sorted.png"
print(f'saving: {file2save}')
plt.savefig(f'{file2save}', dpi=300)
plt.close()



icc_mat = connectome.vec_to_sym_matrix(df['icc'],diagonal=np.repeat(np.nan,len(dat)))
np.fill_diagonal(icc_mat, 1)

sorted = icc_mat[insort, :][:, insort]

rows_to_remove = [113, 776]
sorted = np.delete(sorted, rows_to_remove, axis=0)
sorted = np.delete(sorted, rows_to_remove, axis=1)
d.append([*connectome.sym_matrix_to_vec(sorted, discard_diagonal = True)])

cmap_custom = plt.cm.YlGnBu

plt.figure(figsize=(7, 7))
plt.imshow(sorted, origin='lower', cmap=cmap_custom, vmin=0, vmax=1)
cbar = plt.colorbar(fraction=0.046)
plt.show()

#file2save = parts[10].split('task-')[1].split('_den')  # just want this: 'restMENORDICtrimmed_space-fsLR_seg-4S956Parcels'    ##### possibly just 4S956Parcels????
file2save = f"{outdir}/plots/icc_{feature}_sorted.png"
print(f'saving: {file2save}')
plt.savefig(f'{file2save}', dpi=300)
plt.close()


# save cleaned upper
cols = ['mean_fc','between_sub_var','within_sub_var','icc']
dd = pd.DataFrame(d)
dd = dd.T
dd.columns = cols
dd.to_csv(f'/home/btervocl/shared/projects/martin_SNR/res/subpop/subpop_icc_variances_{feature}_clean.csv', index=False)


dd['mean_fc_abs'] = np.abs(dd['mean_fc'])
dd.corr()

plt.figure(figsize=(8, 6))  # Adjust figure size for better readability
plt.scatter(dd['between_sub_var'], dd['mean_fc'], color='blue', edgecolor='k', alpha=0.7)

# Customize the plot
plt.xlabel('Between subject variance', fontsize=12)
plt.ylabel('Mean connectivity across subjects and sessions', fontsize=12)
#plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
#plt.legend(fontsize=10)

# Adjust tick parameters
plt.tick_params(axis='both', which='major', labelsize=10)

plt.show()







pscalar  = nb.load('/home/btervocl/shared/projects/martin_SNR/input/subpop/sub-1007501_ses-combined_task-restMENORDICtrimmed_run-12_space-fsLR_seg-4S1056Parcels_den-91k_stat-coverage_boldmap.pscalar.nii')
ptseries = nb.load('/home/btervocl/shared/projects/martin_SNR/input/subpop/sub-1007501/ses-2/sub-1007501_ses-2_task-restMENORDICtrimmed_space-fsLR_seg-4S1056Parcels_den-91k_stat-mean_timeseries.ptseries.nii')


psclr = pscalar.get_fdata()
print(psclr.shape)

dt = ptseries.get_fdata()
print(dt.shape)

icc_mat = connectome.vec_to_sym_matrix(df['icc'],diagonal=np.repeat(np.nan,len(dat)))

median_icc = np.nanmedian(icc_mat, axis=0)

new_img = nb.Cifti2Image(median_icc.reshape(1, 1056), header=pscalar.header,
                         nifti_header=pscalar.nifti_header)

new_img.to_filename(f'/home/btervocl/shared/projects/martin_SNR/res/subpop/img/subpop_median_icc_{feature}.pscalar.nii')


cov_icc = np.nanstd(icc_mat, axis=0) / np.nanmean(icc_mat, axis=0)

new_img = nb.Cifti2Image(cov_icc.reshape(1, 1056), header=pscalar.header,
                         nifti_header=pscalar.nifti_header)

new_img.to_filename(f'/home/btervocl/shared/projects/martin_SNR/res/subpop/img/subpop_cov_icc_{feature}.pscalar.nii')