
import numpy as np
import pandas as pd
import os
import glob

import matplotlib.pyplot as plt
import numpy as np

import nibabel as nb
from nilearn import connectome
from pathlib import Path


feature = 'Gordon' #4S1056Parcels
dataset = 'adultcontrols25' # 'subpop', 'MSC', HCPtrt_cneuro, MSC_4runs, subpop_UMN25, subpop_WashU25, adultcontrols25
####################################### 

# Go into this in ipython and test with different dataset. Then finish MSC25 and adultcontrols25

# Prep folder locations
wd = os.getcwd()
wd = Path(os.path.dirname(wd))
indir = wd / 'data' / dataset
outdir = wd / 'res' / dataset
outdir.mkdir(parents=True, exist_ok=True)

clean_data_outdir = wd / 'res' / 'cleaned_variances_reliability'
clean_data_outdir.mkdir(parents=True, exist_ok=True)

# Atlas sorting order
if feature == 'Glasser':
    # Load Glasser sorting index
    indsort = np.loadtxt(f'{wd}/data/cortex_subcortex_community_order.txt',dtype=int) -1
    indsort.shape = (len(indsort),1)
elif feature == 'Gordon':
    atlas = pd.read_csv(f'{wd}/data/Gordon_network_order.csv')
    atlas = atlas.drop(range(333, len(atlas)), axis=0)  # drop the extra rows that are not in the matrix
    indsort = atlas['parcel_order'].values.argsort()
    indsort.shape = (len(indsort),1)
    netsort = atlas['net_order'].values.argsort()
    netsort.shape = (len(netsort),1)
else:
    raise ValueError(f'Unknown feature order: {feature}')

# Create a mask of nans across all subjects and sessions
file_paths = glob.glob(f"{wd}/data/{feature}_nan_rows/*.txt")
all_nan_rows = []
for file_i in sorted(file_paths):
    nan_rows = np.loadtxt(file_i, dtype=bool)
    all_nan_rows.append(nan_rows)

all_nan_rows = pd.DataFrame(all_nan_rows)
template_nan_rows = np.any(all_nan_rows, axis=0)

# Get file paths
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

# average mat
mean_upper = np.mean(np.array(data), axis=0)
mean_mat = connectome.vec_to_sym_matrix(mean_upper,diagonal=np.repeat(np.nan,len(dat)))
np.fill_diagonal(mean_mat, 1)

# Save
icc_var_data = outdir / f'{dataset}_icc_variances_{feature}.csv'
df = pd.read_csv(icc_var_data)

df['mean_fc'] = mean_upper


### Plot
# FC Mat
sorted_mat = mean_mat[indsort,indsort.T]
sorted_mat = sorted_mat[~template_nan_rows, :][:, ~template_nan_rows]

# List labels corresponding to the sorted networks
raw_labels = np.array([f"{i}" for i in atlas['net_order'].values[indsort.flatten()].flatten()]).astype(int)
filtered_labels = raw_labels[~template_nan_rows]
filtered_labels.shape[0] == sorted_mat.shape[0]  # sanity check

# Find indices where the value changes (np.diff != 0 returns True at the index just BEFORE a change)
change_indices = np.where(np.diff(filtered_labels) != 0)[0]
line_positions = change_indices + 0.5 # inbetween pixels

cmap_custom = plt.cm.RdBu_r

fig, ax = plt.subplots(figsize=(7, 7))
im = ax.imshow(sorted_mat, origin='lower', cmap=cmap_custom, vmin=-1, vmax=1)

# add lines to separate networks
ax.vlines(x=line_positions, ymin=-0.5, ymax=len(filtered_labels)-0.5, 
          colors='black', linewidth=1, linestyle='-', alpha=0.5)
ax.hlines(y=line_positions, xmin=-0.5, xmax=len(filtered_labels)-0.5, 
          colors='black', linewidth=1, linestyle='-', alpha=0.5)

plt.colorbar(im, fraction=0.046)

file2save = f"{outdir}/plots/mean_FC_all_subs_{feature}_sorted.png"
print(f'saving: {file2save}')
plt.savefig(f'{file2save}', dpi=300)
plt.close()

# Save sorted cleaned data
d = []
d.append([*connectome.sym_matrix_to_vec(sorted_mat, discard_diagonal = True)])



# BW
plt.rcParams.update({'font.size': 15})

mat = connectome.vec_to_sym_matrix(df['between_sub_var'],diagonal=np.repeat(np.nan,len(dat)))
np.fill_diagonal(mat, 0)

sorted_mat = mat[indsort,indsort.T]
sorted_mat = sorted_mat[~template_nan_rows, :][:, ~template_nan_rows]

cmap_custom = plt.cm.YlGnBu

plt.figure(figsize=(7, 7))
plt.rcParams.update({'font.size': 15})
plt.imshow(sorted_mat, origin='lower', cmap=cmap_custom, vmin=0, vmax=0.05)
cbar = plt.colorbar(fraction=0.046)
plt.show()
plt.tick_params(labelsize=15)

#file2save = parts[10].split('task-')[1].split('_den')  # just want this: 'restMENORDICtrimmed_space-fsLR_seg-4S956Parcels'    ##### possibly just 4S956Parcels????
file2save = f"{outdir}/plots/BW_{dataset}_{feature}_sorted.png"
print(f'saving: {file2save}')
plt.savefig(f'{file2save}', dpi=300)
plt.close()

# Save cleaned data
d.append([*connectome.sym_matrix_to_vec(sorted_mat, discard_diagonal = True)])



#WV
mat = connectome.vec_to_sym_matrix(df['within_sub_var'],diagonal=np.repeat(np.nan,len(dat)))
np.fill_diagonal(mat, 0)

sorted_mat = mat[indsort,indsort.T]
sorted_mat = sorted_mat[~template_nan_rows, :][:, ~template_nan_rows]

cmap_custom = plt.cm.YlGnBu

cutoff = np.round(np.mean(df['within_sub_var']) + np.std(df['within_sub_var']) + np.std(df['within_sub_var']),3)

plt.figure(figsize=(7, 7))
plt.rcParams.update({'font.size': 15})
plt.imshow(sorted_mat, origin='lower', cmap=cmap_custom, vmin=0, vmax=cutoff)
cbar = plt.colorbar(fraction=0.046)

if dataset == 'adultcontrols25':
    ticks = np.linspace(0, cutoff, 9)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(t) for t in ticks])
elif dataset == 'MSC25':
    ticks = np.linspace(0, cutoff, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(t) for t in ticks])
elif dataset == 'subpop_UMN25':
    ticks = np.linspace(0, cutoff, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(t) for t in ticks])

plt.show()
plt.tick_params(labelsize=15)

#file2save = parts[10].split('task-')[1].split('_den')  # just want this: 'restMENORDICtrimmed_space-fsLR_seg-4S956Parcels'    ##### possibly just 4S956Parcels????
file2save = f"{outdir}/plots/WH_{dataset}_{feature}_sorted.png"
print(f'saving: {file2save}')
plt.savefig(f'{file2save}', dpi=300)
plt.close()

# Save cleaned data
d.append([*connectome.sym_matrix_to_vec(sorted_mat, discard_diagonal = True)])



# ICC
icc_mat = connectome.vec_to_sym_matrix(df['icc'],diagonal=np.repeat(np.nan,len(dat)))
np.fill_diagonal(icc_mat, 1)

sorted_mat = icc_mat[indsort,indsort.T]
sorted_mat = sorted_mat[~template_nan_rows, :][:, ~template_nan_rows]

cmap_custom = plt.cm.YlGnBu

plt.figure(figsize=(7, 7))
plt.imshow(sorted_mat, origin='lower', cmap=cmap_custom, vmin=0, vmax=1)
cbar = plt.colorbar(fraction=0.046)
plt.show()
plt.tick_params(labelsize=15)

#file2save = parts[10].split('task-')[1].split('_den')  # just want this: 'restMENORDICtrimmed_space-fsLR_seg-4S956Parcels'    ##### possibly just 4S956Parcels????
file2save = f"{outdir}/plots/icc_{dataset}_{feature}_sorted.png"
print(f'saving: {file2save}')
plt.savefig(f'{file2save}', dpi=300)
plt.close()

# Save cleaned data
d.append([*connectome.sym_matrix_to_vec(sorted_mat, discard_diagonal = True)])

# save cleaned upper
cols = ['mean_fc','between_sub_var','within_sub_var','icc']
clean_df = pd.DataFrame(d)
clean_df = clean_df.T
clean_df.columns = cols
clean_df.to_csv(f'{clean_data_outdir}/icc_variances_{dataset}_{feature}_clean_{sorted_mat.shape[0]}_rois_left.csv', index=False)


# Clean histogram
axes = clean_df.hist(bins=100, figsize=(10, 8))
file2save = f'{clean_data_outdir}/results_histograms_{dataset}_{feature}_clean.png'
plt.savefig(file2save, dpi=300)
plt.close()
print(file2save)



# Export mean ICC map as cifti
if dataset.startswith('subpop_UMN'):
    # Load reference subject data
    print('\nUsing subpop reference subject data')
    pscalar  = nb.load(f'{wd}/data/template_files/sub-1000201_ses-combined_task-restNORDIC_run-11_space-fsLR_seg-Glasser_den-91k_stat-coverage_boldmap.pscalar.nii')
elif dataset.startswith('subpop_WashU'):
    # Load reference subject data
    print('\nUsing subpop WashU reference subject data')
    pscalar  = nb.load(f'{wd}/data/template_files/sub-2003101_ses-combined_task-restNORDIC_run-07_space-fsLR_seg-Glasser_den-91k_stat-coverage_boldmap.pscalar.nii')
elif dataset.startswith('MSC'):
    # Load reference subject data
    print('\nUsing MSC reference subject data')
    pscalar  = nb.load(f'{wd}/data/template_files/sub-MSC04_ses-func06_task-rest_space-fsLR_seg-Glasser_den-91k_stat-coverage_boldmap.pscalar.nii')
# elif dataset == 'HCPtrt_cneuro':
    # Load reference subject data
    # print('\nUsing HCPtrt reference subject data')
    # pconn = nb.load(f'{indir}/sub-HCPtrt_cneuro/ses-1/sub-HCPtrt_cneuro_ses-1_task-rest_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries_FD_02.pconn.nii')
    # pconn_data = pconn.get_fdata()
elif dataset.startswith('adultcontrols'):
    # Load reference subject data
    print('\nUsing adult controls (ABSCAN) reference subject data')
    file2get = glob.glob(f'{wd}/data/template_files/sub-4812**{feature}**coverage_boldmap.pscalar.nii')[0]
    pscalar  = nb.load(file2get)
else:
    raise ValueError(f"Unknown dataset: {dataset}. NOT SAVING PLOTS.")

# Schaefer:
# pscalar  = nb.load('/home/btervocl/shared/projects/martin_SNR/input/subpop/sub-1007501_ses-combined_task-restMENORDICtrimmed_run-12_space-fsLR_seg-4S1056Parcels_den-91k_stat-coverage_boldmap.pscalar.nii')
# ptseries = nb.load('/home/btervocl/shared/projects/martin_SNR/input/subpop/sub-1007501/ses-2/sub-1007501_ses-2_task-restMENORDICtrimmed_space-fsLR_seg-4S1056Parcels_den-91k_stat-mean_timeseries.ptseries.nii')

# Glasser:
# ptseries = nb.load('/home/btervocl/shared/projects/martin_FC_stability/data/subpop/sub-1002901/ses-2/sub-1002901_ses-2_task-restMENORDICtrimmed_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii')
# pscalar  = nb.load('/home/btervocl/shared/projects/martin_FC_stability/data/template_maps/sub-1002901_ses-combined_task-restMENORDICtrimmed_run-10_space-fsLR_seg-Glasser_den-91k_stat-coverage_boldmap.pscalar.nii')


psclr = pscalar.get_fdata()
print(psclr.shape)

#dt = ptseries.get_fdata()
#print(dt.shape)

# ICC
icc_mat = connectome.vec_to_sym_matrix(df['icc'],diagonal=np.repeat(np.nan,psclr.shape[1]))
# #save the ICC matrix
# icc_save = pd.DataFrame(icc_mat)
# icc_save.to_csv(f'/home/btervocl/shared/projects/martin_FC_stability/res/subpop/icc_matrix_{feature}.csv')

median_icc = np.nanmedian(icc_mat, axis=0)
# zero_indices = np.where(median_icc == 0)[0]
# median_icc[299] = np.mean(np.append(median_icc[298],median_icc[300]))
# zero_indices = np.where(median_icc == 0)[0]
# median_icc[zero_indices] = 0.42

new_img = nb.Cifti2Image(median_icc.reshape(1, psclr.shape[1]), header=pscalar.header,
                         nifti_header=pscalar.nifti_header)

new_img.to_filename(f'{outdir}/plots/img/icc_median_{dataset}_{feature}.pscalar.nii')

# mean_icc_mat = np.nanmean(icc_mat, axis=0)
# zero_indices = np.where(mean_icc_mat == 0)[0]
# mean_icc_mat[299] = np.mean(np.append(mean_icc_mat[298],mean_icc_mat[300]))
# zero_indices = np.where(mean_icc_mat == 0)[0]
# mean_icc_mat[zero_indices] = 0.42

# cov_icc = np.nanstd(icc_mat, axis=0) / mean_icc_mat
# zero_indices = np.where(cov_icc == 0)[0]
# cov_icc[299] = np.mean(np.append(cov_icc[298],cov_icc[300]))
# zero_indices = np.where(cov_icc == 0)[0]

# new_img = nb.Cifti2Image(cov_icc.reshape(1, psclr.shape[1]), header=pscalar.header,
#                          nifti_header=pscalar.nifti_header)

# new_img.to_filename(f'/home/btervocl/shared/projects/martin_FC_stability/res/subpop/img/subpop_cov_icc_{feature}.pscalar.nii')

# Within
mat = connectome.vec_to_sym_matrix(df['within_sub_var'],diagonal=np.repeat(np.nan,psclr.shape[1]))
median_icc = np.nanmedian(mat, axis=0)

new_img = nb.Cifti2Image(median_icc.reshape(1, psclr.shape[1]), header=pscalar.header,
                         nifti_header=pscalar.nifti_header)

new_img.to_filename(f'{outdir}/plots/img/WH_median_{dataset}_{feature}.pscalar.nii')


# Between
mat = connectome.vec_to_sym_matrix(df['between_sub_var'],diagonal=np.repeat(np.nan,psclr.shape[1]))
median_icc = np.nanmedian(mat, axis=0)

new_img = nb.Cifti2Image(median_icc.reshape(1, psclr.shape[1]), header=pscalar.header,
                         nifti_header=pscalar.nifti_header)

new_img.to_filename(f'{outdir}/plots/img/BW_median_{dataset}_{feature}.pscalar.nii')






########################################################
# half between / half within-subject variance matrices #
########################################################

# Between-subject variance
between_mat = connectome.vec_to_sym_matrix(df['between_sub_var'],diagonal=np.repeat(np.nan,len(dat)))
np.fill_diagonal(between_mat, 0)

between_mat = between_mat[indsort,indsort.T]
between_mat = between_mat[~template_nan_rows, :][:, ~template_nan_rows]

# Within-subject variance
within_mat = connectome.vec_to_sym_matrix(df['within_sub_var'],diagonal=np.repeat(np.nan,len(dat)))
np.fill_diagonal(within_mat, 0)

within_mat = within_mat[indsort,indsort.T]
within_mat = within_mat[~template_nan_rows, :][:, ~template_nan_rows]

# for within_mat max and min values
cutoff = np.round(np.mean(df['within_sub_var']) + np.std(df['within_sub_var']) + np.std(df['within_sub_var']),3)

# Masks
mask_upper = np.triu(np.ones_like(between_mat, dtype=bool), k=1)
mask_lower = np.tril(np.ones_like(within_mat, dtype=bool), k=-1)

# Masked arrays
upper = np.ma.masked_where(~mask_upper, between_mat)
lower = np.ma.masked_where(~mask_lower, within_mat)

# Plotting
fig, ax = plt.subplots(figsize=(6, 6))
plt.rcParams.update({'font.size': 15})

# First colormap for upper triangle
cmap_A = plt.cm.YlGnBu
im_upper = ax.imshow(upper, origin = 'lower', cmap=cmap_A, vmin=0, vmax=0.05)

# Second colormap for lower triangle
cmap_B = plt.cm.YlGnBu
im_lower = ax.imshow(lower, origin = 'lower', cmap=cmap_B, vmin=0, vmax=cutoff)

ax.tick_params(labelsize=15)

plt.tight_layout()
plt.show()

file2save = Path(outdir) / 'plots' / f"combined_between_within_mat_{feature}.png"
print(f'saving: {file2save}')
plt.savefig(f'{file2save}', dpi=300)
plt.close()

