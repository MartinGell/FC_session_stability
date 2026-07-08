
# USE Conda env: martin_snr

import numpy as np
import pandas as pd
import glob
import warnings
import traceback
import os
import sys
import matplotlib.pyplot as plt
import nibabel as nb

from pathlib import Path
from nilearn import connectome
from joblib import Parallel, delayed
from statsmodels.regression.mixed_linear_model import MixedLM

# suppress warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses


### Functions ###
# function for edgewise ICC
def compute_icc_safe(data, col):
    try:
        # fit the mixed-effects model
        model = MixedLM.from_formula(f'{str(col)} ~ 1', groups='subject_id', data=data)
        rslt = model.fit(method=["bfgs"])

        # extract variances and calc icc
        between_sub_var = rslt.cov_re.iloc[0, 0]
        within_sub_var = rslt.scale
        icc = between_sub_var / (between_sub_var + within_sub_var)

        return {
            'column': col,
            'between_sub_var': between_sub_var,
            'within_sub_var': within_sub_var,
            'icc': icc,
            'error': None
        }

    except Exception as e:
        return {
            'column': col,
            'between_sub_var': 0,
            'within_sub_var': 0,
            'icc': 0,
            'error': str(e)
        }

# Batch computation for parallel processing 
def compute_icc_batch(df, columns):
    results = []
    for col in columns:
        subset_df = df[['subject_id', 'session_id', col]]
        results.append(compute_icc_safe(subset_df, col))
    return results



############ INPUT PARAMS ############
n_batches = 80 # Number of parallel jobs to run

# Which feature and dataset (folder) to use
feature = 'Gordon' # '4S1056Parcels' 'Glasser'
dataset = 'subpop_UMN25' # 'subpop_WashU25', 'MSC25', HCPtrt_cneuro, MSC_4runs subpop_UMN25 adultcontrols25
####################################### 



# Prep folder locations
wd = os.getcwd()
wd = Path(os.path.dirname(wd))
indir = wd / 'data' / dataset
outdir = wd / 'res' / dataset
outdir.mkdir(parents=True, exist_ok=True)

# Atlases
if feature == 'Glasser':
    # Load Glasser sorting index
    indsort = np.loadtxt(f'{wd}/data/cortex_subcortex_community_order.txt',dtype=int) -1
    indsort.shape = (len(indsort),1)
elif feature == 'Gordon':
    atlas = pd.read_csv(f'{wd}/data/Gordon_network_order.csv')
    atlas = atlas.drop(range(333, len(atlas)), axis=0)  # drop the extra rows that are not in the matrix
    indsort = atlas['parcel_order'].values.argsort()
    indsort.shape = (len(indsort),1)
else:
    raise ValueError(f'Unknown feature order: {feature}')

# Get all subjects data locations
subs = glob.glob(f'{indir}/*')
file_paths = glob.glob(f"{indir}/sub-*/ses-*/*{feature}*.pconn.nii")

# file_paths = file_paths[:6] + file_paths[12:] remove subjects below 17 years old for adult controls
# subs = subs[:1] + subs[3:]

# initialise
data = []
all_nan_rows = []

for file_i in sorted(file_paths):

    # extract subject and session IDs from the file path
    parts = file_i.split("/")
    sub_id = parts[-3].split("-")[1]
    ses_id = parts[-2]
    
    print(f'Loading: {sub_id}, {ses_id}')

    # load data and get upper triangle
    nii = nb.load(file_i)
    dat = nii.get_fdata()

    #dat = dat.astype(np.float16)
    dat = np.clip(dat, -0.999999, 0.999999) # shouldnt be necessary, avoids nans
    dat_z = np.arctanh(dat)  # Fisher z-transform

    # sort and plot actual FC matrix, not the z-scored values
    sorted_mat = dat[indsort,indsort.T]

    # remove rows/columns that are entirely NaN (excluding diagonal)
    mat_no_diag = sorted_mat.copy()
    np.fill_diagonal(mat_no_diag, np.nan)
    nan_rows = np.all(np.isnan(mat_no_diag), axis=1)
    all_nan_rows.append([sub_id, *nan_rows])

    sorted_mat = sorted_mat[~nan_rows, :][:, ~nan_rows]

    cmap_custom = plt.cm.RdBu_r

    file2save = outdir / 'plots' / 'FC' / feature / f"{parts[-1].split(".")[0]}.png"
    file2save.parent.mkdir(parents=True, exist_ok=True)
    print(f'saving: {file2save}')

    plt.figure(figsize=(7, 7))
    plt.imshow(sorted_mat, origin='lower', cmap=cmap_custom, vmin=-1, vmax=1)
    cbar = plt.colorbar(fraction=0.046)
    plt.savefig(f'{file2save}', dpi=180)
    plt.close()

    # save upper triangle
    print(dat_z.shape)
    upper = connectome.sym_matrix_to_vec(dat_z, discard_diagonal = True)
    
    # save
    data.append([sub_id, ses_id, *upper])

cols = ['subject_id', 'session_id'] + [f"conn_{i}" for i in range(len(data[0]) - 2)]
df = pd.DataFrame(data, columns=cols)

# Get the columns that are all NaN across subjects and save this information for later use in plotting
nan_cols = ["subject_id"] + [f"{i}" for i in atlas['full_name'].values[indsort.flatten()].flatten()]
all_nan_rows = pd.DataFrame(all_nan_rows, columns=nan_cols)
# save
save_dir = wd / 'data' / f'{feature}_nan_rows' / f"{dataset}_nan_rows_all_subs.csv"
save_dir.parent.mkdir(parents=True, exist_ok=True)
all_nan_rows.to_csv(save_dir, index=False)

# This is the final variable where at least one subject has a NaN value for that row/column
nan_rows_template = np.any(all_nan_rows, axis=0)
nan_rows_template = nan_rows_template[1:len(nan_rows_template)].values # skip the subject_id column
# save this as text
save_dir = wd / 'data' / f'{feature}_nan_rows' / f"{dataset}_nan_rows_template.txt"
np.savetxt(save_dir, nan_rows_template, fmt='%d')



### Computation time ###
# Create batches of columns
column_batches = np.array_split(
    [col for col in df.columns if col not in ['subject_id', 'session_id']], n_batches
)

# Parallel computation
Res = Parallel(n_jobs=n_batches)(
    delayed(compute_icc_batch)(df, batch) for batch in column_batches
)

results_df = pd.DataFrame([item for sublist in Res for item in sublist])

# Sort the results DataFrame by the original column order
original_order = [col for col in df.columns if col not in ['subject_id', 'session_id']]
if not results_df['column'].equals(pd.Series(original_order)):
    print('reordering...')
    results_df['column'] = pd.Categorical(results_df['column'], categories=original_order, ordered=True)
    results_df = results_df.sort_values('column').reset_index(drop=True)

# Save results and errors separately
results = results_df[['column', 'between_sub_var', 'within_sub_var', 'icc']]
error_log = results_df[results_df['error'].notnull()]
error_dir = outdir / f'{dataset}_icc_variances_{feature}_error_log.csv'
error_dir.parent.mkdir(parents=True, exist_ok=True)
error_log[['column', 'error']].to_csv(error_dir, index=False)

print(results.describe())

# save
file2save = f'{outdir}/{dataset}_icc_variances_{feature}.csv'
print(f'saving: {file2save}')
results.to_csv(file2save, index=False)


# Save histograms for quick viewing
axes = results.hist(bins=50, figsize=(10, 8))
plt.savefig(f'{outdir}/{dataset}_results_histograms_{feature}.png')
plt.close()
print(f'saving: {f'{outdir}/{dataset}_results_histograms_{feature}.png'}')




########## PLOTTING RESULTS ##############
if dataset.startswith('subpop_UMN'):
    # Load reference subject data
    print('\nUsing subpop reference subject data')
    if feature == 'Gordon':
        pconn = nb.load(f'{indir}/sub-1000201/ses-1/sub-1000201_ses-1_task-restNORDIC_space-fsLR_seg-Gordon_den-91k_stat-mean_timeseries_FD_02.pconn.nii')
    else:
        pconn = nb.load(f'{indir}/sub-1000201/ses-1/sub-1000201_ses-1_task-restNORDIC_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries_FD_02.pconn.nii')
    pconn_data = pconn.get_fdata()
elif dataset.startswith('subpop_WashU'):
    # Load reference subject data
    print('\nUsing subpop WashU reference subject data')
    if feature == 'Gordon':
        pconn = nb.load(f'{indir}/sub-2003101/ses-3/sub-2003101_ses-3_task-restNORDIC_space-fsLR_seg-Gordon_den-91k_stat-mean_timeseries_FD_02.pconn.nii')
    else:
        pconn = nb.load(f'{indir}/sub-2003101/ses-3/sub-2003101_ses-3_task-restNORDIC_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries_FD_02.pconn.nii')
    pconn_data = pconn.get_fdata()
elif dataset.startswith('MSC'):
    # Load reference subject data
    print('\nUsing MSC reference subject data')
    if feature == 'Gordon':
        pconn = nb.load(f'{indir}/sub-MSC04/ses-func07/sub-MSC04_ses-func07_task-rest_space-fsLR_seg-Gordon_den-91k_stat-mean_timeseries_FD_02.pconn.nii')
    else:
        pconn = nb.load(f'{indir}/sub-MSC04/ses-func07/sub-MSC04_ses-func07_task-rest_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries_FD_02.pconn.nii')
    pconn_data = pconn.get_fdata()
elif dataset == 'HCPtrt_cneuro':
    # Load reference subject data
    print('\nUsing HCPtrt reference subject data')
    pconn = nb.load(f'{indir}/sub-HCPtrt_cneuro/ses-1/sub-HCPtrt_cneuro_ses-1_task-rest_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries_FD_02.pconn.nii')
    pconn_data = pconn.get_fdata()
elif dataset.startswith('adultcontrols'):
    # Load reference subject data
    print('\nUsing adult controls (ABSCAN) reference subject data')
    if feature == 'Gordon':
        pconn = nb.load(f'{indir}/sub-4808/ses-1/sub-4808_ses-1_task-restMENORDICrmnoisevols_space-fsLR_seg-Gordon_den-91k_stat-mean_timeseries_FD_02.pconn.nii')
    else:
        pconn = nb.load(f'{indir}/sub-4808/ses-1/sub-4808_ses-1_task-restMENORDICrmnoisevols_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries_FD_02.pconn.nii')
    pconn_data = pconn.get_fdata()
else:
    raise ValueError(f"Unknown dataset: {dataset}. NOT SAVING PLOTS.")


# Between-subject variance
icc_mat = connectome.vec_to_sym_matrix(results['between_sub_var'],diagonal=np.repeat(np.nan,len(dat)))
np.fill_diagonal(icc_mat, 0)

# sort
sorted_mat = icc_mat[indsort,indsort.T]
sorted_mat = sorted_mat[~nan_rows_template, :][:, ~nan_rows_template] # saved from FC mats above

# check this didnt produce any nans form negative values 
print(f"Number of negative values in sorted_mat: {np.sum(sorted_mat < 0)}")
#sorted_mat[sorted_mat < 0] = 0

# plot as SD instead of raw variance
sorted_mat = np.sqrt(sorted_mat)

cmap_custom = plt.cm.YlGnBu

plt.figure(figsize=(7, 7))
plt.imshow(sorted_mat, origin='lower', cmap=cmap_custom, vmin=0, vmax=0.3)
cbar = plt.colorbar(fraction=0.046)
plt.show()

file2save = outdir / 'plots' / f"BW_{dataset}_{feature}.png"
file2save.parent.mkdir(parents=True, exist_ok=True)
print(f'saving: {file2save}')
plt.savefig(f'{file2save}', dpi=180)
plt.close()

# save as pconn
print(pconn_data.shape)
print(icc_mat.shape)
img_path = outdir / 'plots' / 'img' / f"BW_{dataset}_{feature}.pconn.nii"
img_path.parent.mkdir(parents=True, exist_ok=True)

new_img = nb.Cifti2Image(icc_mat, header=pconn.header,
                         nifti_header=pconn.nifti_header)

new_img.to_filename(img_path)




# Within-subject variance
icc_mat = connectome.vec_to_sym_matrix(results['within_sub_var'],diagonal=np.repeat(np.nan,len(dat)))
np.fill_diagonal(icc_mat, 0)

# sort
sorted_mat = icc_mat[indsort,indsort.T]
sorted_mat = sorted_mat[~nan_rows_template, :][:, ~nan_rows_template] # saved from FC mats above

print(f"Number of negative values in sorted_mat: {np.sum(sorted_mat < 0)}")
#sorted_mat[sorted_mat < 0] = 0

# plot as SD instead of raw variance
sorted_mat = np.sqrt(sorted_mat)

cmap_custom = plt.cm.YlGnBu

within_sub_sd = np.sqrt(results['within_sub_var']) # currently repeating the SD calc, update but be mindful of diagonal and rows set to 0
cutoff = np.round(np.mean(within_sub_sd) + 2 * np.std(within_sub_sd), 3)
#cutoff = np.round(np.mean(results['within_sub_var']) + np.std(results['within_sub_var']) + np.std(results['within_sub_var']),3)

plt.figure(figsize=(7, 7))
#plt.imshow(icc_mat, origin='lower', cmap=cmap_custom)
plt.imshow(sorted_mat, origin='lower', cmap=cmap_custom, vmin=0, vmax=cutoff)
cbar = plt.colorbar(fraction=0.046)
plt.show()

file2save = outdir / 'plots' / f"WH_{dataset}_{feature}.png"
print(f'saving: {file2save}')
plt.savefig(f'{file2save}', dpi=180)
plt.close()

# save as pconn
print(pconn_data.shape)
print(icc_mat.shape)
img_path = outdir / 'plots' / 'img' / f"WH_{dataset}_{feature}.pconn.nii"
img_path.parent.mkdir(parents=True, exist_ok=True)

new_img = nb.Cifti2Image(icc_mat, header=pconn.header,
                         nifti_header=pconn.nifti_header)

new_img.to_filename(img_path)




# Intraclass correlation coefficient (ICC)
icc_mat = connectome.vec_to_sym_matrix(results['icc'],diagonal=np.repeat(np.nan,len(dat)))
np.fill_diagonal(icc_mat, 0)

# sort
sorted_mat = icc_mat[indsort,indsort.T]
sorted_mat = sorted_mat[~nan_rows_template, :][:, ~nan_rows_template] # saved from FC mats above

cmap_custom = plt.cm.YlGnBu

plt.figure(figsize=(7, 7))
plt.imshow(sorted_mat, origin='lower', cmap=cmap_custom, vmin=0, vmax=1)
cbar = plt.colorbar(fraction=0.046)
plt.show()

file2save = outdir / 'plots' / f"icc_{dataset}_{feature}.png"
print(f'saving: {file2save}')
plt.savefig(f'{file2save}', dpi=180)
plt.close()

# save as pconn
print(pconn_data.shape)
print(icc_mat.shape)
img_path = outdir / 'plots' / 'img' / f"icc_{dataset}_{feature}.pconn.nii"
img_path.parent.mkdir(parents=True, exist_ok=True)

new_img = nb.Cifti2Image(icc_mat, header=pconn.header,
                         nifti_header=pconn.nifti_header)

new_img.to_filename(img_path)
###############################


# Next steps:
# - think about other summary measures - look at the icc pconn to see what the distrib acutaly looks like. Presumably this will be different based on where in cortex one is.
