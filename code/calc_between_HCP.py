
import numpy as np
import pandas as pd
import glob
import warnings
import time
import traceback
import os
import sys
from pathlib import Path


import matplotlib.pyplot as plt

import nibabel as nb
from nilearn import connectome
from joblib import Parallel, delayed


#warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore", category=FutureWarning)
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses


feature = 'Gordon' # '4S1056Parcels'
dataset = 'HCP_YA25'
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

# initialise
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

    dat = dat.astype(np.float16)

    print(dat.shape)
    upper = connectome.sym_matrix_to_vec(dat, discard_diagonal = True)
    
    # save
    data.append([sub_id, ses_id, *upper])

cols = ['subject_id', 'session_id'] + [f"conn_{i}" for i in range(len(data[0]) - 2)]
df = pd.DataFrame(data, columns=cols)

def compute_icc_safe(data, col):
    try:
        # fit the mixed-effects model
        between_sub_var = data[col].var(ddof=1)

        return {
            'column': col,
            'between_sub_var': between_sub_var,
            'error': None
        }

    except Exception as e:
        return {
            'column': col,
            'between_sub_var': 0,
            'error': str(e)
        }


# Batch computation for parallel processing 
def compute_icc_batch(df, columns):
    results = []
    for col in columns:
        subset_df = df[['subject_id', 'session_id', col]]
        results.append(compute_icc_safe(subset_df, col))
    return results

# Create batches of columns
column_batches = np.array_split(
    [col for col in df.columns if col not in ['subject_id', 'session_id']], 50
)

Res = Parallel(n_jobs=50)(
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
results = results_df[['column', 'between_sub_var']]
error_log = results_df[results_df['error'].notnull()]
error_dir = outdir / f'{dataset}_icc_variances_{feature}_error_log.csv'
error_dir.parent.mkdir(parents=True, exist_ok=True)
error_log[['column', 'error']].to_csv(error_dir, index=False)

print(results.describe())

# save - possibly switch to datatable
file2save = f'{outdir}/{dataset}_BW_variances_{feature}.csv'
print(f'saving: {file2save}')
results.to_csv(file2save, index=False)


# Save histograms for quick viewing
axes = results.hist(bins=50, figsize=(10, 8))
plt.savefig(f'{outdir}/{dataset}_results_histograms_{feature}.png')
plt.close()
print(f'saving: {f'{outdir}/{dataset}_results_histograms_{feature}.png'}')


###########################
# Create a mask of nans across all subjects and sessions
file_paths = glob.glob(f"{wd}/data/{feature}_nan_rows/*.txt")
all_nan_rows = []
for file_i in sorted(file_paths):
    nan_rows = np.loadtxt(file_i, dtype=bool)
    all_nan_rows.append(nan_rows)

all_nan_rows = pd.DataFrame(all_nan_rows)
template_nan_rows = np.any(all_nan_rows, axis=0)

# Apply the mask to the results
mat = connectome.vec_to_sym_matrix(upper,diagonal=np.repeat(np.nan,len(dat)))
np.fill_diagonal(mat, 1)

cmap_custom = plt.cm.RdBu_r

plt.figure(figsize=(7, 7))
plt.imshow(mat, origin='lower', cmap=cmap_custom, vmin=-1, vmax=1)
cbar = plt.colorbar(fraction=0.046)
plt.show()

file2save = outdir / 'plots' / f"EXAMPLE_SUBJECT_{dataset}_{feature}.png"
file2save.parent.mkdir(parents=True, exist_ok=True)
print(f'saving: {file2save}')
plt.savefig(f'{file2save}', dpi=180)
plt.close()



mat = connectome.vec_to_sym_matrix(results['between_sub_var'],diagonal=np.repeat(np.nan,len(dat)))
np.fill_diagonal(mat, 0)

sorted_mat = mat[indsort,indsort.T]
sorted_mat = sorted_mat[~template_nan_rows, :][:, ~template_nan_rows]

cmap_custom = plt.cm.YlGnBu

plt.figure(figsize=(7, 7))
plt.imshow(sorted_mat, origin='lower', cmap=cmap_custom, vmin=0, vmax=0.05)
cbar = plt.colorbar(fraction=0.046)
plt.show()

file2save = outdir / 'plots' / f"BW_{dataset}_{feature}.png"
print(f'saving: {file2save}')
plt.savefig(f'{file2save}', dpi=180)
plt.close()


# Save sorted cleaned data
clean_data_outdir = wd / 'res' / 'cleaned_variances_reliability'
clean_data_outdir.mkdir(parents=True, exist_ok=True)

d = [*connectome.sym_matrix_to_vec(sorted_mat, discard_diagonal = True)]
cols = ['between_sub_var']

clean_df = pd.DataFrame(d)
clean_df = clean_df.T
clean_df.columns = cols
clean_df.to_csv(f'{clean_data_outdir}/BW_variances_{dataset}_{feature}_clean_{sorted_mat.shape[0]}_rois_left.csv', index=False)


# Clean histogram
axes = clean_df.hist(bins=100, figsize=(10, 8))
file2save = f'{clean_data_outdir}/results_histograms_{dataset}_{feature}_clean.png'
plt.savefig(file2save, dpi=300)
plt.close()
print(file2save)



# # TEST -> scalling BW by icc
# MSC_res = pd.read_csv(f'/home/btervocl/shared/projects/martin_FC_stability/res/MSC/MSC_icc_variances_Glasser.csv')

# scaled_BW = MSC_res['icc'] * results['between_sub_var']

# mat = connectome.vec_to_sym_matrix(scaled_BW,diagonal=np.repeat(np.nan,len(dat)))
# np.fill_diagonal(mat, 0)

# cmap_custom = plt.cm.YlGnBu

# plt.figure(figsize=(7, 7))
# plt.imshow(mat, origin='lower', cmap=cmap_custom, vmin=0, vmax=0.05)
# cbar = plt.colorbar(fraction=0.046)
# plt.show()

# file2save = outdir / 'plots' / f"SCALED_BW_{dataset}_by_MSCicc_{feature}.png"
# print(f'saving: {file2save}')
# plt.savefig(f'{file2save}', dpi=180)
# plt.close()

# # TEST -> scalling BW by WV
# scaled_BW = results['between_sub_var'] - MSC_res['within_sub_var']

# mat = connectome.vec_to_sym_matrix(scaled_BW,diagonal=np.repeat(np.nan,len(dat)))
# np.fill_diagonal(mat, 0)

# cmap_custom = plt.cm.YlGnBu

# plt.figure(figsize=(7, 7))
# plt.imshow(mat, origin='lower', cmap=cmap_custom, vmin=0, vmax=0.05)
# cbar = plt.colorbar(fraction=0.046)
# plt.show()

# file2save = outdir / 'plots' / f"BW_{dataset}_with_MSCwv_subtracted_{feature}.png"
# print(f'saving: {file2save}')
# plt.savefig(f'{file2save}', dpi=180)
# plt.close()