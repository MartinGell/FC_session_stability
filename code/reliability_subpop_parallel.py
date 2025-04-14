
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
#from pymer4.models import Lmer
from statsmodels.regression.mixed_linear_model import MixedLM


#warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore", category=FutureWarning)
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses


feature = 'Glasser' # '4S1056Parcels'
dataset = 'MSC' # 'subpop'
####################################### 



# Prep folder locations
wd = os.getcwd()
wd = Path(os.path.dirname(wd))
indir = wd / 'data' / dataset
outdir = wd / 'res' / dataset
outdir.mkdir(parents=True, exist_ok=True)

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

    # new_img = nb.Cifti2Image(dat, header=nii.header,
    #                      nifti_header=nii.nifti_header)
    # new_img.to_filename(f'{indir}/sub-{sub_id}/{ses_id}/sub-{sub_id}_{ses_id}_task-restMENORDICtrimmed_space-fsLR_den-91k_desc-denoised_bold_FD_02_smoothed_2mm_rounded.dconn.nii')


    print(dat.shape)
    upper = connectome.sym_matrix_to_vec(dat, discard_diagonal = True)
    
    # save
    data.append([sub_id, ses_id, *upper])

cols = ['subject_id', 'session_id'] + [f"conn_{i}" for i in range(len(data[0]) - 2)]
df = pd.DataFrame(data, columns=cols)


# # Reliability
# def compute_icc(col):
#     data = df[['subject_id', 'session_id', col]]

#     model = MixedLM.from_formula(f'{str(col)} ~ 1', groups='subject_id', data=data)
#     rslt = model.fit(method=["bfgs"])

#     between_sub_var = rslt.cov_re.iloc[0, 0].astype(np.float16)
#     within_sub_var = rslt.scale.astype(np.float16)

#     icc = between_sub_var / (between_sub_var + within_sub_var)
    
#     return [between_sub_var, within_sub_var, icc]

# # Calculate
# Res = Parallel(n_jobs=-15)(delayed(compute_icc)(col_i) for col_i in df.columns if col_i not in ['subject_id', 'session_id'])
# cols = ['between_sub_var', 'within_sub_var', 'icc']
# results = pd.DataFrame(Res, columns=cols)

def compute_icc_safe(data, col):
    try:
        # fit the mixed-effects model
        model = MixedLM.from_formula(f'{str(col)} ~ 1', groups='subject_id', data=data)
        rslt = model.fit(method=["bfgs"])

        # extract variances and calc icc
        between_sub_var = rslt.cov_re.iloc[0, 0].astype(np.float16)
        within_sub_var = rslt.scale.astype(np.float16)
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

# Parallel computation
# Res = Parallel(n_jobs=-15)(
#     delayed(compute_icc_safe)(df[['subject_id', 'session_id', col]], col) for col in df.columns if col not in ['subject_id', 'session_id']
# )

# # Convert results to a DataFrame
# results_df = pd.DataFrame(Res)

# Save results and errors separately
results = results_df[['column', 'between_sub_var', 'within_sub_var', 'icc']]
error_log = results_df[results_df['error'].notnull()]
error_dir = outdir / f'{dataset}_icc_variances_{feature}_error_log.csv'
error_dir.parent.mkdir(parents=True, exist_ok=True)
error_log[['column', 'error']].to_csv(error_dir, index=False)

print(results.describe())

# save - possibly switch to datatable
file2save = f'{outdir}/{dataset}_icc_variances_{feature}.csv'
#file2save = f'{outdir}/{dataset}_icc_variances_TEST.csv'
print(f'saving: {file2save}')
results.to_csv(file2save, index=False)


# Save histograms for quick viewing
axes = results.hist(bins=50, figsize=(10, 8))
plt.savefig(f'{outdir}/{dataset}_results_histograms_{feature}.png')
plt.close()




###########################
icc_mat = connectome.vec_to_sym_matrix(upper,diagonal=np.repeat(np.nan,len(dat)))
np.fill_diagonal(icc_mat, 1)

cmap_custom = plt.cm.RdBu_r

plt.figure(figsize=(7, 7))
plt.imshow(icc_mat, origin='lower', cmap=cmap_custom, vmin=-1, vmax=1)
cbar = plt.colorbar(fraction=0.046)
plt.show()

#file2save = parts[10].split('task-')[1].split('_den')  # just want this: 'restMENORDICtrimmed_space-fsLR_seg-4S956Parcels'    ##### possibly just 4S956Parcels????
file2save = f"{outdir}/plots/EXAMPLE_SUBJECT_{feature}.png"
print(f'saving: {file2save}')
plt.savefig(f'{file2save}', dpi=180)
plt.close()



icc_mat = connectome.vec_to_sym_matrix(results['between_sub_var'],diagonal=np.repeat(np.nan,len(dat)))
np.fill_diagonal(icc_mat, 1)

cmap_custom = plt.cm.YlGnBu

plt.figure(figsize=(7, 7))
plt.imshow(icc_mat, origin='lower', cmap=cmap_custom, vmin=0, vmax=0.05)
cbar = plt.colorbar(fraction=0.046)
plt.show()

#file2save = parts[10].split('task-')[1].split('_den')  # just want this: 'restMENORDICtrimmed_space-fsLR_seg-4S956Parcels'    ##### possibly just 4S956Parcels????
file2save = f"{outdir}/plots/BW_{feature}.png"
print(f'saving: {file2save}')
plt.savefig(f'{file2save}', dpi=180)
plt.close()



icc_mat = connectome.vec_to_sym_matrix(results['within_sub_var'],diagonal=np.repeat(np.nan,len(dat)))
np.fill_diagonal(icc_mat, 1)

cmap_custom = plt.cm.YlGnBu

plt.figure(figsize=(7, 7))
plt.imshow(icc_mat, origin='lower', cmap=cmap_custom, vmin=0, vmax=0.01)
cbar = plt.colorbar(fraction=0.046)
plt.show()

#file2save = parts[10].split('task-')[1].split('_den')  # just want this: 'restMENORDICtrimmed_space-fsLR_seg-4S956Parcels'    ##### possibly just 4S956Parcels????
file2save = f"{outdir}/plots/WH_{feature}.png"
print(f'saving: {file2save}')
plt.savefig(f'{file2save}', dpi=180)
plt.close()



icc_mat = connectome.vec_to_sym_matrix(results['icc'],diagonal=np.repeat(np.nan,len(dat)))
np.fill_diagonal(icc_mat, 1)

cmap_custom = plt.cm.YlGnBu

plt.figure(figsize=(7, 7))
plt.imshow(icc_mat, origin='lower', cmap=cmap_custom, vmin=0, vmax=1)
cbar = plt.colorbar(fraction=0.046)
plt.show()

#file2save = parts[10].split('task-')[1].split('_den')  # just want this: 'restMENORDICtrimmed_space-fsLR_seg-4S956Parcels'    ##### possibly just 4S956Parcels????
file2save = f"{outdir}/plots/icc_{feature}.png"
print(f'saving: {file2save}')
plt.savefig(f'{file2save}', dpi=180)
plt.close()
###########################


# icc_mat = connectome.vec_to_sym_matrix(results['icc'],diagonal=np.repeat(np.nan,len(dat)))
# np.fill_diagonal(icc_mat, 1)


# # PLOT matrix
# atlas = pd.read_table('/Users/mgell/Work/SNR/text_files/atlas-4S956Parcels_dseg.tsv')
# indsort = atlas['network_label']

# #display = plotting.plot_matrix(m_weights_mat)
# sorted = icc_mat[indsort,indsort.T]

# cmap_custom = plt.cm.RdBu_r
# max_abs_val = np.nanmax(np.abs(sorted))

# plt.figure(figsize=(7, 7))
# plt.imshow(icc_mat, origin='lower', cmap=cmap_custom, vmin=-max_abs_val, vmax=max_abs_val)
# cbar = plt.colorbar(fraction=0.046)
# ticks = [-max_abs_val, -0.5 * max_abs_val, 0, 0.5 * max_abs_val, max_abs_val]
# cbar.set_ticks(ticks)
# cbar.ax.set_yticklabels(['{:.2g}'.format(t) for t in ticks]) # Format tick labels to two significant digits
# plt.show()

# file2save = file.split('/')[7].split('.')[0].split(f'_{model}_')[0]
# file2save = f"{out_plt}/{file2save}_mean_weights{transform}_all_models_{model}.png"
# print(f'saving: {file2save}')
# plt.savefig(f'{file2save}', dpi=180)
# plt.close()






# Next steps:
# - save icc as pconn.nii - note that there will be surfce and volume components?
# - median icc per ROI
# - think about other summary measures - look at the icc pconn to see what the distrib acutaly looks like. Presumably this will be different based on where in cortex one is.
