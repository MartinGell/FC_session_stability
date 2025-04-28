
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
dataset = 'HCP_YA' # 'subpop', 'MSC', HCPtrt_cneuro
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

        # model = MixedLM.from_formula(f'{str(col)} ~ 1', groups='subject_id', data=data)
        # rslt = model.fit(method=["bfgs"])

        # # extract variances and calc icc
        # between_sub_var = rslt.cov_re.iloc[0, 0].astype(np.float16)
        # within_sub_var = rslt.scale.astype(np.float16)
        # icc = between_sub_var / (between_sub_var + within_sub_var)
        
    #     return {
    #         'column': col,
    #         'between_sub_var': between_sub_var,
    #         'within_sub_var': within_sub_var,
    #         'icc': icc,
    #         'error': None
    #     }

    # except Exception as e:
    #     return {
    #         'column': col,
    #         'between_sub_var': 0,
    #         'within_sub_var': 0,
    #         'icc': 0,
    #         'error': str(e)
    #     }
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
icc_mat = connectome.vec_to_sym_matrix(upper,diagonal=np.repeat(np.nan,len(dat)))
np.fill_diagonal(icc_mat, 1)

cmap_custom = plt.cm.RdBu_r

plt.figure(figsize=(7, 7))
plt.imshow(icc_mat, origin='lower', cmap=cmap_custom, vmin=-1, vmax=1)
cbar = plt.colorbar(fraction=0.046)
plt.show()

file2save = outdir / 'plots' / f"EXAMPLE_SUBJECT_{dataset}_{feature}.png"
file2save.parent.mkdir(parents=True, exist_ok=True)
print(f'saving: {file2save}')
plt.savefig(f'{file2save}', dpi=180)
plt.close()



icc_mat = connectome.vec_to_sym_matrix(results['between_sub_var'],diagonal=np.repeat(np.nan,len(dat)))
np.fill_diagonal(icc_mat, 0)

cmap_custom = plt.cm.YlGnBu

plt.figure(figsize=(7, 7))
plt.imshow(icc_mat, origin='lower', cmap=cmap_custom, vmin=0, vmax=0.05)
cbar = plt.colorbar(fraction=0.046)
plt.show()

file2save = outdir / 'plots' / f"BW_{dataset}_{feature}.png"
print(f'saving: {file2save}')
plt.savefig(f'{file2save}', dpi=180)
plt.close()


# TEST -> scalling BW by icc
MSC_res = pd.read_csv(f'/home/btervocl/shared/projects/martin_FC_stability/res/MSC/MSC_icc_variances_Glasser.csv')

scaled_BW = MSC_res['icc'] * results['between_sub_var']

icc_mat = connectome.vec_to_sym_matrix(scaled_BW,diagonal=np.repeat(np.nan,len(dat)))
np.fill_diagonal(icc_mat, 0)

cmap_custom = plt.cm.YlGnBu

plt.figure(figsize=(7, 7))
plt.imshow(icc_mat, origin='lower', cmap=cmap_custom, vmin=0, vmax=0.05)
cbar = plt.colorbar(fraction=0.046)
plt.show()

file2save = outdir / 'plots' / f"SCALED_BW_{dataset}_by_MSCicc_{feature}.png"
print(f'saving: {file2save}')
plt.savefig(f'{file2save}', dpi=180)
plt.close()

# TEST -> scalling BW by WV
scaled_BW = results['between_sub_var'] - MSC_res['within_sub_var']

icc_mat = connectome.vec_to_sym_matrix(scaled_BW,diagonal=np.repeat(np.nan,len(dat)))
np.fill_diagonal(icc_mat, 0)

cmap_custom = plt.cm.YlGnBu

plt.figure(figsize=(7, 7))
plt.imshow(icc_mat, origin='lower', cmap=cmap_custom, vmin=0, vmax=0.05)
cbar = plt.colorbar(fraction=0.046)
plt.show()

file2save = outdir / 'plots' / f"BW_{dataset}_with_MSCwv_subtracted_{feature}.png"
print(f'saving: {file2save}')
plt.savefig(f'{file2save}', dpi=180)
plt.close()