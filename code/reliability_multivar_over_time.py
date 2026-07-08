
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
from functions.i2c2 import I2C2


#warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore", category=FutureWarning)
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

# amount of data to calculate ICC for
minutes_to_sample = [5, 10, 15, 20, 25]

# Which feature and dataset (folder) to use
feature = 'Gordon' # '4S1056Parcels' 'Glasser'
_dataset = 'subpop_UMN' # 'subpop_WashU25', 'MSC25', HCPtrt_cneuro, MSC_4runs subpop_UMN25 adultcontrols25
####################################### 


# Prep folder locations
wd = os.getcwd()
wd = Path(os.path.dirname(wd))

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


# Calcuate over dataset with different amounts of data
Res = []
for minute_i in minutes_to_sample:
    print(f'\n\nCalculating I2C2 for {minute_i} minutes of data')

    dataset = f'{_dataset}{str(minute_i)}'

    # Update locations
    indir = wd / 'data' / dataset
    outdir = wd / 'res' / dataset
    outdir.mkdir(parents=True, exist_ok=True)

    # Get all subjects data locations
    subs = glob.glob(f'{indir}/*')
    file_paths = glob.glob(f"{indir}/sub-*/ses-*/*{feature}*.pconn.nii")

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

        dat = np.clip(dat, -0.999999, 0.999999) # shouldnt be necessary, avoids nans
        dat_z = np.arctanh(dat)  # Fisher z-transform

        # save upper triangle
        print(dat_z.shape)
        upper = connectome.sym_matrix_to_vec(dat_z, discard_diagonal = True)
        
        # save
        data.append([sub_id, ses_id, *upper])

    cols = ['subject_id', 'session_id'] + [f"conn_{i}" for i in range(len(data[0]) - 2)]
    df = pd.DataFrame(data, columns=cols)

    # drop nans as i2c2 cant handle them (on purpose)
    print(f"Data shape: {df.shape}")
    df = df.dropna(axis=1)
    print(f"Data shape post NaN removal: {df.shape}")

    # now calculate image ICC
    y = df.drop(columns=['subject_id', 'session_id']).values

    res = I2C2(
        y=y,                 # (n_obs, ~70000) stacked, vectorized FC edges — one row per subject-session
        id=df["subject_id"].values,
        visit=df["session_id"].values,
        symmetric=False,      # default; cheaper, and already handles your unbalanced 3-vs-4-session design correctly
        twoway=True,          # remove session-specific mean in addition to overall mean — standard for test-retest FC reliability, guards against scan-order/familiarization effects shared across subjects
        demean=True,
        truncate=True,        # avoid reporting a nonsensical negative ICC when true reliability is near zero (likely with n=15)
        return_demean=False,  # you don't need the residual matrix back unless you're diagnosing something
    )

    # save
    Res.append([minute_i, np.round(res['lambda'], 3)]) 

# Save
results_df = pd.DataFrame(Res, columns=['minutes_sampled', 'I2C2'])

file2save = f'{outdir}/{dataset}_I2C2_{feature}.csv'
print(f'saving: {file2save}')
results_df.to_csv(file2save, index=False)

