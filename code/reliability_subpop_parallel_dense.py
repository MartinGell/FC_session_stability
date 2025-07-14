
import numpy as np
import pandas as pd
import glob
import h5py
import warnings
import sys
import os
from pathlib import Path

import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import time
start_time = time.time()

# Suppress convergence warnings from MixedLM
# warnings.filterwarnings("ignore", category=ConvergenceWarning)
# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=ConvergenceWarning)
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses


### User-specified parameters ###
# Input/output directories and file paths
feature = 'denoised_bold'
indir = '/home/btervocl/shared/projects/martin_FC_stability/data/subpop'
outdir = '/home/btervocl/shared/projects/martin_FC_stability/res/subpop'
file_paths = glob.glob(f"{indir}/sub-*/ses-*/*{feature}*.h5")

# Chunking parameters
n_jobs = 120  #  maximum number of jobs

# For 4B rows in a dconn, you might want to process only 40M rows at a time.
start_row = 1_750_000_001          # Starting row index 
max_rows = 150_000_000  # Maximum number of rows to process in this run (None to process all rows) 
# ---------------------------------------------------




# Determine dataset dimensions from the first file
with h5py.File(file_paths[0], 'r') as f:
    # Assume that the dataset of interest is the first key in the file.
    dataset_name = list(f.keys())[0]
    total_connections_in_file = f[dataset_name].shape[0]

# close hdf5 file
f.close()

# Define the end row based on max_rows, if specified
if max_rows is not None:
    end_row = min(start_row + max_rows, total_connections_in_file)
else:
    end_row = total_connections_in_file

# Total number of connections (rows) we will process in this run
total_connections_to_process = end_row - start_row
#chunk_size = total_connections_to_process/n_jobs
#chunk_size = int(chunk_size)
chunk_boundaries = np.linspace(start_row, end_row, num=n_jobs + 1, dtype=int)

print(f"Processing connections from {start_row} to {end_row - 1} (total {total_connections_to_process}), in {n_jobs} chunks.")


# Compute ICC for a given column safely
def compute_icc_safe(data, col):
    '''
    Compute the intraclass correlation coefficient (ICC) for a given column
    using a random intercept model. Skips fitting if NaNs are present.
    Falls back to 'powell' optimizer if 'bfgs' fails to converge.

    Returns a dictionary with model estimates and error status.

    Parameters
    ----------
    data : DataFrame
        The DataFrame containing the data to compute the ICC for.
    col : str
        The name of the column to compute the ICC for.
    
    Returns
    -------
    dict
        A dictionary containing the column name, between-subject variance, within-
        subject variance, ICC, and any error that occurred during computation.
    '''
    try:
        # Early exit: skip if NaNs in column
        if data[col].isna().any():
            return {
                'column': col,
                'between_sub_var': 0,
                'within_sub_var': 0,
                'icc': 0,
                'optimizer': 'Error',
                'error': 'NaNs in column'
            }

        # First try with BFGS
        model = MixedLM.from_formula(f'{col} ~ 1', groups='subject_id', data=data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            rslt = model.fit(method="bfgs")

        optimizer_used = "bfgs"

        # If not converged, try Powell
        if not rslt.converged:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                rslt = model.fit(method="powell")
            optimizer_used = "powell"

        # Final convergence and variance estimates
        converged = rslt.converged
        between_sub_var = np.float32(rslt.cov_re.iloc[0, 0])
        within_sub_var = np.float32(rslt.scale)
        icc = between_sub_var / (between_sub_var + within_sub_var)

        return {
            'column': col,
            'between_sub_var': between_sub_var,
            'within_sub_var': within_sub_var,
            'icc': icc,
            'optimizer': optimizer_used,
            'error': None
        }

    except Exception as e:
        return {
            'column': col,
            'between_sub_var': 0,
            'within_sub_var': 0,
            'icc': 0,
            'optimizer': 'Error',
            'error': str(e)
        }


# Process a single chunk: load only the specified slice from each file
def process_chunk(chunk_index):
    '''
    Process a single chunk of data, loading only the specified slice from each file.
    This function is meant to be called in parallel by joblib. It loads the data
    for the specified slice, computes the ICC for each column, and returns the results.

    Parameters
    ----------
    chunk_index : int
        The index of the chunk to process. This determines the slice of data to load.
    
    Returns
    -------
    list
        A list of dictionaries containing the ICC results for each column in the chunk.
    '''

    # Determine the slice for this chunk
    chunk_start = chunk_boundaries[chunk_index]
    chunk_end = chunk_boundaries[chunk_index + 1]
    print(f"Processing chunk {chunk_index + 1}/{n_jobs}: connections {chunk_start} to {chunk_end - 1}")

    data = []
    for file_i in sorted(file_paths):
        # Extract subject and session IDs from the file path
        parts = file_i.split("/")
        sub_id = parts[-3].split("-")[1]
        ses_id = parts[-2]
        
        # Open the HDF5 file and load only the desired slice
        with h5py.File(file_i, 'r') as f:
            dataset = f[dataset_name]
            chunk_data = dataset[chunk_start:chunk_end]
        
        # Unpack the chunk data so that each element becomes a separate column
        data.append([sub_id, ses_id, *chunk_data])
    
    # Create DataFrame columns that reflect the absolute connection indices
    cols = ['subject_id', 'session_id'] + [f"conn_{i}" for i in range(chunk_start, chunk_end)]
    df_chunk = pd.DataFrame(data, columns=cols)
    
    # Compute ICC for each connectivity column in the chunk
    results = []
    for col in df_chunk.columns:
        if col in ['subject_id', 'session_id']:
            continue
        subset_df = df_chunk[['subject_id', 'session_id', col]]
        result = compute_icc_safe(subset_df, col)
        results.append(result)
    return results
# ---------------------------------------------------



### Actually analyse data ###
# Parallelize over chunks (each chunk loads its own slice from all files)
all_chunk_results = Parallel(n_jobs=n_jobs)(
    delayed(process_chunk)(i) for i in range(n_jobs)
)

# Flatten the list of results from all chunks
results_list = [result for chunk in all_chunk_results for result in chunk]
results_df = pd.DataFrame(results_list)

# Optionally, reorder results to match the original connectivity order
original_order = [f"conn_{i}" for i in range(start_row, end_row)]
results_df['column'] = pd.Categorical(results_df['column'], categories=original_order, ordered=True)
results_df = results_df.sort_values('column').reset_index(drop=True)

print(results_df.head())

# Print how many rows failed to converge on bfgs:
print(f"\nNumber of rows that failed to converge with bfgs:")
print(sum(results_df['optimizer'] == 'powell'))

# Save the results to a dhf5 file
print("Saving results to HDF5 file...")
file_out = f"{outdir}/mat_parts/icc_results_rows_{start_row}_to_{end_row}.h5"
Path(file_out).parent.mkdir(parents=True, exist_ok=True)

results = results_df[['column', 'between_sub_var', 'within_sub_var', 'icc', 'optimizer']]
results_df['optimizer'] = results_df['optimizer'].astype(str)

# Save the entire results DataFrame
results_df.to_hdf(file_out, key='results', mode='w', format='table')
print(f"{file_out}")

# Save individual arrays as separate HDF5 files
icc_file = f"{outdir}/mat_parts/icc_rows_{start_row}_to_{end_row}.h5"
between_sub_var_file = f"{outdir}/mat_parts/between_sub_var_rows_{start_row}_to_{end_row}.h5"
within_sub_var_file = f"{outdir}/mat_parts/within_sub_var_rows_{start_row}_to_{end_row}.h5"

connection_strings = results_df['column'].astype(str).values
dt = h5py.string_dtype(encoding='utf-8')  # Variable-length UTF-8 strings

with h5py.File(icc_file, 'w') as h5file:
    h5file.create_dataset('icc', data=results_df['icc'].values)
    h5file.create_dataset('connection', data=connection_strings, dtype=dt)  
print(f"Saved ICC to {icc_file}")

with h5py.File(between_sub_var_file, 'w') as h5file:
    h5file.create_dataset('between_sub_var', data=results_df['between_sub_var'].values)
    h5file.create_dataset('connection', data=connection_strings, dtype=dt)  
print(f"Saved between_sub_var to {between_sub_var_file}")

with h5py.File(within_sub_var_file, 'w') as h5file:
    h5file.create_dataset('within_sub_var', data=results_df['within_sub_var'].values)
    h5file.create_dataset('connection', data=connection_strings, dtype=dt)  
print(f"Saved within_sub_var to {within_sub_var_file}")

# check if the output directory exists, if not create it
# Path(f'{file_out}.csv').parent.mkdir(parents=True, exist_ok=True)

# results_df.to_csv(f'{file_out}.csv', index=False)

# Save errors to a CSV file
### this should all go into a separate folder in mat_parts called logs:
error_log = results_df[results_df['error'].notnull()]
error_file = f"{outdir}/icc_error_log_rows_{start_row}_to_{end_row}.csv"
error_log[['column', 'error']].to_csv(error_file, index=False)
print(f"Saving errors...")
print(error_file)

# and called hist:
# Save histograms for quick viewing
axes = results_df.hist(bins=50, figsize=(10, 8))
plt.savefig(f'{outdir}/subpop_results_histograms_{feature}_rows_{start_row}_to_{end_row}.png')
plt.close()

elapsed = time.time() - start_time
print(f"\nFinished in {elapsed:.2f} seconds")
