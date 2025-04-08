
import os
from pathlib import Path
import subprocess

__all__ = ['fetch_data']

def fetch_data(dataset, sub, ses, files):
    """
    Fetch data from an S3 location based on the specified dataset, subjects, sessions, and file names.
    
    Args:
        dataset (str): The name of the dataset.
        sub (list): A list of subject IDs.
        ses (list): A list of session IDs.
        files (list): A list of file names.

    Returns:
        list: A list of paths to the fetched files.
    """


    # examples:
    # location: 's3://btc-r01prelim/processed/xcpd/sub-412001/xcp_d/sub-412001/ses-1/func/'
    # files: sub-412001_ses-1_task-restMENORDICrmnoisevols_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii'
    #        sub-412001_ses-1_task-restMENORDICrmnoisevols_space-fsLR_atlas-4S956Parcels_measure-pearsoncorrelation_conmat.tsv
    #        sub-412001_ses-1_task-restMENORDICrmnoisevols_space-fsLR_atlas-4S956Parcels_timeseries.tsv
    #        sub-412001_ses-1_task-restMENORDICrmnoisevols_run-05_space-fsLR_den-91k_reho.dscalar.nii


    if dataset == 'r01prelim':
        proc_string = 'task-restMENORDICrmnoisevols'

    # Prep
    wd = os.getcwd()
    wd = Path(os.path.dirname(wd))
    out = wd / 'input'

    # List to store all fetched file paths
    fetched_files = []

    for sub_i in sub:
        print(f'Sub: {sub_i}')

        outdir = out / f'sub-{sub_i}'
        # Create the directory if it doesn't exist
        outdir.mkdir(parents=True, exist_ok=True)
        
        for ses_i in ses:
            print(f'Session: {ses_i}')

            outfile = outdir / ses_i
            # Create the directory if it doesn't exist
            outfile.mkdir(parents=True, exist_ok=True)

            for f_i in files:
                print(f'File: {f_i}')

                # Define file name and S3 location
                sub_file = f'sub-{sub_i}_{ses_i}_{proc_string}_{f_i}'
                s3_loc =  f's3://btc-{dataset}/processed/xcpd/sub-{sub_i}/xcp_d/sub-412001/{ses_i}/func/{sub_file}'
                file_in = outfile / f'{sub_file}'

                # Fetch data using subprocess to execute the fetch_data.sh script
                output = subprocess.check_output(['./fetch_data.sh',str(s3_loc),str(file_in)])
                print(f'{output}')

                # Save the file path to the list
                fetched_files.append(str(file_in))

    # Return the list of all fetched file paths
    return fetched_files, bare_file_name



