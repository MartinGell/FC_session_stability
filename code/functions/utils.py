
import pandas as pd
import ast


def filter_output(output: str) -> str:
    """
    Removes everything up to and including a predefined message from the output.

    Args:
        output (str): The full output to process.

    Returns:
        str: The remaining output after the ignored message.
    """
    # Message to ignore everything before and including
    ignore_up_to = (
        "Note: workbench works best on MSI's OpenOnDemand system (ondemand.msi.umn.edu).\nTrying to run it over X-forwarding from other systems will be unstable."
    )
    
    # Split the output around the ignore message
    parts = output.split(ignore_up_to, 1)
    return parts[1].strip() if len(parts) > 1 else output.strip()


def build_subject_session_run_map(csv_path):
    """
    Loads a CSV file containing subject, session, and run mappings, and returns
    a nested dictionary of the form:
    {
        subject_id: {
            session: [list of runs]
        },
        ...
    }
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}")

    # Clean whitespace
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    required_columns = {'subject_id', 'session', 'runs'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Convert 'runs' column from string to list
    def safe_parse_runs(run_str, idx):
        try:
            parsed = ast.literal_eval(run_str)
            if not isinstance(parsed, list):
                raise ValueError
            return parsed
        except Exception:
            raise ValueError(f"Malformed runs list at row {idx}: {run_str}")

    df['runs'] = [safe_parse_runs(r, i) for i, r in enumerate(df['runs'])]

    # Build nested dictionary
    sub_ses_run_map = {}
    for _, row in df.iterrows():
        sub = row['subject_id']
        ses = row['session']
        runs = row['runs']

        if sub not in sub_ses_run_map:
            sub_ses_run_map[sub] = {}

        if ses in sub_ses_run_map[sub]:
            print(f"Warning: Duplicate entry for subject {sub}, session {ses}. Overwriting.")
        
        sub_ses_run_map[sub][ses] = runs

    return sub_ses_run_map



# MSC
def MSC_build_subject_session_map(csv_path):
    """
    Loads a CSV file containing subject and session mappings, and returns
    a dictionary of the form:
    {
        subject_id: [list of sessions],
        ...
    }
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}")

    # Clean whitespace
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    required_columns = {'subject_id', 'session'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Build dictionary
    sub_ses_map = {}
    for _, row in df.iterrows():
        sub = row['subject_id']
        ses = row['session']

        if sub not in sub_ses_map:
            sub_ses_map[sub] = []

        sub_ses_map[sub].append(ses)

    return sub_ses_map
