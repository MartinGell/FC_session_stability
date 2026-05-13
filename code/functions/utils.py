
import pandas as pd
import ast
import numpy as np
import math

from dataclasses import dataclass


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
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

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


# for saving run info
@dataclass
class RunInfo:
    run_name: str
    keep_mask: np.ndarray  # 1=keep, 0=drop
    usable_minutes: float
    TR: float

def minutes_to_frames(minutes, TR):
    """Convert minutes to frames given TR in seconds."""
    return int(math.ceil((minutes * 60.0) / TR))

def trim_mask_to_minutes(mask, TR, minutes_to_keep):
    if minutes_to_keep <= 0:
        return np.zeros_like(mask, dtype=int)
    # don't request more frames than exist
    k_frames = min(np.count_nonzero(mask), minutes_to_frames(minutes_to_keep, TR))
    one_idx = np.flatnonzero(mask == 1)
    if len(one_idx) < k_frames:
        raise ValueError("Requested more frames than available in mask.")
    cut = one_idx[k_frames - 1]
    trimmed = mask.copy()
    trimmed[cut + 1:] = 0
    return trimmed

def allocate_minutes_with_grace(usable_minutes, requested_total_minutes, TR, leaway_TR=3.0):
    """
    Allocate minutes across runs with a 3*TR grace threshold.

    - If sum(usable) + 3*TR < requested_total -> not enough data (returns None)
    - Else allocate to target = min(requested_total, sum(usable))
    """
    total_usable = float(sum(usable_minutes))
    grace_minutes = (leaway_TR * TR) / 60.0

    # Fail if total usable + grace < target
    if total_usable + grace_minutes < requested_total_minutes:
        return None, requested_total_minutes, total_usable, grace_minutes

    target = min(requested_total_minutes, total_usable)
    n = len(usable_minutes)
    base = target / n
    assigned = [min(base, u) for u in usable_minutes]
    deficit = target - sum(assigned)

    # Greedy fill to reach target
    while deficit > 1e-9:
        updated = False
        for i in range(n):
            spare = usable_minutes[i] - assigned[i]
            if spare <= 0:
                continue
            inc = min(spare, deficit)
            assigned[i] += inc
            deficit -= inc
            updated = True
            if deficit <= 1e-9:
                break
        if not updated:
            break  # done

    return assigned, target, total_usable, grace_minutes
