
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
