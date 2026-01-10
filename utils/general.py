import gdown
import os

def download_from_gdrive(file_id: str, output_path: str = None, overwrite: bool = False) -> str:
    """
    Download a file from Google Drive using its file ID.

    Parameters
    ----------
    file_id : str
        The Google Drive file ID (from the shareable link).
    output_path : str, optional
        Path to save the downloaded file. If None, uses the file's original name.
    overwrite : bool, optional
        If False and file exists, skips downloading.

    Returns
    -------
    str
        Path to the downloaded file.
    """
    url = f"https://drive.google.com/uc?id={file_id}"

    # Ensure target directory exists
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Handle existing file
    if output_path and not overwrite and os.path.exists(output_path):
        print(f"✅ File already exists: {output_path}")
        return output_path

    # Download file
    output_path = gdown.download(url=url, output=output_path, quiet=False)

    print(f"✅ Downloaded: {output_path}")
    return output_path