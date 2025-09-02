"""
codegen.py

Generates a short Python script that uses the CDAWebDownloader
class to download and merge CDF data from NASA CDAWeb.
"""

from pathlib import Path
from datetime import datetime


def generate_script(
    base_url: Path,
    start_date: datetime,
    end_date: datetime,
    variables: list[str],
    output_dir: Path,
    dtypes: dict[str,str] | None = None,
    merge_after_download: bool = True
):
    
    """
    Returns the string content of a Python script that downloads data using
    CDAWebDownloader and optionally merges it into a single NetCDF file.

    Parameters
    ----------
    base_url : str
        The base CDAWeb directory URL.
    start_date : datetime
        Start of the date range.
    end_date : datetime
        End of the date range.
    variables : list[str]
        Selected variable names.
    output_dir : str
        Folder where NetCDF files will be saved.
    dtypes : dict, optional
        Mapping of variable name -> dtype (e.g. {"Bx": "float32"}). Defaults to None.
    merge_after_download : bool, optional
        If True, merge datasets into a single file (default=True).
    """

    # specify dtypes beforehand b/c they're optional
    dtype_arg = f",\n    dtypes={dtypes}" if dtypes else ""
    
    # Normalize output_dir to plain string
    output_dir_str = str(Path(output_dir).expanduser().resolve())

    # Always do the download
    script = f"""\
from pathlib import Path
from cdaweb_downloader.core import CDAWebDownloader

# Initialize the downloader
downloader = CDAWebDownloader("{base_url}")

out_folder = downloader.download_and_save_multiple_cdfs(
    start_date="{start_date.strftime('%Y-%m-%d')}",
    end_date="{end_date.strftime('%Y-%m-%d')}",
    selected_variables={variables}{dtype_arg},
    output_dir=Path("{output_dir_str}"),
    use_tqdm=True
)
"""

    # Only merge if requested
    if merge_after_download:
        script += f"""
merged_path = downloader.merge_downloaded_datasets(out_folder)
"""

    return script
