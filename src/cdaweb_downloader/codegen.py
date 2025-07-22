"""
codegen.py

Generates a short Python script that uses the CDAWebDownloader
class to download and merge CDF data from NASA CDAWeb.
"""

from datetime import datetime

def generate_script(base_url, start_date, end_date, variables, output_path):
    """
    Returns the string content of a Python script that downloads data using
    CDAWebDownloader and saves it to a NetCDF file.

    Parameters:
    - base_url (str): The base CDAWeb directory URL.
    - start_date (datetime): Start of the date range.
    - end_date (datetime): End of the date range.
    - variables (list[str]): Selected variable names.
    - output_path (str): Path to save merged NetCDF file.
    """
    return f"""\
from cdaweb_downloader.core import CDAWebDownloader

downloader = CDAWebDownloader("{base_url}")
ds = downloader.download_and_merge(
    start_date="{start_date.strftime('%Y-%m-%d')}",
    end_date="{end_date.strftime('%Y-%m-%d')}",
    selected_variables={variables}
)
ds.to_netcdf("{output_path}")
print("Saved merged dataset to: {output_path}")
"""
