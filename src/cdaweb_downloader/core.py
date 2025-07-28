"""
Core logic for downloading and merging CDF files from CDAWeb.
"""

from datetime import datetime
from dateutil.parser import parse as date_parse
from .cdf_handler import load_cdf_from_url, subset_dataset, merge_datasets
from .utils import list_dir, extract_date_from_filename
import xarray as xr
import re

class CDAWebDownloader:
    def __init__(self, base_url: str):
        """
        Args:
            base_url (str): Instrument-level URL like:
                https://cdaweb.gsfc.nasa.gov/pub/data/ace/mag/level_2_cdaweb/mfi_h0
        """
        self.base_url = base_url.rstrip("/")

    def download_and_merge(
            self, 
            start_date : datetime | str, 
            end_date   : datetime | str, 
            selected_variables : list[str]
    ) -> xr.Dataset:
        """
        Downloads and merges .CDF files across multiple year folders
        if their dates fall within the given range.
        
        Parameters
        ----------
        start_date : datetime.datetime (or str to be converted)
        """
        datasets = []
        
        if isinstance(start_date, str): start_date = date_parse(start_date)
        if isinstance(end_date, str): end_date = date_parse(end_date)

        for year in range(start_date.year, end_date.year + 1):
            year_url = f"{self.base_url}/{year}/"
            try:
                for name, url in list_dir(year_url):
                    if not name.endswith(".cdf"):
                        continue
                    file_date = extract_date_from_filename(name)
                    if file_date is None or not (start_date <= file_date <= end_date):
                        continue
                    print(f"Downloading {url}")
                    ds, _ = load_cdf_from_url(url)
                    subset = subset_dataset(ds, selected_variables)
                    datasets.append(subset)
            except Exception as e:
                print(f"Failed to access or download from {year_url}: {e}")

        if not datasets:
            raise ValueError("No valid datasets found for the given range.")

        combined = merge_datasets(datasets)
        for var in combined.data_vars:
            combined[var].attrs.pop("units", None)
        return combined