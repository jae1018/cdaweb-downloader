"""
Core logic for downloading and merging CDF files from CDAWeb.
"""

from datetime import datetime
from dateutil.parser import parse as date_parse
from .cdf_handler import load_cdf_from_url, subset_dataset, merge_datasets, collapse_all_attrs_to_json
from .utils import list_dir, extract_date_from_filename
import xarray as xr
import re
from pathlib import Path

class CDAWebDownloader:
    
    def __init__(self, base_url: str):
        """
        Args:
            base_url (str): Instrument-level URL like:
                https://cdaweb.gsfc.nasa.gov/pub/data/ace/mag/level_2_cdaweb/mfi_h0
        """
        self.base_url = base_url.rstrip("/")
    
    def _download_and_save_single_cdf(
            self,
            url : str,
            selected_variables : list[str],
            output_dir      : Path
    ) -> None:
        
        """
        Downloads a single cdf from CDAWeb, filters it based on the
        selected_variables parameters, and then saved it to output_folder
        (using the same filename as the original file).
        
        PARAMETERS
        ----------
        url : str
            Exact link to the cdf to download
        selected_variables : list of strs
            List of data variable names in the cdf to keep
        output_dir : Path
            Path object to desired output folder for the cdf
        
        RETURNS
        -------
        None
        """
        
        print(f"Downloading {url}")
        ds, _ = load_cdf_from_url(url)
        subset = subset_dataset(ds, selected_variables)
        
        
        ## get rid of "units" in attrs (can cause issues with saving netcdf)
        # do that for each data var ...
        subset = collapse_all_attrs_to_json(subset)
        """
        for var in subset.data_vars:
            subset[var].attrs.pop("units", None)
        # ... as well as each data coord
        for var in subset.coords:
            subset[var].attrs.pop("units", None)
        subset.attrs 
        """
        
        # save cdf as netcdf with same filename from url
        filepath = output_dir / Path(url).name
        # and change ending from .cdf to .nc
        filepath = filepath.parent / filepath.name.replace(filepath.suffix,'.nc')
        print(f"Saved dataset at {filepath}")
        subset.to_netcdf(filepath)
        
        

    def download_and_save_multiple_cdfs(
            self, 
            start_date : datetime | str, 
            end_date   : datetime | str, 
            selected_variables : list[str],
            output_dir : str | None = None
    ) -> Path:
        """
        Downloads and merges .CDF files across multiple year folders
        if their dates fall within the given range.
        
        PARAMETERS
        ----------
        start_date : datetime.datetime (or str to be converted)
        end_date : datetime.datetime (or str to be converted)
        selected_variables : list of strs
            List of data variable names in the cdf to keep
        output_dir : Path
            Path object to desired output folder for the cdf
            
        RETURNS
        -------
        Path object to folder of saved netcdfs.
        """
        
        if isinstance(start_date, str): start_date = date_parse(start_date)
        if isinstance(end_date, str): end_date = date_parse(end_date)
        
        # create out_dir if does not exist
        out_dir = Path(output_dir) if output_dir else Path.cwd()
        out_dir = out_dir / "cached_cdaweb_netcdfs"
        out_dir.mkdir(parents=True, exist_ok=True)

        for year in range(start_date.year, end_date.year + 1):
            year_url = f"{self.base_url}/{year}/"
            for name, url in list_dir(year_url):
                
                # attempt to download file...        
                try:
                
                    # check that file is of type .cdf
                    if not name.endswith(".cdf"):
                        continue
                    
                    # check that file is within requested date range
                    file_date = extract_date_from_filename(name)
                    if file_date is None or not (start_date <= file_date <= end_date):
                        continue
                    
                    # download file
                    self._download_and_save_single_cdf(
                                url, 
                                selected_variables, 
                                out_dir
                    )

                # ... and print exception if fails
                except Exception as e:
                    print(f"Failed to access or download from {year_url}: {e}")

        #if not datasets:
        #    raise ValueError("No valid datasets found for the given range.")

        return out_dir