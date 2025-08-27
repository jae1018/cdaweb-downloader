"""
Core logic for downloading and merging CDF files from CDAWeb.

Notes to self for later implementation:
1) Would eventually like automatically detect time coordinate(s) and align
   datasets based on that. For datasets with just 1 time dim, will probably
   be pretty easy, however some missions have multiple time coordinates
   in their cdfs (such as THEMIS, which records at a variety of resolutions).
   Moreover, sometimes the time dimensions do not have consistent names!
   (My experience was when looking at reduced (~3s) and full (96s/384s)
   that sometimes the reduced time name was 'record1' and other times
   it was 'record2'! ).
2) When implementing (1), be aware the cdflib.cdf_to_xarray ca only make
   data vars, coords, and dims based on data present with the given cdf.
   So if a particular cdf has bad data (such that much or all of it may
   be missing), then the corresponding coords and dims for such data may
   also not be present. Downstream xarray functions like concat, join, merge
   may not work (as intended, if at all) if some of those coords / dims are
   missing! It will likely be necessary to look over a list of downlaoded
   datasets and consider the "union" of all coords and dims , then remove
   datasets from the list that do not match this list of coords / dims.
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
        url: str,
        selected_variables: list[str],
        output_dir: Path,
        dtypes: dict[str, str] | None = None  # <-- NEW: optional dict of {var_name: dtype_str}
    ) -> None:
        """
        Downloads a single cdf from CDAWeb, filters it based on the
        selected_variables parameters, applies optional dtype casting,
        and then saves it to output_folder (using the same filename as the original file).
        
        PARAMETERS
        ----------
        url : str
            Exact link to the cdf to download.
        selected_variables : list of strs
            List of data variable names in the cdf to keep.
        output_dir : Path
            Path object to desired output folder for the cdf.
        dtypes : dict, optional
            Mapping from variable name → dtype string (e.g. {"var1": "float32"}).
            If None, defaults to existing data types.
    
        RETURNS
        -------
        None
        """
    
        print(f"Downloading {url}")
    
        # Load the dataset from the provided CDF URL
        ds, _ = load_cdf_from_url(url)
    
        # Keep only the requested variables
        subset = subset_dataset(ds, selected_variables)
    
        # -------------------------------
        # NEW: Apply custom dtype casting
        # -------------------------------
        if dtypes:
            print("Applying user-selected dtypes to variables and coordinates...")
            for name, dtype in dtypes.items():
                # Data variables
                if name in subset.data_vars:
                    current_dtype = str(subset[name].dtype)
                    try:
                        subset[name] = subset[name].astype(dtype)
                        print(f"  ✔ var {name}: {current_dtype} → {dtype}")
                    except Exception as e:
                        print(f"  ⚠ WARNING: failed to cast var '{name}' ({current_dtype} → {dtype}): {e}")
                # Coordinates
                elif name in subset.coords:
                    current_dtype = str(subset.coords[name].dtype)
                    try:
                        subset = subset.assign_coords({name: subset.coords[name].astype(dtype)})
                        print(f"  ✔ coord {name}: {current_dtype} → {dtype}")
                    except Exception as e:
                        print(f"  ⚠ WARNING: failed to cast coord '{name}' ({current_dtype} → {dtype}): {e}")
                else:
                    print(f"  ⚠ Skipping '{name}' — not found among vars or coords in subset.")
        else:
            print("No custom dtypes provided — using default data types.")
    
        # Collapse attributes into JSON-safe strings (avoids NetCDF serialization issues)
        subset = collapse_all_attrs_to_json(subset)
    
        """
        # OLD ATTR STRIP LOGIC (kept for reference)
        for var in subset.data_vars:
            subset[var].attrs.pop("units", None)
        for var in subset.coords:
            subset[var].attrs.pop("units", None)
        subset.attrs
        """
    
        # Save CDF as NetCDF with the same filename from URL but new extension
        filepath = output_dir / Path(url).name
        filepath = filepath.parent / filepath.name.replace(filepath.suffix, '.nc')
    
        # Final save step
        subset.to_netcdf(filepath)
        print(f"Saved dataset at {filepath}")
        
        

    def download_and_save_multiple_cdfs(
            self, 
            start_date : datetime | str, 
            end_date   : datetime | str, 
            selected_variables : list[str],
            dtypes : dict[str, str] = None,
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
                                out_dir,
                                dtypes = dtypes
                    )

                # ... and print exception if fails
                except Exception as e:
                    print(f"Failed to access or download from {year_url}: {e}")

        #if not datasets:
        #    raise ValueError("No valid datasets found for the given range.")

        return out_dir