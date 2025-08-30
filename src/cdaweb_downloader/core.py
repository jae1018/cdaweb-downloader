"""
Core logic for downloading and merging CDF files from CDAWeb.

Notes to self for later implementation:
  1) When convertable data-types is shown for data vars and coords, it also
     offers to convert strs to numeric type - this should not be permitted!
     Also, be aware of datetime data types
  2) Have logger.py also absorb all the warnings spit out by cdflib in
     cdf_handler.py
  3) Use Dask in merge_downloaded_datasets so that don't require huge amounts
     of RAM to concat everything, just roughly the size of a single file
     (but this will likely increase the time to merge). Suppoooooosedly,
     it should be easy but we'll see later lol
  4) Modify output_dir in codegen.py so that the printed path for the user
     is based on Path rather than outright str - that way, it will be easier
     for Windows users (e.g. something like Path.home() / "Documents" / "project"
     instead of /Users/me/Documents/project )
"""

from datetime import datetime
from dateutil.parser import parse as date_parse
import xarray as xr
from pathlib import Path

from .cdf_handler import load_cdf_from_url, subset_dataset, collapse_all_attrs_to_json, clean_object_coords
from .utils import list_dir, extract_date_from_filename
from .merge import align_datasets_over_time_dims
from .logger import logger



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
        dtypes: dict[str, str] | None = None
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
    
        logger.info(f"Downloading {url}")
    
        # Load the dataset from the provided CDF URL
        ds, _ = load_cdf_from_url(url)
    
        # Keep only the requested variables
        subset = subset_dataset(ds, selected_variables)
    
    
        # -------------------------------
        # NEW: Apply custom dtype casting
        # -------------------------------
        if dtypes:
            logger.info("Applying user-selected dtypes to variables and coordinates...")
            for name, dtype in dtypes.items():
                
                # Data variables
                if name in subset.data_vars:
                    current_dtype = str(subset[name].dtype)
                    try:
                        subset[name] = subset[name].astype(dtype)
                        logger.info(f"  ✔ var {name}: {current_dtype} → {dtype}")
                    except Exception as e:
                        logger.warning(
                            f"  ⚠ Failed to cast var '{name}' ({current_dtype} → {dtype}): {e}"
                        )
                        
                # Coordinates
                elif name in subset.coords:
                    current_dtype = str(subset.coords[name].dtype)
                    try:
                        subset = subset.assign_coords({name: subset.coords[name].astype(dtype)})
                        logger.info(f"  ✔ coord {name}: {current_dtype} → {dtype}")
                    except Exception as e:
                        logger.warning(
                            f"  ⚠ Failed to cast coord '{name}' ({current_dtype} → {dtype}): {e}"
                        )
                
                # Non data-vars / coords
                else:
                    logger.warning(f"  ⚠ Skipping '{name}' — not found in dataset.")
                    
        else:
            logger.info("No custom dtypes provided — using default data types.")
    
    
        # Collapse attributes into JSON-safe strings (avoids NetCDF serialization issues)
        subset = collapse_all_attrs_to_json(subset)
    
        # Save CDF as NetCDF with the same filename from URL but new extension
        filepath = output_dir / Path(url).name
        filepath = filepath.parent / filepath.name.replace(filepath.suffix, '.nc')
    
        # Final save step
        subset.to_netcdf(filepath)
        logger.info(f"Saved dataset at {filepath}")
        
        

    def download_and_save_multiple_cdfs(
            self, 
            start_date: datetime | str, 
            end_date: datetime | str, 
            selected_variables: list[str],
            dtypes: dict[str, str] = None,
            output_dir: str | None = None,
            progress_callback = None,
            use_tqdm : bool = False
    ) -> Path:
        
        """
        Downloads multiple .CDF files from CDAWeb within the specified date range,
        saves them locally as NetCDF, and reports progress via an optional callback.
    
        Parameters
        ----------
        start_date : datetime.datetime or str
            Beginning of the date range (inclusive). If a string is given, it will
            be parsed with `dateutil.parser.parse`.
        end_date : datetime.datetime or str
            End of the date range (inclusive). If a string is given, it will
            be parsed with `dateutil.parser.parse`.
        selected_variables : list of str
            Names of variables to extract from each downloaded file.
        dtypes : dict[str, str], optional
            Mapping of variable name → dtype string (e.g. {"Bx": "float32"}).
            If None, the dataset is kept at its default dtypes.
        output_dir : str or Path, optional
            Base directory where files will be saved. If not provided, the current
            working directory is used. A subfolder named "cached_cdaweb_netcdfs"
            will be created inside it.
        progress_callback : callable, optional
            Function that receives `(completed, total)` as arguments after
            each file attempt (whether successful or failed). This allows a
            GUI progress bar or logger to be updated during long downloads.
        use_tqdm : bool, optional
            If True, shows a tqdm progress bar in the terminal (ignored if a 
            progress_callback is provided).
    
        Returns
        -------
        Path
            The directory containing the downloaded NetCDF files.
    
        Notes
        -----
        - Files that cannot be downloaded or parsed are skipped, but the progress
          count is still incremented to ensure user feedback is consistent.
        - This function does not merge datasets; merging should be performed
          downstream (e.g. with `merge_downloaded_datasets`).
        """
        
        if isinstance(start_date, str): 
            start_date = date_parse(start_date)
        if isinstance(end_date, str): 
            end_date = date_parse(end_date)
    
        # create out_dir if does not exist
        out_dir = Path(output_dir) if output_dir else Path.cwd()
        out_dir = out_dir / "cached_cdaweb_netcdfs"
        out_dir.mkdir(parents=True, exist_ok=True)
    
        # --- Collect all candidate files first ---
        file_list = []
        for year in range(start_date.year, end_date.year + 1):
            year_url = f"{self.base_url}/{year}/"
            for name, url in list_dir(year_url):
                if not name.endswith(".cdf"):
                    continue
                file_date = extract_date_from_filename(name)
                if file_date and (start_date <= file_date <= end_date):
                    file_list.append((name, url))
    
        total_files = len(file_list)
        completed = 0
        logger.info(f"Found {total_files} files to download.")
        
        # --- Setup tqdm if requested and no GUI callback ---
        pbar = None
        if use_tqdm and progress_callback is None:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=total_files, unit="file", desc="Downloading")
            except ImportError:
                logger.warning("tqdm not installed; proceeding without progress bar.")
    
        # --- Download loop ---
        for name, url in file_list:
            try:
                self._download_and_save_single_cdf(
                    url, 
                    selected_variables, 
                    out_dir,
                    dtypes=dtypes
                )
            except Exception as e:
                logger.error(f"Failed to download {url}: {e}")
    
            # Increment progress / tqdm bar if enabled
            completed += 1
            if progress_callback:
                progress_callback(completed, total_files)
            elif pbar:
                pbar.update(1)
        
        # close tqdm bar after for loop
        if pbar:
            pbar.close()
            
        logger.info(f"Downloaded files to: {out_dir}")
    
        return out_dir
    
    
    
    def merge_downloaded_datasets(self, folder: Path) -> Path:
        """
        Merge all downloaded .nc files in the given folder and save as merged_dataset.nc.
        
        Parameters
        ----------
        folder : Path
            Directory containing the .nc files from download_and_save_multiple_cdfs.
        
        Returns
        -------
        Path
            Path to the saved merged_dataset.nc
        """
        
        logger.info(f"Merging all NetCDF files in {folder}")
        
        # load datasets (sorted!)
        ds_list = []
        for f in sorted(folder.glob("*.nc")):
            # open with chunking
            ds = xr.open_dataset(f, engine="netcdf4", chunks={})
            # sanitize coords
            ds = clean_object_coords(ds)
            # save to list
            ds_list.append(ds)
        
        # align datasets over time
        final_ds = align_datasets_over_time_dims(ds_list)
        
        # rechunk afer merge
        final_ds = final_ds.chunk("auto")
        
        # save aligned and merged dataset into parent folder of cdf_folder
        merged_ds_path = folder.parent / 'merged_dataset.nc'
        
        # compute=True, engine='netcdf4' means that contents are stream
        # from original files to currently-generated cumulative file,
        # saving resources on RAM (all files don't need to be loaded at once!)
        final_ds.to_netcdf(merged_ds_path, compute=True, engine="netcdf4")
    
        logger.info(f"Merged dataset saved at {merged_ds_path}")
    
        return merged_ds_path