"""
Core logic for downloading and merging CDF files from CDAWeb.

Notes to self for later implementation:
    
--- IMPORTANT ---
  1) Compressing the final merged dataset is reasonably possible but adds
     extra time based on compression levels for netcdf (for 10GB, could be
     like 30 mins to 1 hour but would compress to size of ~3 or 4 GB).
     Could infer time_dims from codes in merge.py, chunk over those, then
     specify the encoding dict in the .to_netcdf() function (a complevel of
     four [balanced between 1 and 9] would probably be the most sensible).
  2) Sometimes CDFs will record mis-aligned info between a data vari
     and a dim (e.g. a dim 'record0' has length 0 but a data_var that depends
     on it has length 1). In merge.py, should eventually make a function
     that chekcs the integrity of an entire dataset based on data-var /
     coord and dim alignment.
  3) Sometimes cdfs may rarely have non-unique times - in cases like this,
     it's probably better to just drop both.
  4) Some datasets (e.g. FGM data for MMS) have quite tight time res (~0.1s
     for survey data) - it may be necessary to introduce an avg-down scheme
     for such data sizes (~450GB for 10 years of data). Part of this should
     involve a new GUI window that asks the user for the desired time res
     and possibly also have them indicate which variable is the time (although
     might could just infer this from first dims along all the vars). This will
     also possibly require the user to specify some max and min values as well
     as quality-flag control.
  5) Update code so that it correctly interprets dates in files that have
     hour, minute, second info (having issue with mms fpi/dis-moms b/c they
     look like mms1_fpi_fast_l2_dis-moms_20151130000000_v3.4.0.nc)
    
                                                                   
--- SECONDARY ---
  1) Have logger.py also absorb all the warnings spit out by cdflib in
     cdf_handler.py
  2) After merging, delete folder of cached cdfs? Something to consider.
  3) Add option for categorical encoding for some float variables (but this
     would only really be useful in cases of energy bins or something, would
     be useless for things like mag, velocity, temp measurements, etc)
  4) Instead of just saying "dataset" etc in merge.py logger statements,
     better to just print the path to the offending file.
  5) Make scroll wheel work for variables-to-keep list in Step 1.
  6) Make a back button?
            
"""

from datetime import datetime
from dateutil.parser import parse as date_parse
import xarray as xr
from pathlib import Path

from .cdf_handler import load_cdf_from_url, subset_dataset, collapse_all_attrs_to_json, clean_object_coords
from .utils import is_numeric_dtype, crawl_for_cdfs
from .merge import align_datasets_over_time_dims
from .logger import logger



class CDAWebDownloader:
    
    
    
    def __init__(self, base_url: str):
        """
        Args:
            base_url (str): Instrument-level URL like:
                https://cdaweb.gsfc.nasa.gov/pub/data/ace/mag/level_2_cdaweb/mfi_h0
        """
        ##self.base_url = base_url.rstrip("/")
        # Normalize base_url to always end with '/'.
        # Ensures urljoin treats it as a directory (critical for MMS-like
        # directory structures e.g. YYYY/MM/{cdfs}).
        if not base_url.endswith("/"):
            base_url += "/"
        self.base_url = base_url
    
    
    
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
        # Apply custom dtype casting (numeric-only)
        # -------------------------------
        if dtypes:
            logger.info("Applying user-selected dtypes to variables and coordinates...")
            for name, dtype in dtypes.items():
    
                # Data variables
                if name in subset.data_vars:
                    arr = subset[name]
                    current_dtype = str(arr.dtype)
    
                    if not is_numeric_dtype(arr):
                        logger.warning(f"  Skipping var '{name}' — non-numeric dtype ({current_dtype})")
                        continue
    
                    try:
                        subset[name] = arr.astype(dtype)
                        logger.info(f"  var {name}: {current_dtype} → {dtype}")
                    except Exception as e:
                        logger.warning(
                            f"  Failed to cast var '{name}' ({current_dtype} → {dtype}): {e}"
                        )
    
                # Coordinates
                elif name in subset.coords:
                    arr = subset.coords[name]
                    current_dtype = str(arr.dtype)
    
                    if not is_numeric_dtype(arr):
                        logger.warning(f"  Skipping coord '{name}' — non-numeric dtype ({current_dtype})")
                        continue
    
                    try:
                        subset = subset.assign_coords({name: arr.astype(dtype)})
                        logger.info(f"  coord {name}: {current_dtype} → {dtype}")
                    except Exception as e:
                        logger.warning(
                            f"  Failed to cast coord '{name}' ({current_dtype} → {dtype}): {e}"
                        )
    
                # Non data-vars / coords
                else:
                    logger.warning(f"  Skipping '{name}' — not found in dataset.")
                    
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
        
        logger.info("Starting download...")
    
        # --- Collect all candidate files first ---
        file_list = crawl_for_cdfs(self.base_url, start_date, end_date)
    
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
        for idx, (name, url) in enumerate(file_list):
            try:
                logger.info(f"Attempting download for file {idx+1} / {len(file_list)}")
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