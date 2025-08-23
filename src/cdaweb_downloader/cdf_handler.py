"""
Functions for loading, subsetting, and merging CDF files into xarray datasets.
"""

from cdflib.xarray import cdf_to_xarray
import xarray as xr
import tempfile
import requests
import json

def collapse_all_attrs_to_json(ds: xr.Dataset) -> xr.Dataset:
    """
    Collapses all attributes (dataset, coordinates, and data variables)
    into a single JSON string to avoid NetCDF serialization issues.
    Non-serializable objects (e.g. NumPy datetimes) are stringified.
    """
    def safe_json(obj):
        try:
            return json.dumps(obj, default=str)
        except Exception:
            return json.dumps(str(obj))

    # Dataset-level attributes
    if ds.attrs:
        ds.attrs = {"_original_attrs": safe_json(ds.attrs)}

    # Coordinates
    for coord in ds.coords:
        if ds.coords[coord].attrs:
            ds.coords[coord].attrs = {
                "_original_attrs": safe_json(ds.coords[coord].attrs)
            }

    # Data variables
    for var in ds.data_vars:
        if ds[var].attrs:
            ds[var].attrs = {
                "_original_attrs": safe_json(ds[var].attrs)
            }

    return ds

def load_cdf_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.cdf') as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    return cdf_to_xarray(tmp_path, to_datetime=True, fillval_to_nan=True), len(response.content) / 1024**2  # size in MB

def subset_dataset(ds, variable_list):
    return ds[variable_list]

def merge_datasets(
        datasets : list[xr.Dataset]
) -> xr.Dataset:
    """
    Merges a list of xarray datasets based on the time dimension. Currently,
    time_dim is assumed to be one of ("Epoch", "time", "epoch")
    
    Parameters
    ----------
    datasets : list of xarray datasets
        List of xarray datasets to merge in time (and all should at minimum
        contain the same time_dim coordiante name)
    
    Returns
    -------
    xarray dataset
    
    TO DO:
    ------
    Better time_dim inferring
    """
    possible_time_strs = ("Epoch", "time", "epoch")
    time_dim = next((d for d in possible_time_strs if d in datasets[0].dims), "time")
    #dim = "Epoch" if "Epoch" in datasets[0].dims else "time"
    # Clean conflicting attrs (e.g. 'units') on time dimension
    for ds in datasets:
        #time_dim = "Epoch" if "Epoch" in ds.dims else "time"
        if time_dim in ds.coords:
            ds[time_dim].attrs = {}  # Clear potentially conflicting encoding attrs
    return xr.concat(datasets, dim=time_dim)
