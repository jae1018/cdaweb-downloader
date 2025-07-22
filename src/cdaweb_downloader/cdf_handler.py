"""
Functions for loading, subsetting, and merging CDF files into xarray datasets.
"""

from cdflib.xarray import cdf_to_xarray
import xarray as xr
import tempfile
import requests

def load_cdf_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.cdf') as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    return cdf_to_xarray(tmp_path, to_datetime=True, fillval_to_nan=True), len(response.content) / 1024**2  # size in MB

def subset_dataset(ds, variable_list):
    return ds[variable_list]

def merge_datasets(datasets):
    dim = "Epoch" if "Epoch" in datasets[0].dims else "time"
    # Clean conflicting attrs (e.g. 'units') on time dimension
    for ds in datasets:
        time_dim = "Epoch" if "Epoch" in ds.dims else "time"
        if time_dim in ds.coords:
            ds[time_dim].attrs = {}  # Clear potentially conflicting encoding attrs
    return xr.concat(datasets, dim=dim)
