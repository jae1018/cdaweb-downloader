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

