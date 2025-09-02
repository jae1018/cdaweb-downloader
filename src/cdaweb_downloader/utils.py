"""
Utility functions for directory parsing, date extraction, and download size estimation.
"""



import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
import re
import numpy as np



def is_numeric_dtype(arr) -> bool:
    """Return True if an xarray DataArray is numeric and castable."""
    kind = np.dtype(arr.dtype).kind
    return kind in {"i", "u", "f"}  # int, unsigned int, float



# might be useful later?
#def eligible_cast_dtypes(arr):
#    """Return allowed dtype options based on current dtype."""
#    if not is_numeric_dtype(arr):
#        return []  # nothing allowed
#    return ["float32", "float64", "int32", "int64"]



def list_dir(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    entries = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and href != '../':
            full_url = urljoin(url, href)
            entries.append((href, full_url))
    return entries



def extract_date_from_filename(filename):
    match = re.search(r"(\d{8})", filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d")
    return None



def get_instrument_base_url(file_url: str) -> str:
    """
    Given a full .cdf file URL, return the base instrument-level URL
    (i.e., the folder containing all year-based subfolders).

    Example:
        "https://.../mfi_h0/1998/ac_h0_mfi_19980101_v04.cdf"
        → "https://.../mfi_h0"
    """
    return "/".join(file_url.rstrip("/").split("/")[:-2])