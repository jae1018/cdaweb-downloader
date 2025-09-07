"""
Utility functions for directory parsing, date extraction, and download size estimation.
"""



import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
import re
import numpy as np



def crawl_for_cdfs(url: str, start_date, end_date) -> list[tuple[str, str]]:
    """
    Recursively descend CDAWeb directories until .cdf files are found.
    Collects only those with dates in [start_date, end_date].
    """
    entries = list_dir(url)
    cdf_files = []

    # Partition: files vs subdirs
    subdirs = []
    for name, full_url in entries:
        if name.lower().endswith(".cdf"):
            file_date = extract_date_from_filename(name)
            if file_date and start_date <= file_date <= end_date:
                cdf_files.append((name, full_url))
        elif name.endswith("/"):
            subdirs.append(full_url)

    # If we already found .cdf files here, stop recursion
    if cdf_files:
        return cdf_files

    # Otherwise, go deeper
    for sub in subdirs:
        cdf_files.extend(crawl_for_cdfs(sub, start_date, end_date))

    return cdf_files



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
    """
    List the directory at url using BeautifulSoup
    
    Note that in the CDAWeb directory, the first 5 hyperlinks listed will
    not correspond to cdfs (add more info about that later) - so just skip
    the first 5!
    """
    
    # gotta be a better way to do this - but there's these links at the top
    # that should be ignored
    bad_links = [
        '?C=N;O=D',
        '?C=M;O=A',
        '?C=S;O=A',
        "Parent Directory"
    ]
    
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    entries = []
    for link in soup.find_all('a'):        
        
        # confirm that link does not contain substr involving any of bad
        # link fragments above
        bad_link_found = False
        for bad_link in bad_links:
            if bad_link in str(link):
                bad_link_found = True
                break
        if bad_link_found:
            continue
        
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