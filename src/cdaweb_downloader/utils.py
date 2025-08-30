"""
Utility functions for directory parsing, date extraction, and download size estimation.
"""



import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
import re



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