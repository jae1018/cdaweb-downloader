"""
main.py

Entry point for the cdaweb-downloader CLI tool.
It launches the GUI interface for browsing, selecting,
and downloading CDF data from NASA's CDAWeb.
"""

from .downloader_gui import run_gui

def main():
    run_gui()

if __name__ == "__main__":
    main()