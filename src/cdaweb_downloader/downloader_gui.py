"""
Main Tkinter GUI class that drives the CDAWeb Downloader app.

Allows the user to:
- Browse the CDAWeb file tree
- Load a sample .CDF file
- Select variables to extract
- Specify a date range
- Download and merge data into a single NetCDF file
- Optionally generate a standalone script to replicate the download
"""

import tkinter as tk
from tkinter import messagebox, scrolledtext, filedialog
from datetime import datetime, timedelta
from dateutil.parser import parse as date_parse
from .utils import list_dir, get_instrument_base_url
from .cdf_handler import load_cdf_from_url
from .core import CDAWebDownloader
from .codegen import generate_script
import xarray as xr


def run_gui():
    """
    Launches the CDAWeb GUI application.
    """
    app = CDAWebGUI()
    app.mainloop()


class CDAWebGUI(tk.Tk):
    
    """
    The main application window for the CDAWeb Downloader GUI.

    Attributes:
        DEFAULT_URL (str): Root URL for CDAWeb data.
        base_url (str): Current base path for downloads.
        current_url (str): URL the file browser is currently showing.
        entries (list): List of (name, url) pairs for display.
        selected_variables (list): List of selected variable names from sample CDF.
        reference_file_size (float): Size (in MB) of the sample file to estimate total download.
        path_var (tk.StringVar): Tkinter variable bound to the current URL path bar.
        date_range (tuple): Start and end datetime objects for download.
        file_url_sample (str): URL to the currently selected sample CDF.
        ds_sample (xarray.Dataset): Parsed sample CDF dataset.
    """
    
    DEFAULT_URL = "https://cdaweb.gsfc.nasa.gov/pub/data/"

    def __init__(self):
        """
        Initializes the GUI window and sets up the layout and default values.
        """
        super().__init__()
        self.title("CDAWeb Downloader")
        self.geometry("1100x750")

        self.base_url = self.DEFAULT_URL
        self.downloader = CDAWebDownloader(self.base_url)

        self.current_url = self.base_url
        self.entries = []
        self.selected_variables = []
        self.reference_file_size = 0
        self.path_var = tk.StringVar(value=self.base_url)
        self.date_range = None
        self.file_url_sample = None
        self.ds_sample = None

        self.build_widgets()
        self.update_listing()

    def build_widgets(self):
        """
        Constructs the GUI layout — path bar, file browser, and script button.
        """
        path_frame = tk.Frame(self)
        path_frame.pack(fill='x', padx=10, pady=5)
        
        # Back button
        tk.Button(path_frame, text="← Back", command=self.navigate_up).pack(side='left')

        tk.Label(path_frame, text="Current URL:").pack(side='left')
        tk.Entry(path_frame, textvariable=self.path_var, width=100).pack(side='left', fill='x', expand=True)

        list_frame = tk.Frame(self)
        list_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.listbox = tk.Listbox(list_frame)
        self.listbox.pack(side='left', fill='both', expand=True)
        self.listbox.bind('<Double-1>', self.on_double_click)

        scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        scrollbar.pack(side='right', fill='y')
        self.listbox.config(yscrollcommand=scrollbar.set)

        tk.Button(self, text="Generate Python Script Only", command=self.generate_script_only).pack(pady=10)
        
    def navigate_up(self):
        """
        Moves up one folder level in the current_url and refreshes the file list.
        """
        parts = self.current_url.rstrip('/').split('/')
        if len(parts) > 5:  # Prevent navigating above CDAWeb root
            self.current_url = '/'.join(parts[:-1]) + '/'
            self.path_var.set(self.current_url)
            self.update_listing()

    def update_listing(self):
        """
        Fetches and displays directory contents at `self.current_url` in the Listbox.
        """
        try:
            self.listbox.delete(0, tk.END)
            self.entries = list_dir(self.current_url)
            for name, url in self.entries:
                self.listbox.insert(tk.END, name)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load directory:\n{e}")

    def on_double_click(self, event):
        """
        Handles double-click behavior on the file list:
        - If a directory, navigates into it
        - If a `.cdf` file, loads it as a sample
    
        Args:
            event: Tkinter double-click event
        """
        index = self.listbox.curselection()[0]
        name, url = self.entries[index]
        if name.endswith('/'):
            self.current_url = url
            self.path_var.set(self.current_url)
            self.update_listing()
        elif name.endswith('.cdf'):
            self.file_url_sample = url
            self.load_and_display_cdf(url)
            self.base_url = get_instrument_base_url(url)   # NEW
            self.downloader = CDAWebDownloader(self.base_url)  # REINIT with new base URL

    def load_and_display_cdf(self, url):
        """
        Loads the selected .CDF file and shows a variable selection window.
    
        Args:
            url (str): URL to the selected .cdf file
        """
        try:
            ds, size_mb = load_cdf_from_url(url)
            self.reference_file_size = size_mb
            self.ds_sample = ds
            self.show_dataset_selector(ds)
        except Exception as e:
            messagebox.showerror("CDF Error", str(e))

    def show_dataset_selector(self, ds):
        """
        Opens a window where users can select which variables to keep from the sample CDF.
    
        Args:
            ds (xarray.Dataset): The loaded sample dataset
        """
        win = tk.Toplevel(self)
        win.title("Step 1: Select Variables")
        win.geometry("1000x700")
    
        # LEFT: Text preview of xarray.Dataset
        text = scrolledtext.ScrolledText(win, wrap=tk.WORD, width=60)
        text.pack(side='left', fill='both', expand=True, padx=5)
        text.insert(tk.END, str(ds))
        text.config(state=tk.DISABLED)
    
        # RIGHT: Variables checklist inside a scrollable canvas
        checklist_container = tk.Frame(win)
        checklist_container.pack(side='right', fill='both', expand=False, padx=10, pady=10)
    
        checklist_frame = tk.LabelFrame(checklist_container, text="Variables to Keep")
        checklist_frame.pack(fill='both', expand=True)
    
        canvas = tk.Canvas(checklist_frame, height=400)
        scrollbar = tk.Scrollbar(checklist_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
    
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
    
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
    
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
        # Variable selection checkboxes
        var_vars = {var: tk.BooleanVar(value=True) for var in ds.data_vars}
        for var, v in var_vars.items():
            tk.Checkbutton(scrollable_frame, text=var, variable=v).pack(anchor='w')
    
        # Button BELOW checklist
        def go_to_step2():
            self.selected_variables = [k for k, v in var_vars.items() if v.get()]
            if not self.selected_variables:
                messagebox.showwarning("Selection Error", "Select at least one variable.")
                return
            win.destroy()
            self.show_date_range_window()
    
        tk.Button(checklist_container, text="Next: Choose Date Range", command=go_to_step2).pack(pady=10)

    def show_date_range_window(self):
        """
        Opens a window to allow the user to specify a date range for data download.
    
        Also estimates total download size based on the sample file size and number of days.
        """
        win = tk.Toplevel(self)
        win.title("Step 2: Choose Date Range")
        win.geometry("400x300")

        tk.Label(win, text="Start Date (YYYY-MM-DD):").pack()
        start_entry = tk.Entry(win)
        start_entry.pack()

        tk.Label(win, text="End Date (YYYY-MM-DD):").pack()
        end_entry = tk.Entry(win)
        end_entry.pack()

        result_label = tk.Label(win, text="")
        result_label.pack()

        def estimate():
            try:
                s = date_parse(start_entry.get())
                e = date_parse(end_entry.get())
                if e < s:
                    raise ValueError("End before start")
                self.date_range = (s, e)
                days = (e - s).days + 1
                total = days * self.reference_file_size
                unit = "MB" if total < 1024 else "GB"
                if unit == "GB":
                    total /= 1024
                result_label.config(text=f"Estimated download: ~{total:.2f} {unit}")
            except Exception as e:
                messagebox.showerror("Date Error", str(e))

        def next_step():
            if self.date_range is None:
                messagebox.showwarning("Estimate First", "Please estimate first.")
                return
            win.destroy()
            self.show_local_save_dialog()

        tk.Button(win, text="Estimate", command=estimate).pack(pady=5)
        tk.Button(win, text="Next: Save", command=next_step).pack(pady=10)

    def show_local_save_dialog(self):
        """
        Prompts the user for a folder and filename to save the merged NetCDF file locally.
    
        Then initiates the download and merge process.
        """
        folder = filedialog.askdirectory(title="Choose Folder to Save .nc File")
        if not folder:
            return
        filename = filedialog.asksaveasfilename(
            title="Enter Output .nc File Name",
            defaultextension=".nc",
            filetypes=[("NetCDF Files", "*.nc")]
        )
        if not filename:
            return
        self.download_and_merge_cdfs(filename)

    def download_and_merge_cdfs(self, output_path):
        """
        Downloads and merges selected variables over the specified date range,
        then saves the result as a .nc file.
    
        Args:
            output_path (str): Filepath to save the merged NetCDF file
        """
        try:
            s, e = self.date_range
            ds = self.downloader.download_and_merge(s, e, self.selected_variables)
            ds.to_netcdf(output_path)
            messagebox.showinfo("Saved", f"Saved to: {output_path}")
        except Exception as e:
            messagebox.showerror("Download Error", str(e))

    def generate_script_only(self):
        """
        Prompts user for a location to save a Python script that replicates the current download config.
    
        The script will use the current base URL, selected date range, and chosen variables.
        """
        try:
            if self.date_range is None or not self.selected_variables or self.file_url_sample is None:
                messagebox.showwarning("Missing Info", "You must select a sample CDF, variables, and a date range first.")
                return

            output_path = filedialog.asksaveasfilename(
                title="Choose Where to Save Script",
                defaultextension=".py",
                filetypes=[("Python Files", "*.py")]
            )
            if not output_path:
                return

            code = generate_script(
                base_url=self.base_url,
                start_date=self.date_range[0],
                end_date=self.date_range[1],
                variables=self.selected_variables,
                output_path="<your_output_file.nc>"  # Placeholder the user can change
            )
            with open(output_path, "w") as f:
                f.write(code)
    
            messagebox.showinfo("Script Saved", f"Script written to: {output_path}")
        except Exception as e:
            messagebox.showerror("Script Generation Error", str(e))

