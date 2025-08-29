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
from tkinter import ttk
from tkinter import messagebox, scrolledtext, filedialog

from datetime import datetime, timedelta
from dateutil.parser import parse as date_parse
import xarray as xr
from pathlib import Path
import time

from .utils import list_dir, get_instrument_base_url
from .cdf_handler import load_cdf_from_url
from .core import CDAWebDownloader
from .codegen import generate_script



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
        
    TO DO:
    1) Add box-check button for selecting what var is time variable (or
       find better way to automatically infer it from cdflib?)
    2) Can sometimes have I/O mistakes at the end after downloading all the
        data... might be better to save first downloaded dataset as temp
        file or something and if see error occurs so that issues are caught
        early. If safe, then download everything after
    3) Handle data types (e.g. int16, float32) better - some are getting
       accidentally promoted (int32 -> float32) when merging leading to
       ballooning memory costs
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
        self.output_dir = None

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

        #tk.Button(self, text="Generate Python Script Only", command=self.generate_script_only).pack(pady=10)
        
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
            
        
    def estimate_merge_memory(
            self, 
            start_date: datetime, 
            end_date: datetime
    ) -> float:
        """
        Estimate memory (in GB) required for merging datasets.
        Uses reference file size from the sample dataset × number of days × 2.
        """
        days = (end_date - start_date).days + 1
        total_mb = days * self.reference_file_size * 2   # 2× factor for merging
        return total_mb / 1024.0   # return in GB
        


    def show_dataset_selector(
            self, 
            ds: xr.Dataset
    ):
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
    
        # RIGHT: Variables checklist with count, preview toggle, filter, and select/deselect listed
        checklist_container = tk.Frame(win)
        checklist_container.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        checklist_frame = tk.LabelFrame(checklist_container, text="Variables to Keep")
        checklist_frame.pack(fill='both', expand=True)
        
        # --- Header row: selected-count + preview toggle (affects LEFT preview) ---
        header = tk.Frame(checklist_frame)
        header.pack(fill='x', padx=5, pady=(6, 2))
        
        selected_count_label = tk.Label(header, text="0 variables selected")
        selected_count_label.pack(side='left')
        
        preview_selected_only_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            header, text="Preview selected only",
            variable=preview_selected_only_var
        ).pack(side='right')
        
        # --- Top controls: Filter + buttons ---
        top_controls = tk.Frame(checklist_frame)
        top_controls.pack(fill='x', padx=5, pady=5)
        
        tk.Label(top_controls, text="Filter:").pack(side='left')
        filter_var = tk.StringVar(value="")
        tk.Entry(top_controls, textvariable=filter_var, width=24).pack(side='left', padx=(6, 10))
        
        btn_select = tk.Button(top_controls, text="Select Listed")
        btn_deselect = tk.Button(top_controls, text="De-select Listed")
        btn_select.pack(side='left', padx=2)
        btn_deselect.pack(side='left', padx=2)
        
        # --- Scrollable area for the checkboxes (reuses your Canvas pattern) ---
        canvas = tk.Canvas(checklist_frame, height=400)
        scrollbar = tk.Scrollbar(checklist_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # --- State + logic ---
        all_vars = list(ds.data_vars)
        
        # Default: all DE-selected
        var_vars = {var: tk.BooleanVar(value=False) for var in all_vars}
        
        def update_selected_count_label():
            n = sum(1 for v in var_vars.values() if v.get())
            selected_count_label.config(text=f"{n} variable{'s' if n != 1 else ''} selected")
        
        def update_preview():
            # LEFT preview reflects either the full ds or only selected variables
            if preview_selected_only_var.get():
                chosen = [k for k, v in var_vars.items() if v.get()]
                subset = ds[chosen] if chosen else xr.Dataset()
            else:
                subset = ds
            text.config(state=tk.NORMAL)
            text.delete("1.0", tk.END)
            text.insert(tk.END, str(subset))
            text.config(state=tk.DISABLED)
        
        def update_ui():
            update_selected_count_label()
            update_preview()
        
        # Keep references to visible (filtered) widgets
        check_widgets = {}  # var_name -> Checkbutton
        
        def rebuild_checklist():
            # Clear existing
            for w in scrollable_frame.winfo_children():
                w.destroy()
            check_widgets.clear()
        
            q = filter_var.get().strip().lower()
            for var in all_vars:
                if q and q not in var.lower():
                    continue
                cb = tk.Checkbutton(
                    scrollable_frame, text=var, variable=var_vars[var],
                    anchor='w', justify='left', command=update_ui  # update when toggled
                )
                cb.pack(fill='x', anchor='w')
                check_widgets[var] = cb
        
            # Keep UI in sync after rebuilding (e.g., after filter change)
            update_ui()
        
        def select_listed():
            for var in check_widgets.keys():
                var_vars[var].set(True)
            update_ui()
        
        def deselect_listed():
            for var in check_widgets.keys():
                var_vars[var].set(False)
            update_ui()
        
        btn_select.configure(command=select_listed)
        btn_deselect.configure(command=deselect_listed)
        filter_var.trace_add("write", lambda *_: rebuild_checklist())
        preview_selected_only_var.trace_add("write", lambda *_: update_ui())
        
        # Initial render
        rebuild_checklist()
        
        # --- Bottom: Next button (unchanged API, now reads var_vars) ---
        def go_to_step2():
            self.selected_variables = [k for k, v in var_vars.items() if v.get()]
            if not self.selected_variables:
                messagebox.showwarning("Selection Error", "Select at least one variable.")
                return
            win.destroy()
            self.show_dtype_selection_window()
        
        tk.Button(checklist_container, text="Next: Choose Data Types", command=go_to_step2).pack(pady=10)
        
    
    def _collect_dependent_coords(
            ds: xr.Dataset, 
            selected_vars: list[str]
    ) -> list[str]:
        """
        Return a sorted list of coordinate names that the selected variables
        *actually* depend on. We include:
          - all coords attached to each selected DataArray (da.coords)
          - index coords for each dimension (if present in ds.coords)
    
        This avoids pulling in unrelated coords from the dataset.
        """
        deps = set()
        for v in selected_vars:
            if v not in ds:
                continue
            da = ds[v]
            # coords explicitly attached to the DataArray
            for c in da.coords:
                if c in ds.coords:
                    deps.add(c)
            # index coords for dimensions (e.g., 'Epoch', 'time', energy axes, etc.)
            for d in da.dims:
                if d in ds.coords:
                    deps.add(d)
        # stable order
        return sorted(deps)
    
    
    
    def show_dtype_selection_window(self):
        """
        Opens a window allowing the user to override default dtypes for their
        previously selected variables, while previewing the subsetted dataset.
    
        Displays:
          - LEFT: Text preview of xarray.Dataset subset to the selected variables.
          - RIGHT: A scrollable list of variables with radio buttons to select new dtypes,
                   grouped as "Data Variables" and "Coordinates".
          - TOP: A warning block about dtype precision/overflow limits.
    
        Stores results in `self.selected_dtypes`.
        """
        if self.ds_sample is None or not self.selected_variables:
            messagebox.showerror("Error", "No dataset or variables selected.")
            return
    
        win = tk.Toplevel(self)
        win.title("Step 2: Choose Data Types")
        win.geometry("1100x750")
    
        # -------------------------------
        # Layout container: left + right
        # -------------------------------
        container = tk.Frame(win)
        container.pack(fill="both", expand=True, padx=10, pady=10)
    
        # -------------------------------
        # LEFT PANEL: Dataset preview
        # -------------------------------
        preview_frame = tk.Frame(container, width=450)
        preview_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
    
        tk.Label(preview_frame, text="Dataset Preview (selected variables only):",
                 font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
    
        text_preview = scrolledtext.ScrolledText(preview_frame, wrap=tk.WORD, width=60)
        text_preview.pack(fill="both", expand=True, padx=5, pady=5)
    
        # Subset the dataset to only selected variables for the preview
        subset_ds = self.ds_sample[self.selected_variables]
        text_preview.insert(tk.END, str(subset_ds))
        text_preview.config(state=tk.DISABLED)
    
        # -------------------------------
        # RIGHT PANEL: Dtype selection
        # -------------------------------
        right_panel = tk.Frame(container)
        right_panel.pack(side="right", fill="both", expand=True)
    
        # Warning text block
        tk.Label(
            right_panel,
            text=(
                "Select desired data types for each variable (defaults are pre-selected).\n"
                "DATA TYPE LIMITS:\n"
                "⚠ Float32 can overflow near ~10^38 (≈log10=38.53).\n"
                "⚠ Float64 can overflow near ~10^308 (≈log10=308.25).\n"
                "⚠ Int32 range: ±2.1e9.\n"
                "⚠ Int64 range: ±9.2e18."
            ),
            justify="left",
            fg="orange"
        ).pack(anchor="w", pady=(0, 5))
    
        # -------- Single scrollable area for BOTH groups (only one scrollbar) --------
        canvas = tk.Canvas(right_panel, height=500)
        scrollbar = tk.Scrollbar(right_panel, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
    
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
    
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
        # -------------------------------
        # Dtype options and storage
        # -------------------------------
        dtype_options = ["float32", "float64", "int32", "int64"]
        self.selected_dtypes = {}
    
        # Prepare group lists
        data_vars = list(self.selected_variables)
        coords = CDAWebGUI._collect_dependent_coords(self.ds_sample, self.selected_variables)
    
        # Compute a pleasant width for the name column; use fixed font for neat alignment
        all_names = data_vars + coords if (data_vars or coords) else ["(none)"]
        max_name_len = max(len(n) for n in all_names)
        name_col_chars = min(max_name_len + 2, 48)  # cap so it doesn't get huge
    
        def render_group(parent, title, names):
            """Render one group (Data Variables / Coordinates) as a tidy grid without column headers."""
            if not names:
                return
        
            group = tk.LabelFrame(parent, text=title, padx=8, pady=6)
            group.pack(fill="x", expand=True, pady=(10, 6))
        
            # Grid: col 0 = name, cols 1..4 = radio buttons (with text labels)
            group.grid_columnconfigure(0, weight=1)  # name column stretches a bit
            for c in range(1, 5):
                group.grid_columnconfigure(c, weight=0, minsize=84)  # keep columns aligned
        
            name_font = ("TkFixedFont", 16)
        
            # Rows (no header row)
            r = 0
            for name in names:
                if name not in self.ds_sample:
                    continue
        
                current_dtype = str(self.ds_sample[name].dtype)
                default_choice = current_dtype if current_dtype in dtype_options else "float32"
                self.selected_dtypes[name] = tk.StringVar(value=default_choice)
        
                tk.Label(
                    group, text=name, font=name_font, anchor="w", width=name_col_chars
                ).grid(row=r, column=0, sticky="w", padx=(2, 6), pady=3)
        
                # Radio buttons with their own text (float32/float64/int32/int64)
                for c, dtype in enumerate(dtype_options, start=1):
                    tk.Radiobutton(
                        group, text=dtype, value=dtype, variable=self.selected_dtypes[name]
                    ).grid(row=r, column=c, sticky="w", padx=6, pady=3)
        
                r += 1
    
        # Render groups inside the one scrollable frame
        render_group(scrollable_frame, "Data Variables", data_vars)
        render_group(scrollable_frame, "Coordinates", coords)
    
        # -------------------------------
        # NEXT BUTTON
        # -------------------------------
        def proceed():
            # Convert Tk variables into a plain dict
            self.selected_dtypes = {k: v.get() for k, v in self.selected_dtypes.items()}
            win.destroy()
            self.show_date_range_window()
    
        tk.Button(win, text="Next: Choose Date Range", command=proceed).pack(pady=10)
    

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

        """
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
        """

        def next_step():
            s = date_parse(start_entry.get())
            e = date_parse(end_entry.get())
            if e < s:
                messagebox.showerror("Date Error", "End date is before start date.")
            self.date_range = (s, e)
            if self.date_range is None:
                messagebox.showwarning("Estimate First", "Please estimate first.")
                return
            win.destroy()
            self.show_merge_option_window()

        #tk.Button(win, text="Estimate", command=estimate).pack(pady=5)
        tk.Button(win, text="Next: Specify if merging", command=next_step).pack(pady=10)
    
    
    def show_merge_option_window(self):
        """
        Asks the user if they want to merge datasets after downloading.
        Also provides a memory usage estimate.
        """
        s, e = self.date_range
        est_gb = self.estimate_merge_memory(s, e)
    
        win = tk.Toplevel(self)
        win.title("Merge Option")
        win.geometry("400x200")
    
        tk.Label(
            win,
            text=(
                f"Merging datasets requires about 2x the memory of all "
                f"datasets to be loaded (or roughly, ~{self.reference_file_size:.2f} "
                f"MB * {(e-s).days+1} days * ~2 = ~{est_gb:.2f} GB of RAM).\n"
                "Do you want to merge all datasets after downloading?"
            ),
            justify="left", wraplength=350
        ).pack(pady=10)
    
        def choose_merge(do_merge: bool):
            self.merge_after_download = do_merge
            win.destroy()
            self.show_local_save_dialog()
    
        tk.Button(
            win, 
            text="Yes, Merge", 
            command=lambda: choose_merge(True)).pack(side="left", padx=20, pady=20
        )
        tk.Button(
            win, 
            text="No, Keep Separate", 
            command=lambda: choose_merge(False)).pack(side="right", padx=20, pady=20
        )
    

                                                      
    def show_local_save_dialog(self):
        """
        Prompts the user for a folder to save the downloaded NetCDF files.
        Then asks if they want to generate a script before starting the download.
        """
        messagebox.showinfo(
            "Select Save Folder",
            "Please choose the folder where downloaded NetCDF files will be saved."
        )
        self.output_dir = filedialog.askdirectory()
        if not self.output_dir:
            return
    
        # After choosing output folder, ask if they want a script
        if messagebox.askyesno("Generate Script", "Do you want to generate a script for this configuration?"):
            output_path = filedialog.asksaveasfilename(
                title="Choose Where to Save Script",
                defaultextension=".py",
                filetypes=[("Python Files", "*.py")]
            )
            if output_path:
                code = generate_script(
                    base_url=self.base_url,
                    start_date=self.date_range[0],
                    end_date=self.date_range[1],
                    variables=self.selected_variables,
                    output_dir="{self.output_dir}",   # placeholder
                    dtypes=self.selected_dtypes
                )
                with open(output_path, "w") as f:
                    f.write(code)
                messagebox.showinfo("Script Saved", f"Script written to: {output_path}")
    
        # Start the download (script written or not)
        self.download_and_merge_cdfs()



    def download_and_merge_cdfs(self):
        """
        Download and save selected variables over the specified date range,
        showing a live progress bar as files are downloaded. Optionally merges
        the datasets afterward if the user selected that option.
    
        Workflow:
        1. Opens a progress window with a label and ttk.Progressbar.
        2. Calls `download_and_save_multiple_cdfs` with a progress_callback
           that updates the progress bar and label after each file is saved.
        3. After downloads finish:
            - If merging was requested, merges all .nc files and saves a merged dataset.
            - Otherwise, simply informs the user where the .nc files were saved.
        4. Closes the progress window and exits gracefully.
        """
        try:
            s, e = self.date_range
            total_days = (e - s).days + 1
    
            # Progress window
            prog_win = tk.Toplevel(self)
            prog_win.title("Downloading...")
            prog_label = tk.Label(prog_win, text=f"Downloaded 0 / {total_days} files")
            prog_label.pack(pady=10)
    
            progress = ttk.Progressbar(prog_win, length=300, mode="determinate")
            progress.pack(pady=10)
            progress["maximum"] = total_days
            progress["value"] = 0
            prog_win.update()
    
            # record start time for ETA calculation
            start_time = time.perf_counter()
    
            # --- Wrapper to update progress each iteration ---
            def progress_callback(done, total):
                elapsed = time.perf_counter() - start_time
                avg_time = elapsed / done if done > 0 else 0
                remaining = avg_time * (total - done)
    
                # format ETA
                mins, secs = divmod(int(remaining), 60)
                hrs, mins = divmod(mins, 60)
                if hrs > 0:
                    eta_str = f"ETA: {hrs:d}h {mins:02d}m {secs:02d}s"
                else:
                    eta_str = f"ETA: {mins:02d}m {secs:02d}s"
    
                # update GUI
                progress["maximum"] = total
                progress["value"] = done
                prog_label.config(text=f"Downloaded {done} / {total} files   {eta_str}")
                prog_win.update()
    
            # Perform downloads with progress updates
            cdf_folder = self.downloader.download_and_save_multiple_cdfs(
                start_date=s,
                end_date=e,
                selected_variables=self.selected_variables,
                output_dir=self.output_dir,
                dtypes=self.selected_dtypes,
                progress_callback=progress_callback
            )
    
            # Merge if requested
            if self.merge_after_download:
                merged_ds_path = self.downloader.merge_downloaded_datasets(cdf_folder)
                prog_win.destroy()
                messagebox.showinfo("Merged dataset saved",
                                    f"Saved to: {merged_ds_path}")
                print(f"Final merged dataset saved at {merged_ds_path}")
            else:
                prog_win.destroy()
                messagebox.showinfo("Download Complete", f"Saved .nc files to {cdf_folder}")
    
            # Graceful exit
            self.quit()
            self.destroy()
    
        except Exception as e:
            try:
                prog_win.destroy()
            except Exception:
                pass
            messagebox.showerror("Download Error", str(e))

