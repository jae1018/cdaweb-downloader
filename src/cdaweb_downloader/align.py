# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np
import xarray as xr

from typing import Callable
from tqdm import tqdm



def filter_ds_by_qflag(
    ds: xr.Dataset,
    qflag_str: str,
    filt_func: Callable[[np.ndarray], np.ndarray],
) -> xr.Dataset:
    """
    Filter a dataset by masking with a quality flag.

    Instead of building a huge integer index array, this applies a mask
    lazily with .where(). Bad samples become NaN. This avoids exploding
    memory and is much faster with dask.
    
    Also, feel like this function should go in another module...

    PARAMETERS
    ----------
    ds : xr.Dataset
        Dataset containing variables and qflag.
    qflag_str : str
        Name of the quality flag variable in ds.
    filt_func : Callable[[np.ndarray], np.ndarray]
        Function mapping qflag values -> boolean mask
        (e.g., lambda q: q <= 1).

    RETURNS
    -------
    xr.Dataset
        Dataset with the same variables, but bad samples replaced with NaN.
    """
    mask = filt_func(ds[qflag_str])
    return ds.where(mask)



def _make_bin_edges_with_deltas(
        irreg_times : np.ndarray, 
        max_delta : float = None
) -> np.ndarray:
    """
    Construct bin edges around irregularly spaced times.

    This function generates a set of bin edges to be used for averaging
    higher-resolution data onto irregular PEIF (low-resolution) time centers.
    It uses the spacing between consecutive PEIF times to determine
    the "half-width" of each bin.

    Special handling is included for:
      - Abnormally large gaps: any time delta greater than `max_delta`
        (default: 600 s, i.e. ~50 s larger than the highest expected cadence)
        is replaced with the previous delta. This prevents bins that are
        unreasonably wide.
      - Edges: extra "fake" deltas are inserted at the beginning and end,
        so that binning around the first and last PEIF times can be done
        symmetrically without additional if-checks.
    
    IMPORTANT NOTE - If your s/c data has a changing time resolution (e.g
      like THEMIS-ESA Full going b/w 90s and 384s, then you should consider
      passing 
      
    FUTURE NOTE / TO DO:
        Make the max_delta determined based on past points in small window
        (e.g. last 10 points all had time delta 1min and now at point with
         2 min gap, so use local time_delta = 1min).

    Parameters
    ----------
    peif_times : np.ndarray
        1D sorted array of PEIF center times (e.g., in epoch seconds).
    max_delta_time_secs : float, optional
        Threshold for detecting anomalously large time steps. Defaults to 600.

    Returns
    -------
    np.ndarray
        1D array of bin edges of length len(peif_times) + 1.
        These edges can be passed to np.searchsorted to map
        high-res samples into PEIF bins.
    max_delta
        float, max_delta used to infer bins
    """
    irreg_times = np.asarray(irreg_times)
    if irreg_times.ndim != 1:
        raise ValueError("irreg_times_times must be 1D")

    # Compute consecutive differences between PEIF times
    time_diffs = np.diff(irreg_times)
    
    # assume slightly higher than median of time diffs if max_delta was not given
    if max_delta is None:
        max_delta = 1.1 * np.median(time_diffs)

    # Replace any "too large" deltas with the previous valid delta
    big_delta_time_idxs = np.where(time_diffs > max_delta)[0]
    for idx in range(len(big_delta_time_idxs)):
        prev_delta = time_diffs[big_delta_time_idxs[idx] - 1]
        time_diffs[big_delta_time_idxs[idx]] = prev_delta

    # Pad at the start and end so indexing is consistent
    time_diffs = np.insert(time_diffs, 0, time_diffs[0])   # duplicate first delta
    time_diffs = np.insert(time_diffs, -1, time_diffs[-1]) # duplicate last delta

    # Construct bin edges:
    # For each center time, left edge = center - previous/2, right edge = center + next/2
    edges = np.empty(len(irreg_times) + 1, dtype=irreg_times.dtype)
    for i in range(len(irreg_times)):
        lb = irreg_times[i] - time_diffs[i] / 2
        ub = irreg_times[i] + time_diffs[i + 1] / 2
        edges[i] = lb
        if i == len(irreg_times) - 1:
            edges[i + 1] = ub  # last edge
    return edges



def _binned_mean_with_nans(
        times: np.ndarray, 
        values: np.ndarray, 
        bin_edges: np.ndarray
) -> np.ndarray:
    """
    Compute averages of high-resolution values over irregular bins.

    This function takes a high-res time series (`times`, `values`) and a set of
    bin edges (typically from `make_bin_edges_with_deltas`), and computes
    averages of `values` inside each bin.

    Key details:
      - Works with either 1D values (time,) or N-D values (time, ...).
      - NaNs are ignored: bins containing NaNs still yield means over the
        non-NaN samples.
      - Empty bins (no samples) are filled with NaN.

    Implementation detail:
      Uses `np.add.reduceat` for speed. This is essentially a fast
      segmented sum/count along the time axis.

    Parameters
    ----------
    times : np.ndarray
        Sorted array of high-res time coordinates.
    values : np.ndarray
        High-res data aligned with `times`. Shape must be (time, ...) where
        the first dimension matches len(times).
    bin_edges : np.ndarray
        Bin edges of length N+1 defining N bins. Usually from
        `make_bin_edges_with_deltas`.

    Returns
    -------
    np.ndarray
        Array of averaged values with shape (n_bins, ...) where n_bins = len(bin_edges) - 1.
        For multi-dimensional inputs, all extra dimensions are preserved.
    """
    # Map bin edges into indices of the times array
    idx = np.searchsorted(times, bin_edges)

    # Only keep indices strictly less than len(times) (reduceat requirement)
    valid_idx = idx[:-1]
    valid_idx = valid_idx[valid_idx < len(times)]

    # Replace NaNs with 0.0 for summation, and build a mask of valid entries
    vals = np.nan_to_num(values, nan=0.0)
    mask = (~np.isnan(values)).astype(int)

    # Reduce sums and counts along the time axis
    sums = np.add.reduceat(vals, valid_idx, axis=0)
    counts = np.add.reduceat(mask, valid_idx, axis=0)

    # Compute means safely
    with np.errstate(divide="ignore", invalid="ignore"):
        means = sums / counts
    means[counts == 0] = np.nan

    # Pad result up to the expected number of bins (in case last bins were empty)
    out_shape = (len(bin_edges) - 1,) + values.shape[1:]
    out = np.full(out_shape, np.nan, dtype=values.dtype)
    out[:means.shape[0], ...] = means

    return np.asarray(out)



def avg_down_to_ds(
    # need time_dim and dataset for low time res
    low_res_ds       : xr.Dataset,
    low_res_time_dim : str,
    # need time_dim, dataset, and vars to avg down for high time res
    high_res_ds       : xr.Dataset,
    high_res_time_dim : str,
    high_res_vars     : list[str],
    max_time_delta    : float = None,
    use_tqdm : bool = False
) -> xr.Dataset:
    """
    Average down high-resolution xarray data to irregular low-res times.

    This is the orchestration function that ties everything together:
    - Takes a low-res dataset (with PEIF times).
    - Constructs appropriate bin edges from those times.
    - For each variable of interest in the high-res dataset:
        * Extracts its values
        * Calls `binned_mean_with_nans` to average down into PEIF bins
        * Wraps result back into an xarray.DataArray with preserved
          dimensions and coordinates.

    Parameters
    ----------
    peif_ds : xr.Dataset
        Dataset containing the PEIF (low-resolution) time coordinate.
    peif_time_dim : str
        Name of the PEIF time dimension/coordinate in `peif_ds`.
    high_res_ds : xr.Dataset
        Dataset containing high-resolution data to be averaged.
    high_res_time_dim : str
        Name of the time dimension in `high_res_ds`.
    high_res_vars : list[str]
        Names of variables in `high_res_ds` to average down.

    Returns
    -------
    xr.Dataset
        New Dataset with the same variables as `high_res_vars`, but averaged
        to the PEIF resolution. Each variable:
          - Has leading dimension `peif_time_dim`
          - Preserves any trailing dimensions (e.g., vector components)
          - Carries over variable attributes from `high_res_ds`
    """
    # Extract times
    low_res_times = low_res_ds[low_res_time_dim].compute().values
    high_res_times = high_res_ds[high_res_time_dim].compute().values

    # Build bin edges from PEIF times (with anomaly handling)
    edges = _make_bin_edges_with_deltas(
                    low_res_times, 
                    max_delta = max_time_delta
    )

    # Average each requested variable
    data_vars = {}
    for var in tqdm(high_res_vars, disable=not use_tqdm, desc="Averaging"):
        values = high_res_ds[var].compute().values
        means = _binned_mean_with_nans(high_res_times, values, edges)
        means = np.asarray(means)  # ensure proper numeric dtype

        # Reconstruct dimensions and coords
        var_dims = (low_res_time_dim,) + high_res_ds[var].dims[1:]
        var_coords = {low_res_time_dim: low_res_times}
        for extra_dim in high_res_ds[var].dims[1:]:
            var_coords[extra_dim] = high_res_ds[var].coords[extra_dim]

        data_vars[var] = xr.DataArray(
            means,
            dims=var_dims,
            coords=var_coords,
            name=var,
            attrs=high_res_ds[var].attrs,
        )

    return xr.Dataset(data_vars)



def avg_down_unif_grid(
    high_res_ds: xr.Dataset,
    high_res_time: str,
    high_res_vars: list[str],
    freq: str,
    new_time_name : str = None,
    use_tqdm : bool = False
) -> xr.Dataset:
    
    def to_epoch_seconds(dt64_arr):
        """Convert datetime64[ns] or float times to epoch seconds (float)."""
        if np.issubdtype(dt64_arr.dtype, np.datetime64):
            # Ensure numpy array, strip tz, get seconds since 1970-01-01
            epoch_start = np.datetime64("1970-01-01T00:00:00", "ns")
            return (dt64_arr.astype("datetime64[ns]") - epoch_start).astype("timedelta64[ns]").astype(np.int64) / 1e9
        else:
            return np.asarray(dt64_arr, dtype=float)
    
    # Extract times
    ds_times = high_res_ds[high_res_time].load().data
    
    # Case 1: datetime64 → use pandas directly
    if np.issubdtype(ds_times.dtype, np.datetime64):
        t_start = pd.to_datetime(ds_times[0]).floor(freq)
        t_end   = pd.to_datetime(ds_times[-1]).ceil(freq)
    else:
        # Assume float epoch seconds
        t_start = pd.to_datetime(ds_times[0], unit="s").floor(freq)
        t_end   = pd.to_datetime(ds_times[-1], unit="s").ceil(freq)
    
    # Build bin edges
    freq_delta = pd.to_timedelta(freq)
    unif_times = pd.date_range(start=t_start, end=t_end, freq=freq)
    time_edges = to_epoch_seconds(unif_times - freq_delta/2)
    unif_times = unif_times.to_numpy()[:-1]  # centers
    
    # Average variables
    data_vars = {}
    for var in tqdm(high_res_vars, disable=not use_tqdm, desc=f"Avg’ing down vars to {freq}"):
        values = high_res_ds[var].compute().values
        means = _binned_mean_with_nans(to_epoch_seconds(ds_times), values, time_edges)
        
        data_vars[var] = xr.DataArray(
            means,
            dims=(high_res_time,) + high_res_ds[var].dims[1:],
            coords={high_res_time: unif_times, **{d: high_res_ds[var].coords[d] for d in high_res_ds[var].dims[1:]}},
            name=var,
            attrs=high_res_ds[var].attrs,
        )
    
    if new_time_name is None:
        new_time_name = high_res_time
    return xr.Dataset(data_vars).rename({high_res_time: new_time_name})



def avg_down_unif_grid_OLD(
    high_res_ds: xr.Dataset,
    high_res_time: str,
    high_res_vars: list[str],
    freq: str,
    new_time_name : str = None,
    use_tqdm : bool = False
) -> xr.Dataset:
    """
    Average high-resolution xarray data onto a uniform time grid.

    For uniform bins, the edges are constructed from pandas.date_range
    spanning the data with +1 extra step, then shifted by half a frequency
    interval so bins are centered on the requested grid.
    
    BE AWARE THAT TIME OF GIVEN DATASET MAY NOT BE DATETIME64! SOMETIMES
    IS INT EPOCH (EITHER 1970, OR 2000)!!!

    PARAMETERS
    ----------
    high_res_ds : xr.Dataset
        Dataset containing high-resolution data to be averaged.
    high_res_time_dim : str
        Name of the time coordinate in `high_res_ds`.
    high_res_vars : list of str
        Variables from `high_res_ds` to average down.
    freq : str
        Frequency string understood by pandas (e.g. "1min", "30s").

    RETURNS
    -------
    xr.Dataset
        Dataset with averaged variables aligned to a uniform grid.
    """
    
    
    def npdt64_2_epochfloat(dt64_arr):
        return ( dt64_arr - pd.to_datetime('1970-01-01') ).total_seconds()
    
    
    # .load() to load all into memory and .data to directly access arr
    # (.values would make numpy arr as deepcopy!)
    ds_times = high_res_ds[high_res_time].load().data
    
    #####
    # HANDLE THIS PART BETTER - MESSES UP FOR EPOCH SOMETIMES!!!!!!
    #####
    t_start = pd.to_datetime(ds_times[0], unit='s').floor(freq)
    t_end   = pd.to_datetime(ds_times[-1], unit='s').floor(freq)

    # Frequency delta as Timedelta
    freq_delta = pd.to_timedelta(freq)

    # Construct bin edges (start -> end + one extra step)
    unif_times = pd.date_range(start=t_start, end=t_end + freq_delta, freq=freq)

    # edges as halfway between consec times - then converted to np.dt64
    time_edges = npdt64_2_epochfloat( (unif_times - freq_delta / 2) )

    # Chop off last time (needed for static array of time-edges) and
    # convert to numpy datetime64
    unif_times = unif_times[:-1].to_numpy()

    data_vars = {}
    for var in tqdm(high_res_vars, 
                    disable=not use_tqdm, 
                    desc=f"Avg\'ing down vars to {freq}"):
        
        ## get means of data according to time intervals
        values = high_res_ds[var].compute().values
        # make sure all times are raw epoch seconds to make blazing fast
        means = _binned_mean_with_nans(ds_times, values, time_edges)

        ## get appropriate dims and coords based on ds
        var_dims = (high_res_time,) + high_res_ds[var].dims[1:]
        var_coords = {high_res_time: unif_times}
        for extra_dim in high_res_ds[var].dims[1:]:
            var_coords[extra_dim] = high_res_ds[var].coords[extra_dim]

        ## turn avg'd down data into dataarray and add to dict
        data_vars[var] = xr.DataArray(
            means,
            dims   = var_dims,
            coords = var_coords,
            name   = var,
            attrs  = high_res_ds[var].attrs,
        )
    
    # change name of time str if desired
    if new_time_name is None: new_time_name = high_res_time
    
    return (
        # Convert dict of dataarrays into dataset ...
        xr.Dataset(data_vars)
        # ... and change default time dim name
            .rename({high_res_time : new_time_name})
    )



def avg_down_with_filters(
    ds: xr.Dataset,
    time_var: str,
    qflag_var: str,
    filter_rules: dict[str, Callable[[xr.DataArray], xr.DataArray]],
    freq: str | xr.DataArray | np.ndarray,
    new_time_name: str = "time",
    use_tqdm : bool = False
) -> xr.Dataset:
    """
    Apply variable-specific quality-flag filters and average down
    to either a **uniform** or **irregular** time grid.

    This function combines three steps:
      1. Apply per-variable filtering rules based on the quality flag.
      2. Average each filtered variable down onto a new time grid.
      3. Merge the results into a single dataset.

    -------------------------------------------------------------------
    PARAMETERS
    -------------------------------------------------------------------
    ds : xr.Dataset
        The input dataset containing science variables, a time variable,
        and a quality flag variable.

    time_var : str
        Name of the time variable in `ds` (e.g. "tha_peir_time").

    qflag_var : str
        Name of the quality flag variable in `ds`
        (e.g. "tha_peir_data_quality").
        The quality flag should be an integer bitmask array.

    filter_rules : dict[str, Callable[[xr.DataArray], xr.DataArray]]
        Dictionary mapping variable names (strings) to filtering functions.
        Each function should take the quality flag `DataArray` and return
        a boolean mask (True = keep, False = drop).
        
        Example:
            {
              "density":  lambda q: q < 64,   # keep unless maneuver
              "velocity": lambda q: q < 8,    # keep only if < 8
            }

    freq : str or xr.DataArray or np.ndarray
        Defines the output time grid.
        
        - If `str`: interpreted as a Pandas frequency string 
          (e.g. "1min", "30s"). A **uniform time grid** will be built
          spanning the data range.
          
        - If `xr.DataArray` or `np.ndarray`: treated as **explicit time centers**
          (e.g. from another dataset such as PEIF). In this case, bins
          will be constructed dynamically around these times.

    new_time_name : str, optional
        Name of the output time dimension. Defaults to "time".

    -------------------------------------------------------------------
    RETURNS
    -------------------------------------------------------------------
    xr.Dataset
        A dataset containing the selected variables:
          - Each variable is filtered according to its own rule.
          - Each variable is averaged onto the chosen time grid.
          - Results are merged into one dataset.
        Any bins that are entirely NaN (no surviving data) are dropped.

    -------------------------------------------------------------------
    NOTES
    -------------------------------------------------------------------
    - Memory efficiency: Only one variable is processed at a time. 
      This avoids loading the entire dataset into memory.
      
    - Flexibility: Different variables can tolerate different levels 
      of data quality. For example:
        * Densities may be valid even with certain flags set.
        * Velocities may require stricter filtering (e.g. q < 8).
        
    - Time grid choice:
        * Use a string (e.g. "1min") for uniform resampling.
        * Use another dataset's time variable for dynamic binning.

    -------------------------------------------------------------------
    EXAMPLES
    -------------------------------------------------------------------
    # Example 1: Uniform 1-minute grid
    filter_rules = {
        "tha_peir_density":  lambda q: q < 64,   # drop maneuvers only
        "tha_peir_velocity": lambda q: q < 8,    # stricter for velocity
    }

    esa_1min = avg_down_with_filters(
        ds=esa_ds,
        time_var="tha_peir_time",
        qflag_var="tha_peir_data_quality",
        filter_rules=filter_rules,
        freq="1min",
        new_time_name="time_1min"
    )

    # Example 2: Align to another dataset's irregular grid
    esa_dynamic = avg_down_with_filters(
        ds=esa_ds,
        time_var="tha_peir_time",
        qflag_var="tha_peir_data_quality",
        filter_rules=filter_rules,
        freq=esa_ds["tha_peif_time"],   # dynamic target grid
        new_time_name="time_peif"
    )
    """
    results = []

    for var, filt_func in tqdm(filter_rules.items(), 
                               disable=not use_tqdm, 
                               desc=f"Avg\'ing down vars to {freq}"):
        # Restrict dataset to just the variables we need:
        # - the science variable
        # - the quality flag
        # - the time variable
        ds_sub = ds[[var, qflag_var, time_var]]
        
        # Apply per-variable filtering (lazy: no immediate load)
        #mask = filt_func(ds_sub[qflag_var])
        #ds_masked = ds_sub.where(mask)
        
        raw_mask = filt_func(ds_sub[qflag_var])

        # Ensure mask is a DataArray with correct coords/dims
        mask = xr.DataArray(
            np.asarray(raw_mask),
            coords={time_var: ds_sub[time_var]},
            dims=[time_var]
        )
        
        ds_masked = ds_sub.where(mask)

        # Case 1: uniform time grid (freq is str)
        if isinstance(freq, str):
            ds_down = avg_down_unif_grid(
                high_res_ds=ds_masked,
                high_res_time=time_var,
                high_res_vars=[var],
                freq=freq,
                new_time_name=new_time_name,
                use_tqdm = False
            )
        
        # Case 2: irregular/dynamic grid (freq is array of times)
        else:
            # Wrap freq into a dataset with a time coordinate
            if isinstance(freq, xr.DataArray):
                target_times = freq
            else:
                target_times = xr.DataArray(freq, dims=["target_time"])
            
            ds_down = avg_down_to_ds(
                # low res ds params
                low_res_ds       = target_times.to_dataset(name="target_time"),
                low_res_time_dim = "target_time",
                # high res ds params
                high_res_ds       = ds_masked,
                high_res_time_dim = time_var,
                high_res_vars     = [var],
                # dsiable tqdm bar
                use_tqdm = False
            ).rename({"target_time": new_time_name})

        results.append(ds_down)

    # Merge all variables together
    merged = xr.merge(results, compat="no_conflicts")

    # Drop time bins where all variables are NaN
    merged = merged.dropna(dim=new_time_name, how="all")

    return merged



def interp(
    target_ds: xr.Dataset,
    target_time: str,
    source_ds: xr.Dataset,
    source_time: str,
    vars_to_interp: list[str] = None,
    suffix: str | None = None,
    chunks: dict | None = None,
) -> xr.Dataset:
    """
    Interpolate selected variables from one dataset onto the timebase of another.
    
    There are times where interpolation can be reasonable (e.g. S/C position
    at 1min res onto wave data at 1s res, or multi-instrument observations
    made at spin resolution but 90 degrees out of phase)... but there are
    also many, many times that interpolation would be unwise.
    
    -- Think prudently before blindly interpolating! --

    PARAMETERS
    ----------
    source_ds : xr.Dataset
        Dataset containing variables to interpolate.
    target_ds : xr.Dataset
        Dataset providing the target time coordinate.
    source_time : str
        Name of the time coordinate in source_ds.
    target_time : str
        Name of the time coordinate in target_ds.
    vars_to_interp : list of str
        Variables from source_ds to interpolate.
    suffix : str, optional
        If provided, appended to variable names in the returned dataset
        (e.g., "_interp").
    chunks : dict, optional
        If provided, rechunk `source_ds` before interpolation to control
        memory usage (e.g., {"Epoch": 500_000}).

    RETURNS
    -------
    xr.Dataset
        Dataset containing interpolated variables aligned to target_time.
    """

    if vars_to_interp is None:
        raise ValueError("vars_to_interp must be provided")

    # Optionally rechunk source dataset for performance
    if chunks is not None:
        source_ds = source_ds.chunk(chunks)

    # Ensure we have the target coordinate values
    target_time_vals = target_ds[target_time]

    # Perform interpolation
    interp = source_ds[vars_to_interp].interp({source_time: target_time_vals}, method="linear")

    # Handle dimension/coord naming conflicts
    if source_time in interp.dims and source_time != target_time:
        interp = interp.swap_dims({source_time: target_time})
    elif source_time in interp.coords and source_time != target_time:
        # Already aligned, just drop the duplicate coord
        interp = interp.drop_vars(source_time)

    # Optionally rename variables with suffix
    if suffix is None:
        suffix = ""
    if suffix:
        interp = interp.rename({v: f"{v}{suffix}" for v in vars_to_interp})

    return interp