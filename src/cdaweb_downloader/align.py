# -*- coding: utf-8 -*-



import numpy as np
import xarray as xr



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
    max_delta = np.median(time_diffs)

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
    means = sums / counts
    means[counts == 0] = np.nan

    # Pad result up to the expected number of bins (in case last bins were empty)
    out_shape = (len(bin_edges) - 1,) + values.shape[1:]
    out = np.full(out_shape, np.nan, dtype=values.dtype)
    out[:means.shape[0], ...] = means

    return np.asarray(out)



def avg_down(
    # need time_dim and dataset for low time res
    low_res_ds       : xr.Dataset,
    low_res_time_dim : str,
    # need time_dim, dataset, and vars to avg down for high time res
    high_res_ds       : xr.Dataset,
    high_res_time_dim : str,
    high_res_vars     : list[str],
    max_time_delta    : float = None
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
    for var in high_res_vars:
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



def interp_to_target_times(
        # dataset and time dim of the target times
        target_ds : xr.Dataset,
        target_time : str,
        # dataset, time_dim, and list of data_vars of source
        source_ds : xr.Dataset,
        source_time : str,
        vars_to_interp : list[str] = None,
        suffix: str | None = None,
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

    RETURNS
    -------
    xr.Dataset
        Dataset containing interpolated variables aligned to target_time.
    """
    
    # Ensure we have the target coordinate values
    target_time_vals = target_ds[target_time].load().values
    
    # Interpolate onto target time axis
    interp = (
                # Select only requested vars
                source_ds[vars_to_interp]
                    # interpolate from target times onto source times
                    .interp({source_time : target_time_vals})
                    # Rename time coord to match target
                    .interp.rename({source_time: target_time})
    )
    
    # Optionally rename variables with suffix
    if suffix is None: suffix = ""
    interp = interp.rename({v: f"{v}{suffix}" for v in vars_to_interp})
        
    return interp