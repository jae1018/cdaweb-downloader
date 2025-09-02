#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import xarray as xr
import warnings
import numpy as np

from .logger import logger



def _track_first_dim(
        ds_list: list[xr.Dataset]
) -> dict[str, str]:
    
    """
    Creates a list of dicts wherein each key is a data variable in
    ds_list[i] and the value is the first dim name
    
    PARAMETERS
    -----------
    ds_list - list of xarray datasets
    
    RETURNS
    -------
    dict[str, str]
    """
    
    first_dim_list = []
    for ds in ds_list:
        dvs_to_first_dim = {}
        for var in ds.data_vars:
            dvs_to_first_dim[var] = ds[var].dims[0]
        first_dim_list.append( dvs_to_first_dim )
    return first_dim_list



def _safe_rename_dims(
        ds: xr.Dataset,
        renames_list: list[ dict[str, str] ]
) -> xr.Dataset:
    """
    Safely renames the dims of a dataset. This is only needed in the rare
    cases where dims need to actually be swapped (e.g. an original reference
    may be that two important dims are record0 and record1, but due to a CDF
    error, record1 and record0 are accidentally flipped

    Parameters
    ----------
    ds : xarray dataset
        The dataset to be renamed
    renames_list : list[ dict[str, str] ]
        A list of *single-ton* dicts (single-ton so that swapped dims don't
        overwrite, e.g. {'record0':'record1', 'record1':'record0'} would just
        cancel out ot {'record0':'record0'}!).

    Returns
    -------
    xarray dataset with renamed dims
    """
    
    # This should be fixed later! but don't ahve time rn
    # LAter, should turn renames_list into two mutually exclusive lists:
    # One that contains mappings without collisions, and another that contains
    # dims that need to be swapped (could be A->B->A or A->B->C->A or cycles
    # like that). For now, just *very naively* turn into cumulative dict and
    # apply rename
    # for warnings later whe  fixing tis
    cumulative = {}
    for d in renames_list:
        for k, v in d.items():
            if k in cumulative and cumulative[k] != v:
                warnings.warn(f"Conflict for {k}: {cumulative[k]} vs {v}")
                
    # naive way for now
    cumulative = {k: v for d in renames_list for k, v in d.items()}
    return ds.rename(cumulative)
    



def _enforce_consistent_first_dim_name(
        ds_list: list[xr.Dataset]
) -> list[xr.Dataset]:
    
    """
    Ensure that the first dim names for each data var are consistent in
    naming across all datasets in ds_list (the first dataset in the list
    is used as reference).
    
    PARAMETERS
    ----------
    ds_list - list of xarray datasets
    
    RETURNS
    -------
    list of dims-renamed xarray datasets
    """
    
    # Extract the mappings from data vars to first dim name
    first_dim_list = _track_first_dim(ds_list)
    
    # consider first dataset in ds_list as prime reference
    ref_dict = first_dim_list[0]
    
    # For each dataset in ds_list ...
    for i in range(len(ds_list)):
        ds = ds_list[i]
        rename_list = []
        
        # ... check first dim for each data var ...
        for var in ds.data_vars:
            
            # sometimes a dataset may having missing data (and hence is
            # missing this data variable) - in that case, just continue
            if var not in ref_dict:
                continue
            
            # if the first dim for the data var and the ref_dict, don't match,
            # then rename the first dim in the dataset ds to that of the ref_dict
            if ds[var].dims[0] != ref_dict[var]:
                rename_list.append( {ds[var].dims[0] : ref_dict[var]} )
        
        # ... and safely reassign dims (although saving back to just ds might 
        # be fine here too?)
        ds_list[i] = _safe_rename_dims(ds, rename_list)
    
    return ds_list



def _remove_datasets_with_bad_dims(
        ds_list : list[ xr.Dataset ]
) -> list[ xr.Dataset ]:
    """
    Removes datasets from ds_list whose dims do not match the expected set 
    of dims (expected meaning the union of all dims of the datasets).
    
    PARAMETERS
    ----------
    ds_list - list of xarray datasets
    
    RETURNS
    -------
    filtered ds_list (with offending datasets removed)
    """
    
    # get union of all dims
    union_dims = set().union(*[set(ds.dims) for ds in ds_list]) if ds_list else set()
    
    # get idxs of datasets in ds_list where dims match union_dims
    dims_setdiff_per_ds = [ np.setdiff1d( union_dims, set(ds.dims) ) for ds in ds_list ]
    matching_dims_idxs = np.where( [ 
            len(dims_setdiff_per_ds[i]) == 0 for i in range(len(dims_setdiff_per_ds)) 
    ] )[0]
    
    # log info on if any datasets were removed
    logger.info(
        f'{len(ds_list) - matching_dims_idxs.size} datasets found with '
        'not-matching dimensions - removing these from time concatenation.'
    )
    
    return [ ds_list[i] for i in range(len(ds_list)) if i in matching_dims_idxs ]



def align_datasets_over_time_dims(
        ds_list: list[xr.Dataset]
) -> list[xr.Dataset]:
        
    """
    Aligns the given list of datasets by their time dims. The aligned
    datasets are returned as a mutually-exclusive list where each
    resulting dataset is aliged for a particular time dim (i.e. in the
    case of datasets where different time-resolution data are available).
    
    PARAMETERS
    ----------
    ds_list - list of xarray datasets
        Each dataset in this list might contain multiple time vars. Some,
        due to CDF issues, might also be missing some or all of their 
        corresponding time dims
    
    RETURNS
    -------
    list of xarray datasets
        These datasets are aligned per time var (so if there are 3 time
        resolutions, this list will contain 3 datasets).
    """
    
    # First enforce that all datasets in ds_list have similar first_dim
    # names (based on first dataset in list)
    ds_list = _enforce_consistent_first_dim_name(ds_list)
    
    # Assume first dataset in list contains correct time info
    # This WILL NOT BE CORRECT IF THIS FIRST DATASET CONTAINS
    # MISSING TIME DATA!!!!
    time_dims = list(set(
        # get values from dict, then convert to set to get unqiues
        _track_first_dim(ds_list[:1])[0].values()
    ))
    
    # for each time name, need to build list of datasets where all other time
    # names are dropped
    combined_datasets = []
    for time_dim in time_dims:
        other_time_dims = list( set(time_dims) - set([time_dim]) )
        
        # build list of datasets, all only containing data involving
        # current time_dim in for loop
        ##ds_list_single_time_dim = [ ds.drop_dims(other_time_dims, errors="ignore") 
        ##                            for ds in ds_list ]
        
        ## If time_dim has length 0 (but other variables hve non-zero length),
        ## then drop.dims(other_time_dims) will throw error - fix better later!
        ds_list_single_time_dim = []
        for ds in ds_list:
            
            # throw away dataset if time_dim has length 0
            if ds[time_dim].size == 0:
                continue
            
            # Otherwise, drop dims and keep
            else:
                ds_list_single_time_dim.append(
                    ds.drop_dims(other_time_dims, errors="ignore") 
                )
                
        
        # purge datasets from list if dims don't match
        ds_list_single_time_dim = _remove_datasets_with_bad_dims(
                                        ds_list_single_time_dim
        )
        
        # concat dataset along time_dim then save to list
        combined_datasets.append( 
            xr.concat(ds_list_single_time_dim, 
                      dim    = time_dim, 
                      coords = "minimal", 
                      join   = "override")
        )
    
    # Merge all concatenated datasets into one
    return xr.merge(combined_datasets, compat='no_conflicts')
    