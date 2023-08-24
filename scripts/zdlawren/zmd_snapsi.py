""" This python script generates a zonal mean dataset for a 
specified set of SNAPSI data. It takes required commandline 
args that specify the source_id (model name), 
sub_experiment_id (model init), and experiment_id -- in that 
order -- which are used to query the SNAPSI intake catalog. 

This script uses pyzome (https://github.com/zdlawrence/pyzome)
to compute the zonal mean datasets. 

Currently the script iterates over ensemble members and 
outputs files for these individually. In principle, the 
pyzome operations should work lazily on all ensemble members, but 
I ran into issues with my SLURM jobs being killed when I tried 
to keep all ensemble members together.  

Original Author: Z. D. Lawrence
Last modified: 2023-07-18
""" 

import argparse
import pathlib
from datetime import datetime

import intake
import numpy as np
import xarray as xr

from pyzome.recipes import create_zonal_mean_dataset


def is_zmd_file_bad(path_to_file):
    """ A brute force function to tell if a zmd file 
    is bad. If a SLURM job doesn't complete in time, 
    a zmd file may be written to disk filled with nans.
    Here we simply check each of the fields, and throw 
    an error if any has more nans than half its full size.
    This is done for the try except in the script below
    so that the file can be remade.
    """

    tmp = xr.open_dataset(path_to_file)
    if len(tmp.data_vars) < 17:
        raise ValueError("zmd file does not have enough fields")
    
    for key, field in tmp.data_vars.items():
        nbad = np.isnan(field).sum().data
        total_size = field.nbytes/field.dtype.itemsize
        if nbad > total_size/2:
            raise ValueError("Field has too many nans")


# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="source_id")
parser.add_argument("subexperiment", type=str, help="sub_experiment_id")
parser.add_argument("experiment", type=str, help="experiment_id")
args = parser.parse_args()

# Pull args into variables for convenience
source_id = args.model
experiment_id = args.experiment
sub_experiment_id = args.subexperiment

# Open the Intake catalog and subset to the specific model, init, and experiment
catalog = intake.open_esm_datastore("/gws/nopw/j04/snapsi/test-snapsi-catalog-fast.json")
subset = catalog.search(
    variable_id=["ua", "va", "ta", "zg", "wap"],
    source_id=source_id, 
    sub_experiment_id=sub_experiment_id,
    experiment_id=experiment_id
)

# Setup the output directory
output_dir = pathlib.Path(f"/work/scratch-nopw2/zdlawren/zmd/{source_id}/{sub_experiment_id}/{experiment_id}/")
output_dir.mkdir(parents=True, exist_ok=True)

# Convert our query into a dataset and rename data_vars 
# to names that pyzome expects
ds = subset.to_dask(xarray_open_kwargs={"engine":"h5netcdf"})
print(ds)
ds = ds.rename({"ua":"u", "va":"v", "wap":"w", "ta":"T", "zg": "Z"})
if 'lat_bnds' in ds.coords:
    ds = ds.drop('lat_bnds')
if 'lon_bnds' in ds.coords:
    ds = ds.drop('lon_bnds')
if 'sub_experiment_id' in ds.coords:
    ds = ds.isel(sub_experiment_id=0)
    ds = ds.drop('sub_experiment_id')

# some models named their pressure level coordinate "snap34"
if ('snap34') in ds.coords:
    ds = ds.rename({"snap34": "plev"})

# Iterate over ensemble members
for member in ds.member_id.values:
    
    # Set up the output name/path
    output_file = f"{source_id}_{sub_experiment_id}_{experiment_id}_{member}_zmd.nc"
    output_path = f"{str(output_dir)}/{output_file}"
    
    # Check if file already exists and if it can be read as a first order
    # handler for re-running jobs that didn't finish in time before
    if pathlib.Path(output_path).exists() is True:
        try:
            is_zmd_file_bad(output_path)
        except Exception:
            print(f"{output_path} already exists but has issues ... Remaking")
            pass
        else:
            print(f"{output_path} already exists and is complete! Skipping ...")
            continue
    
    print(f"Now working on {member} for {source_id} {experiment_id} {sub_experiment_id}")
    zmd = create_zonal_mean_dataset(
        ds.sel(member_id=member),
        verbose=True, 
        include_waves=True, 
        waves=[1,2,3], 
        fftpkg="xrft"
    ) 

    if "zonal_wavenum" in zmd.coords:
        zmd = zmd.rename({"zonal_wavenum": "wavenum_lon"})
    
    # Set up encoding dictionary to ensure everything gets saved as
    # float32 (with no compression to ensure dask chunking can be used on output)
    comp = dict(dtype="float32")
    encoding = {var: comp for var in zmd.data_vars}    

    # Since we're using dask for everything, no actual computations 
    # are done until the to_netcdf method is called
    print(f"Saving to {output_path}")
    zmd.attrs['history'] = f'Created by zdlawren using pyzome on {datetime.utcnow()}'
    try:
        zmd.to_netcdf(output_path, encoding=encoding)
    except Exception as e:
        print(f"(ERROR) Unable to complete zmd file for {member} of {source_id} {experiment_id} {sub_experiment_id}")
        print(f"(ERROR) Exception: {e}")
