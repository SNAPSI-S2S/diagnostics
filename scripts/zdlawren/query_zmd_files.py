import os
import argparse
import pathlib
import numpy as np
import xarray as xr

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="source_id")
parser.add_argument("--compile_complete", action="store_true", help="compile complete set of files into individual")
parser.add_argument("--clean_partial", action="store_true", help="cleanup partial files from jobs that died early")
args = parser.parse_args()

dates = ["s20180125", "s20180208", "s20181213", "s20190108", "s20190829", "s20191001"]
exps = ["nudged","control","free"]
if args.model in {"GloSea6", "IFS"}:
    exps += ["nudged-full","control-full"]
elif args.model in {"CNRM-CM61"}:
    exps += ["nudged-full"]
elif args.model == "SPEAR":
    exps = ["free", "nudged-full"]

dat_path = pathlib.Path("/work/scratch-nopw2/zdlawren/zmd")
paths = [dat_path / args.model / f"{date}/{exp}" for date in dates for exp in exps]

for path in paths:
    if path.is_dir() is False:
        print(f"{path} does not exist")
        continue
    nc_files = list(path.glob("*.nc"))
    num_nc_files = len(nc_files)
    print(f"{path} -> {num_nc_files} nc files")
    
    if num_nc_files == 0:
        continue
    fi_sizes = [ncfi.stat().st_size/(1024*1024) for ncfi in nc_files]
    max_fi_size = np.max(fi_sizes)

    non_full = np.where(fi_sizes != max_fi_size)[0]
    for ix in non_full:
        print(f"\t{nc_files[ix].stem} only {fi_sizes[ix]} MiB")
        if args.clean_partial is True: 
            print(f"\tRemoving {nc_files[ix].stem}")
            nc_files[ix].unlink() 
            num_nc_files -= 1

    if (args.compile_complete is True) and (num_nc_files >= 50) and (len(non_full) == 0):
        init = path.parts[6]
        experiment = path.parts[7]

        output_file = pathlib.Path(f"/gws/nopw/j04/snapsi/processed/{args.model}/{experiment}/{init}/zonal_means/{args.model}_{experiment}_{init}_zonalmeans.nc")
        
        print(f"\t(compile_complete=True) Now compiling final dataset for {args.model} {experiment} {init}")
        if (output_file.exists() is False) or (np.abs(output_file.stat().st_size/(1024*1024) - np.sum(fi_sizes)) >= 15):
            output_file.parent.mkdir(parents=True, exist_ok=True)
            mfds = xr.open_mfdataset(nc_files, combine="nested", concat_dim="member_id")
            print(f"\t(compile_complete=True) Writing compiled dataset to {output_file}")
            mfds.to_netcdf(output_file)
            mfds = None
        else:
            print(f"\t(compile_complete=True) {output_file} already exists; skipping!")
       
