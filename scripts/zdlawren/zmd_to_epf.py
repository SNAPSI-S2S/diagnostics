"""This python script converts data from zonal mean datasets
of the SNAPSI output into files containing the Eliassen-Palm
flux components.

Original Author: Z. D. Lawrence
Last modified: 2023-08-14
"""

import sys
import argparse
import pathlib
from datetime import datetime

import xarray as xr
from pyzome import tem

DATA_ROOT = pathlib.Path("/gws/nopw/j04/snapsi/processed")


def parse_commandline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="")
    parser.add_argument("--clobber", action="store_true", help="Overwrite old EP-flux files")
    return parser.parse_args()


def main():
    args = parse_commandline_args()

    zmd_files = sorted(list(DATA_ROOT.glob(f"**/{args.model}*zonalmeans.nc")))
    if len(zmd_files) == 0:
        print(f"(ERROR) No zonal mean dataset files found for {model}")
        sys.exit(1)

    for fi in zmd_files:
        # Set up the output file name from the input file name
        output_file = pathlib.Path(
            str(fi).replace("zonal_means", "ep_fluxes").replace("zonalmeans", "epfluxes")
        )

        # Skip if file already exists and we're not clobbering
        if args.clobber is False and output_file.exists():
            print(f"{output_file} already exists; skipping")
            continue

        # Otherwise, setup the output path
        output_path = output_file.parent
        output_path.mkdir(exist_ok = True)
        print(f"Converting {fi} to {output_file}")

        # Open the zonal mean dataset files, and keep only what we need:
        # zonal winds, temps, and eddy fluxes. Then compute EP fluxes
        zmd = xr.open_dataset(fi)[["u", "T", "uv", "vT", "uw", "uv_k", "vT_k", "uw_k"]]
        if args.model == "era5":
            zmd = zmd.rename({"pres":"plev","zonal_wavenum":"wavenum_lon"})
            zmd["plev"].attrs["units"] = "Pa"
        epfy, epfz = tem.epflux_vector(zmd.u, zmd.T, zmd.uv, zmd.vT, zmd.uw)
        epfy_k, epfz_k = tem.epflux_vector(zmd.u, zmd.T, zmd.uv_k, zmd.vT_k, zmd.uw_k)

        # Add attributes for the data being saved to netcdf
        epfy.name = "epfy"
        epfy.attrs["long_name"] = "meridional component of EP flux"

        epfz.name = "epfz"
        epfz.attrs["long_name"] = "vertical component of EP flux"

        epfy_k.name = "epfy_k"
        epfy_k.attrs["long_name"] = "meridional component of EP flux due to zonal wavenumber k"

        epfz_k.name = "epfz_k"
        epfz_k.attrs["long_name"] = "vertical component of EP flux due to zonal wavenumber k"

        # Collect the fields into an xarray Dataset
        epf_ds = xr.merge([epfy, epfz, epfy_k, epfz_k], combine_attrs="drop")

        # Set up encoding dictionary to ensure we use float32 with light compression
        comp = dict(dtype="float32", zlib=True, complevel=3)
        encoding = {var: comp for var in epf_ds.data_vars}

        print(f"Saving to {output_file}")
        epf_ds.attrs['history'] = f'Created by zdlawren using pyzome on {datetime.utcnow()}'
        try:
            epf_ds.to_netcdf(output_file, encoding=encoding)
        except Exception as e:
            print(f"(ERROR) Unable to complete EP-flux file for {member} of {source_id} {experiment_id} {sub_experiment_id}")
            print(f"(ERROR) Exception: {e}")

            # Remove any file that is partially created/written, for whatever reason
            if output_file.exists():
                output_file.unlink()

if __name__ == "__main__":
    main()
