""" This python script automatically generates a series of 
bash scripts for computing zonal mean datasets of the SNAPSI
data, which can be submitted thru SLURM with sbatch. 

It requires commandline arguments specifying the 
source_id (model name) and sub_experiment_id (init date) as
targets; from there it will generate scripts for every available 
experiment listed in the SNAPSI intake catalog (nominally, 
at least 'control', 'nudged', and 'free')

This is sort of a kludgey way to accomplish this task, but I 
am not very familiar with SLURM and how to pass specific args 
to scripts.

Original Author: Z. D. Lawrence
Last modified: 2023-08-10
"""

import os
import re
import time
import pathlib
import argparse
import intake

def is_valid_duration(duration):
    """ Check for a valid job duration string that 
    will be accepted by SLURM. Format of HH:MM:SS
    """
    pattern = r"^([0-9]{2}):([0-9]{2}):([0-9]{2})$"
    match = re.match(pattern, duration)
    if match is None:
        return False 
    hours, mins, secs = int(match.group(1)), int(match.group(2)), int(match.group(3))
    return (0 <= hours < 24) and (0 <= mins < 60) and (0 <= secs < 60)


def is_valid_memsize(memsize):
    """ Check for a valid memsize string that 
    will be accepted by SLURM. Format of {num}G
    to specify the number of gigabytes. num must 
    be less than 100 (this is arbitrary).
    """
    pattern = r"^([0-9]+)G$"
    match = re.match(pattern, memsize)
    if match is None:
        return False
    memory_gb = int(match.group(1))
    return memory_gb < 100


USER_HOME = pathlib.Path("~/").expanduser()

# Default locations - can modify these for your setup!
# The default python location in particular is specific to zdlawren's setup!
DEFAULT_OUTPUT_DIR = USER_HOME / "autoscripts"
DEFAULT_PYTHON_LOC = USER_HOME / "mambaforge/envs/py311/bin/python" 
DEFAULT_ZMD_LOC = USER_HOME / "snapsi-diagnostics/scripts/zdlawren/zmd_snapsi.py"

# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="source_id")
parser.add_argument("subexperiment", type=str, help="sub_experiment_id")
parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTPUT_DIR))
parser.add_argument("--python", type=str, default=str(DEFAULT_PYTHON_LOC))
parser.add_argument("--zmd", type=str, default=str(DEFAULT_ZMD_LOC))
parser.add_argument("--submit", action="store_true", help="submit with sbatch")
parser.add_argument("--wait", type=int, default=5)
parser.add_argument("--mem", type=str, default="20G", help="memory allocation for job in format of [num]G to specify the number of GB")
parser.add_argument("--timelimit", type=str, default="06:00:00", help="duration of job in format of HH:MM:SS")
args = parser.parse_args()

# First make some checks that would stop the script from running
# if there are issues with the commandline args
if is_valid_duration(args.timelimit) is False:
    msg = f"'{args.timelimit}' is not a valid timelimit; needs HH:MM:SS"
    raise ValueError(msg)

if is_valid_memsize(args.mem) is False:
    msg = f"'{args.mem}' is not a valid memsize; needs [num]G where [num] is an int < 100"
    raise ValueError(msg)

valid_models = {"GLOBO", "GEM-NEMO", "GloSea6-GC32", "GRIMs", "GloSea6", "CNRM-CM61"}
if args.model not in valid_models:
    msg = f"First arg '{args.model}' must be one of {valid_models}"
    raise ValueError(msg)

valid_subexps = ("s20180125", "s20180208", "s20181213", "s20190108", "s20190829", "s20191001")
if args.subexperiment not in valid_subexps:
    msg = f"Second arg '{args.subexperiment}' must be one of {valid_subexps}"
    raise ValueError(msg)

# iff the output directory is the default, let's ensure we have a location to 
# put the scripts. only do this for the default location
if args.outdir == str(DEFAULT_OUTPUT_DIR):
    DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)

# Open the Intake catalog and subset on the source_id and subexperiment
catalog = intake.open_esm_datastore("/gws/nopw/j04/snapsi/test-snapsi-catalog-fast.json")
subset = catalog.search(
    variable_id=["ua", "va", "ta", "zg", "wap"],
    source_id=args.model,
    sub_experiment_id=args.subexperiment,
)

# Get list of available experiment IDs, and iterate over them
experiment_ids = subset.df.experiment_id.unique()
for eid in experiment_ids:
    # Auto-genned script name
    scripts_fi = f"{args.model}_{args.subexperiment}_{eid}.sh"
    
    # Content that goes into the file
    print(f"Generating {args.outdir}/{scripts_fi}")
    text_for_script = "#!/bin/bash\n\n"
    text_for_script += f"#SBATCH --job-name=\"{args.model}_{args.subexperiment}_{eid}\"\n"
    text_for_script += f"#SBATCH --mem={args.mem}\n"
    text_for_script += f"#SBATCH --time={args.timelimit}\n\n"
    text_for_script += f"{args.python} -u {args.zmd} {args.model} {args.subexperiment} {eid}\n"

    # Write string to file and chmod it for use
    with open(f"{args.outdir}/{scripts_fi}", "w") as fi:
        fi.write(text_for_script)
    chmod =  f"chmod u+x {args.outdir}/{scripts_fi}"
    os.system(chmod)

    # Submit the jobs with sbatch if desired
    if args.submit is True:
        cmd = f"sbatch {args.outdir}/{scripts_fi}"
        print(f"Submitting `{cmd}`")
        os.system(cmd)
        time.sleep(args.wait)
